#!/usr/bin/env python3
"""FPS ablation on source-held-out 5-fold CV (novel-video eval).

Subsampling simulates lower extract FPS by keeping every *n*th frame per clip from
30 fps parquets (same as ``fps_sensitivity.py``).

For FPS ablation charts, prefer a **fixed** decision threshold (default 0.24, matching
the prior ``fps_sensitivity.py`` protocol). Per-FPS threshold tuning can mask quality
loss by shifting the cutoff (e.g. 0.27 @ 30 fps → ~0.20 @ 1 fps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_TABULAR_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(_TABULAR_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.metrics import fbeta_score  # noqa: E402

from fold_split import (  # noqa: E402
    DEFAULT_FOLDS_CSV,
    filter_to_fold_sources,
    load_source_folds,
    split_frames_by_fold,
)
from fps_sensitivity import (  # noqa: E402
    BASE_FPS,
    DEFAULT_DECISION_THRESHOLD,
    stride_for_target_fps,
    subsample_frames_within_clips,
)
from train import (  # noqa: E402
    _DEFAULT_RUNS_ROOT,
    active_feature_columns,
    load_all_frames,
    resolve_run_dir,
    train_and_evaluate,
)

DEFAULT_TARGET_FPS = (30.0, 10.0, 5.0, 2.0, 1.0)
DEFAULT_OUT_DIR = REPO_ROOT / "models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold/fps_ablation_novel"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=_DEFAULT_RUNS_ROOT)
    p.add_argument("--feature-run-id", type=str, default="all_clips_features_v2")
    p.add_argument("--folds-csv", type=Path, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--target-fps", type=float, nargs="+", default=list(DEFAULT_TARGET_FPS))
    p.add_argument("--base-fps", type=int, default=BASE_FPS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--f-beta", type=float, default=2.0)
    p.add_argument("--feature-subset", choices=("all", "base"), default="all")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Fixed threshold for primary chart metric (default 0.24, prior ablation).",
    )
    p.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also report per-FPS tuned threshold (can mask FPS loss; default off).",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument(
        "--plot-metric",
        choices=("fixed", "tuned"),
        default="fixed",
        help="Which F2 series to plot (default: fixed threshold).",
    )
    p.add_argument("--save-plot", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _f2_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    y_pred = (probs >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))


def run_novel_fps_cv(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_run_dir(args)
    feature_columns = active_feature_columns(args.feature_subset)
    all_df, manifest = load_all_frames(run_dir, feature_columns)
    source_folds = load_source_folds(args.folds_csv)
    cv_df = filter_to_fold_sources(all_df, source_folds)
    fold_ids = sorted(set(source_folds.values()))
    fixed_thr = float(args.decision_threshold)

    fps_results: list[dict[str, Any]] = []
    for target_fps in sorted(set(args.target_fps), reverse=True):
        stride = stride_for_target_fps(target_fps, base_fps=args.base_fps)
        fold_rows: list[dict[str, Any]] = []
        oof_frames: list[pd.DataFrame] = []

        for fold_id in fold_ids:
            train_df, val_df = split_frames_by_fold(cv_df, fold_id, source_folds)
            train_sub = subsample_frames_within_clips(train_df, stride)
            val_sub = subsample_frames_within_clips(val_df, stride)

            # Fixed threshold (primary ablation protocol)
            fixed_args = argparse.Namespace(**vars(args))
            fixed_args.tune_threshold = False
            fixed_args.decision_threshold = fixed_thr
            report_fixed, val_preds, _ = train_and_evaluate(
                train_sub,
                val_sub,
                feature_columns=feature_columns,
                args=fixed_args,
                manifest=manifest,
                eval_split_name=f"fold_{fold_id}",
                split_method="5fold_cv_fps",
            )
            mf = report_fixed["metrics"]

            row: dict[str, Any] = {
                "fold_id": fold_id,
                "val_source_ids": sorted(
                    sid for sid, fid in source_folds.items() if fid == fold_id
                ),
                "frame_stride": stride,
                "n_val_rows": len(val_sub),
                "decision_threshold": fixed_thr,
                "f2": float(mf["f2"]),
                "recall": float(mf["recall"]),
                "precision": float(mf["precision"]),
            }

            if args.tune_threshold:
                tuned_args = argparse.Namespace(**vars(args))
                tuned_args.tune_threshold = True
                tuned_args.decision_threshold = None
                report_tuned, _, _ = train_and_evaluate(
                    train_sub,
                    val_sub,
                    feature_columns=feature_columns,
                    args=tuned_args,
                    manifest=manifest,
                    eval_split_name=f"fold_{fold_id}",
                    split_method="5fold_cv_fps",
                )
                mt = report_tuned["metrics"]
                row["f2_tuned"] = float(mt["f2"])
                row["recall_tuned"] = float(mt["recall"])
                row["precision_tuned"] = float(mt["precision"])
                row["tuned_threshold"] = report_tuned["decision_threshold"]

            fold_rows.append(row)
            oof_frames.append(val_preds)

        oof = pd.concat(oof_frames, ignore_index=True)
        yt = oof["is_playing"].astype(int).to_numpy()
        probs = oof["pred_prob_playing"].astype(float).to_numpy()
        yp_fixed = (probs >= fixed_thr).astype(int)

        pooled_f2_fixed = _f2_at_threshold(yt, probs, fixed_thr)
        val_all = subsample_frames_within_clips(cv_df, stride)
        y_val = val_all["is_playing"].astype(int).to_numpy()
        baseline_f2 = float(
            fbeta_score(y_val, np.ones(len(y_val), dtype=int), beta=args.f_beta, zero_division=0)
        )

        fold_f2s = [r["f2"] for r in fold_rows]
        entry: dict[str, Any] = {
            "target_fps": target_fps,
            "frame_stride": stride,
            "decision_threshold": fixed_thr,
            "fold_f2_mean": float(np.mean(fold_f2s)),
            "fold_f2_std": float(np.std(fold_f2s, ddof=1)) if len(fold_f2s) > 1 else 0.0,
            "pooled_oof_f2": pooled_f2_fixed,
            "baseline_f2": baseline_f2,
            "n_oof_rows": int(len(oof)),
            "folds": fold_rows,
        }
        if args.tune_threshold:
            tuned_f2s = [r["f2_tuned"] for r in fold_rows]
            entry["fold_f2_tuned_mean"] = float(np.mean(tuned_f2s))
            entry["fold_f2_tuned_std"] = float(np.std(tuned_f2s, ddof=1)) if len(tuned_f2s) > 1 else 0.0
            oof_tuned: list[pd.DataFrame] = []
            for fold_id in fold_ids:
                train_df, val_df = split_frames_by_fold(cv_df, fold_id, source_folds)
                train_sub = subsample_frames_within_clips(train_df, stride)
                val_sub = subsample_frames_within_clips(val_df, stride)
                tuned_args = argparse.Namespace(**vars(args))
                tuned_args.tune_threshold = True
                tuned_args.decision_threshold = None
                _, val_preds_t, _ = train_and_evaluate(
                    train_sub,
                    val_sub,
                    feature_columns=feature_columns,
                    args=tuned_args,
                    manifest=manifest,
                    eval_split_name=f"fold_{fold_id}",
                    split_method="5fold_cv_fps",
                )
                oof_tuned.append(val_preds_t)
            oof_t = pd.concat(oof_tuned, ignore_index=True)
            entry["pooled_oof_f2_tuned"] = float(
                fbeta_score(
                    oof_t["is_playing"].astype(int).to_numpy(),
                    oof_t["pred_playing"].astype(int).to_numpy(),
                    beta=args.f_beta,
                    zero_division=0,
                )
            )

        fps_results.append(entry)
        msg = (
            f"fps={target_fps:5.1f} stride={stride:2d}  "
            f"F2@fixed={pooled_f2_fixed:.4f}  fold_mean={np.mean(fold_f2s):.4f}  baseline={baseline_f2:.4f}"
        )
        if args.tune_threshold:
            msg += f"  F2@tuned={entry['pooled_oof_f2_tuned']:.4f}"
        print(msg)

    return {
        "feature_run_id": manifest.get("run_id") or run_dir.name,
        "eval": "novel_video_source_5fold",
        "folds_csv": str(args.folds_csv.relative_to(REPO_ROOT)),
        "base_fps": args.base_fps,
        "target_fps": sorted(set(args.target_fps), reverse=True),
        "f_beta": args.f_beta,
        "decision_threshold": fixed_thr,
        "tune_threshold_also_reported": args.tune_threshold,
        "by_fps": fps_results,
    }


def plot_fps_ablation(report: dict[str, Any], *, out_path: Path, dpi: int, metric: str = "fixed") -> None:
    rows = sorted(report["by_fps"], key=lambda r: r["target_fps"])
    fps = np.array([r["target_fps"] for r in rows], dtype=float)
    if metric == "tuned":
        f2 = np.array([r.get("pooled_oof_f2_tuned", r["pooled_oof_f2"]) for r in rows], dtype=float)
        f2_std = np.array([r.get("fold_f2_tuned_std", r["fold_f2_std"]) for r in rows], dtype=float)
    else:
        f2 = np.array([r["pooled_oof_f2"] for r in rows], dtype=float)
        f2_std = np.array([r["fold_f2_std"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.fill_between(fps, f2 - f2_std, f2 + f2_std, color="#E8881A", alpha=0.15, zorder=2)
    ax.plot(fps, f2, color="#E8881A", marker="o", markersize=10, linewidth=2.5, zorder=3)
    ax.set_xlabel("frames per second sampled", fontsize=11)
    ax.set_ylabel("F2", fontsize=11)
    ax.set_xticks(fps)
    ax.set_xticklabels([str(int(x)) if x == int(x) else str(x) for x in fps])
    ymin = max(0.5, float((f2 - f2_std).min()) - 0.05)
    ymax = min(0.95, float((f2 + f2_std).max()) + 0.02)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#CCCCCC", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    thr = report.get("decision_threshold", 0.24)
    ax.set_title(f"Novel-video 5-fold F2 (threshold={thr:.2f})", fontsize=12, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        format=out_path.suffix.lstrip(".") or "png",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_novel_fps_cv(args)
    json_path = out_dir / "fps_ablation_novel.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table_rows = []
    for r in report["by_fps"]:
        for f in r["folds"]:
            row = {
                "target_fps": r["target_fps"],
                "frame_stride": r["frame_stride"],
                "fold_id": f["fold_id"],
                "f2_fixed": f["f2"],
                "recall": f["recall"],
                "precision": f["precision"],
                "n_val_rows": f["n_val_rows"],
            }
            if "f2_tuned" in f:
                row["f2_tuned"] = f["f2_tuned"]
                row["tuned_threshold"] = f.get("tuned_threshold")
            table_rows.append(row)
        pooled = {
            "target_fps": r["target_fps"],
            "frame_stride": r["frame_stride"],
            "fold_id": "pooled",
            "f2_fixed": r["pooled_oof_f2"],
            "recall": None,
            "precision": None,
            "n_val_rows": r["n_oof_rows"],
        }
        if "pooled_oof_f2_tuned" in r:
            pooled["f2_tuned"] = r["pooled_oof_f2_tuned"]
        table_rows.append(pooled)
    pd.DataFrame(table_rows).to_csv(out_dir / "fps_ablation_novel.csv", index=False)

    plot_path = args.save_plot or (out_dir / "fps_ablation_novel.jpg")
    plot_fps_ablation(report, out_path=plot_path.resolve(), dpi=args.dpi, metric=args.plot_metric)
    print(f"\nwrote {json_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
