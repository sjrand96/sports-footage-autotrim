#!/usr/bin/env python3
"""Simulate lower extract/inference FPS via within-clip frame subsampling + retrain XGB."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TABULAR_DIR = Path(__file__).resolve().parent
if str(_TABULAR_DIR) not in sys.path:
    sys.path.insert(0, str(_TABULAR_DIR))

from cv_threshold import apply_threshold, classification_metrics  # noqa: E402
from train import (  # noqa: E402
    _DEFAULT_RUNS_ROOT,
    _fbeta_metric_key,
    build_xgb_classifier,
    load_train_test_frames,
    resolve_run_dir,
)

BASE_FPS = 30
DEFAULT_TARGET_FPS = (30, 15, 10, 5, 2, 1, 0.5)
DEFAULT_DECISION_THRESHOLD = 0.24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FPS sensitivity: subsample frames per clip, retrain XGB, fixed threshold."
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=_DEFAULT_RUNS_ROOT)
    p.add_argument("--feature-run-id", type=str, default=None)
    p.add_argument(
        "--target-fps",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_FPS),
        help=f"Simulated FPS levels (default: {list(DEFAULT_TARGET_FPS)}).",
    )
    p.add_argument(
        "--no-always-playing-baseline",
        action="store_true",
        help="Skip always-predict-playing reference rows.",
    )
    p.add_argument(
        "--base-fps",
        type=int,
        default=BASE_FPS,
        help="Assumed native FPS in parquets (default 30).",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=f"P(playing) cutoff (default: from xgb_report.json or {DEFAULT_DECISION_THRESHOLD}).",
    )
    p.add_argument("--feature-subset", choices=("all", "base"), default="all")
    p.add_argument("--f-beta", type=float, default=2.0)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Results table path (default: {run_dir}/fps_sensitivity.csv).",
    )
    p.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Plot path (default: {run_dir}/fps_sensitivity.png).",
    )
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def stride_for_target_fps(target_fps: float, *, base_fps: int) -> int:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps >= base_fps:
        return 1
    return max(1, round(base_fps / target_fps))


def subsample_frames_within_clips(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every ``stride``-th frame per clip (sorted by frame_idx)."""
    if stride <= 1:
        return df

    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("clip_key", sort=False):
        ordered = group.sort_values("frame_idx", kind="stable")
        parts.append(ordered.iloc[::stride])
    return pd.concat(parts, ignore_index=True)


def load_decision_threshold(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    report_path = run_dir / "xgb_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("decision_threshold") is not None:
            return float(report["decision_threshold"])
    return DEFAULT_DECISION_THRESHOLD


def train_and_eval_subsampled(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    args: argparse.Namespace,
    decision_threshold: float,
    target_fps: float,
    stride: int,
) -> dict[str, Any]:
    from feature_extraction.core.feature_columns import float_fillna_cols_for_features

    train_sub = subsample_frames_within_clips(train_df, stride)
    test_sub = subsample_frames_within_clips(test_df, stride)
    fill_cols = float_fillna_cols_for_features(feature_columns)

    X_train = train_sub[feature_columns].copy()
    X_test = test_sub[feature_columns].copy()
    for col in fill_cols:
        X_train[col] = X_train[col].fillna(-1.0)
        X_test[col] = X_test[col].fillna(-1.0)

    y_train = train_sub["is_playing"].astype(int).to_numpy()
    y_test = test_sub["is_playing"].astype(int).to_numpy()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    model = build_xgb_classifier(args, scale_pos_weight=scale_pos_weight)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    model.fit(X_train_np, y_train)

    y_prob = model.predict_proba(X_test_np)[:, 1]
    y_pred = apply_threshold(y_prob, decision_threshold)
    metrics = classification_metrics(y_test, y_pred, beta=args.f_beta)
    fbeta_key = _fbeta_metric_key(args.f_beta)

    return {
        "model": "xgb",
        "target_fps": target_fps,
        "base_fps": args.base_fps,
        "frame_stride": stride,
        "decision_threshold": decision_threshold,
        "n_train_rows": int(len(train_sub)),
        "n_test_rows": int(len(test_sub)),
        "n_train_clips": int(train_sub["clip_key"].nunique()),
        "n_test_clips": int(test_sub["clip_key"].nunique()),
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        f"test_{fbeta_key}": metrics[fbeta_key],
    }


def always_playing_baseline(
    test_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    target_fps: float,
    stride: int,
) -> dict[str, Any]:
    """Predict playing=1 on every subsampled test frame (recall=1; Fβ floor reference)."""
    test_sub = subsample_frames_within_clips(test_df, stride)
    y_test = test_sub["is_playing"].astype(int).to_numpy()
    y_pred = np.ones(len(y_test), dtype=np.int32)
    metrics = classification_metrics(y_test, y_pred, beta=args.f_beta)
    fbeta_key = _fbeta_metric_key(args.f_beta)
    pos_rate = float(np.mean(y_test)) if len(y_test) else float("nan")

    return {
        "model": "always_playing",
        "target_fps": target_fps,
        "base_fps": args.base_fps,
        "frame_stride": stride,
        "decision_threshold": None,
        "n_train_rows": None,
        "n_test_rows": int(len(test_sub)),
        "n_train_clips": None,
        "n_test_clips": int(test_sub["clip_key"].nunique()),
        "test_label_positive_rate": pos_rate,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        f"test_{fbeta_key}": metrics[fbeta_key],
    }


def results_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.sort_values(["target_fps", "model"], ascending=[False, True]).reset_index(drop=True)


def plot_sensitivity(df: pd.DataFrame, *, fbeta_key: str, out_path: Path, dpi: int) -> None:
    xgb = df[df["model"] == "xgb"].sort_values("target_fps", ascending=False)
    base = df[df["model"] == "always_playing"].sort_values("target_fps", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not xgb.empty:
        fps = xgb["target_fps"].to_numpy()
        ax.plot(fps, xgb[f"test_{fbeta_key}"], marker="o", linewidth=2, label=f"XGB {fbeta_key}")
        ax.plot(fps, xgb["test_f1"], marker="s", linewidth=1.2, linestyle="--", alpha=0.75, label="XGB F1")
        thr = float(xgb["decision_threshold"].iloc[0])
        title_thr = f", XGB threshold = {thr:.3f}"
    else:
        title_thr = ""

    if not base.empty:
        ax.plot(
            base["target_fps"],
            base[f"test_{fbeta_key}"],
            marker="D",
            linewidth=1.5,
            linestyle=":",
            color="#555555",
            label=f"always playing ({fbeta_key})",
        )

    ax.set_xlabel("Simulated FPS (within-clip subsample)")
    ax.set_ylabel("Test metric")
    xticks = sorted(df["target_fps"].unique(), reverse=True)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x).replace(".0", "") for x in xticks])
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Test metrics vs simulated FPS{title_thr}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_table(df: pd.DataFrame, *, fbeta_key: str) -> None:
    cols = [
        "model",
        "target_fps",
        "frame_stride",
        "n_test_rows",
        f"test_{fbeta_key}",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_accuracy",
        "test_label_positive_rate",
    ]
    view = df[cols].copy()
    numeric = [c for c in cols if c.startswith("test_")]
    for col in numeric:
        view[col] = view[col].map(lambda v: f"{v:.4f}")
    print(view.to_string(index=False))


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args)

    from feature_extraction.core.feature_columns import active_feature_columns

    feature_columns = active_feature_columns(args.feature_subset)
    train_df, test_df, manifest = load_train_test_frames(run_dir, feature_columns)
    decision_threshold = load_decision_threshold(run_dir, args.decision_threshold)
    fbeta_key = _fbeta_metric_key(args.f_beta)

    print("=== FPS sensitivity (fixed threshold, retrain per level) ===")
    print(f"feature_run_id: {manifest.get('run_id') or run_dir.name}")
    print(f"decision_threshold: {decision_threshold:.4f} (held constant)")
    print(f"base_fps: {args.base_fps}")
    print("")

    rows: list[dict[str, Any]] = []
    for target_fps in sorted(set(args.target_fps), reverse=True):
        stride = stride_for_target_fps(target_fps, base_fps=args.base_fps)
        if not args.no_always_playing_baseline:
            base_row = always_playing_baseline(
                test_df,
                args=args,
                target_fps=target_fps,
                stride=stride,
            )
            rows.append(base_row)
            print(
                f"fps={target_fps:4} stride={stride:2d} [always_playing] "
                f"test_rows={base_row['n_test_rows']:5d} "
                f"{fbeta_key}={base_row[f'test_{fbeta_key}']:.4f} "
                f"recall={base_row['test_recall']:.4f} prec={base_row['test_precision']:.4f} "
                f"(label+rate={base_row['test_label_positive_rate']:.3f})"
            )

        row = train_and_eval_subsampled(
            train_df,
            test_df,
            feature_columns=feature_columns,
            args=args,
            decision_threshold=decision_threshold,
            target_fps=target_fps,
            stride=stride,
        )
        rows.append(row)
        print(
            f"fps={target_fps:4} stride={stride:2d} [xgb]          "
            f"train_rows={row['n_train_rows']:6d} test_rows={row['n_test_rows']:5d} "
            f"{fbeta_key}={row[f'test_{fbeta_key}']:.4f} "
            f"recall={row['test_recall']:.4f} prec={row['test_precision']:.4f}"
        )

    results = results_to_dataframe(rows)
    print("")
    print_table(results, fbeta_key=fbeta_key)

    save_csv = args.save_csv or (run_dir / "fps_sensitivity.csv")
    save_plot = args.save_plot or (run_dir / "fps_sensitivity.png")
    payload = {
        "feature_extraction_run_id": manifest.get("run_id"),
        "decision_threshold": decision_threshold,
        "base_fps": args.base_fps,
        "target_fps": list(sorted(set(args.target_fps), reverse=True)),
        "f_beta": args.f_beta,
        "rows": rows,
    }
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_csv, index=False)
    payload_path = save_csv.with_suffix(".json")
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_sensitivity(results, fbeta_key=fbeta_key, out_path=save_plot, dpi=args.dpi)

    print(f"\nsaved table: {save_csv}")
    print(f"saved plot:  {save_plot}")
    print(f"saved meta:  {payload_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
