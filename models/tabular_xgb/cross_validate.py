#!/usr/bin/env python3
"""5-fold source-grouped cross validation for tabular XGBoost.

    python models/tabular_xgb/cross_validate.py \\
        --feature-run-id all_clips_features_v2 \\
        --run-id all_clips_features_v2-xgb-5fold

Fold assignments: ``data/5_fold_train_test_split.csv`` (columns ``video_id``, ``fold_id``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TABULAR_DIR = Path(__file__).resolve().parent
if str(_TABULAR_DIR) not in sys.path:
    sys.path.insert(0, str(_TABULAR_DIR))

from feature_extraction.core.feature_columns import active_feature_columns  # noqa: E402
from fold_split import (  # noqa: E402
    DEFAULT_FOLDS_CSV,
    filter_to_fold_sources,
    load_source_folds,
    split_frames_by_fold,
)
from train import (  # noqa: E402
    _fbeta_metric_key,
    load_all_frames,
    resolve_run_dir,
    test_predictions_to_csv,
    train_and_evaluate,
)

logger = logging.getLogger(__name__)

CV_ROOT = REPO_ROOT / "models" / "tabular_xgb" / "cv"

SUMMARY_SCALAR_KEYS = ("precision", "recall", "f1")
SUMMARY_COUNT_KEYS = ("tp", "fp", "tn", "fn")


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _metrics_for_summary(report: dict[str, Any], *, fbeta_key: str) -> dict[str, float]:
    m = report["metrics"]
    cm = report["confusion_matrix"]
    out = {
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),
        fbeta_key: float(m[fbeta_key]),
        "tp": float(cm["tp"]),
        "fp": float(cm["fp"]),
        "tn": float(cm["tn"]),
        "fn": float(cm["fn"]),
    }
    return out


def aggregate_fold_results(folds: list[dict[str, Any]], *, fbeta_key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_folds": len(folds)}
    scalar_keys = SUMMARY_SCALAR_KEYS + (fbeta_key,)
    for key in scalar_keys + SUMMARY_COUNT_KEYS:
        mean, std = mean_std([float(f["metrics"][key]) for f in folds])
        summary[key] = {"mean": mean, "std": std}

    thresholds = [float(f["decision_threshold"]) for f in folds]
    thr_mean, thr_std = mean_std(thresholds)
    summary["decision_threshold"] = {"mean": thr_mean, "std": thr_std, "per_fold": thresholds}
    return summary


def format_summary_line(key: str, mean: float, std: float, *, digits: int = 3) -> str:
    if key in SUMMARY_COUNT_KEYS:
        return f"{key:>10s}: {mean:8.0f} ± {std:6.0f}"
    return f"{key:>10s}: {mean:{digits}.{digits}f} ± {std:{digits}.{digits}f}"


def print_cv_summary(summary: dict[str, Any], *, fbeta_key: str) -> None:
    print(f"\n=== {summary['n_folds']}-fold CV summary ===")
    for key in SUMMARY_SCALAR_KEYS + (fbeta_key,):
        entry = summary[key]
        print(format_summary_line(key, entry["mean"], entry["std"]))

    thr = summary["decision_threshold"]
    print(
        f"{'threshold':>10s}: {thr['mean']:.4f} ± {thr['std']:.4f}  "
        f"(per fold: {[round(t, 4) for t in thr['per_fold']]})"
    )

    for key in SUMMARY_COUNT_KEYS:
        entry = summary[key]
        print(format_summary_line(key, entry["mean"], entry["std"]))


def _warn_excluded_sources(all_df: pd.DataFrame, source_folds: dict[str, int]) -> None:
    in_run = set(all_df["source_id"].astype(str).unique())
    in_fold = set(source_folds)
    extra = sorted(in_run - in_fold)
    if extra:
        logger.warning(
            "excluding %d source(s) not in fold CSV from CV: %s",
            len(extra),
            ", ".join(extra),
        )


def run_cross_validation(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_run_dir(args)
    feature_columns = active_feature_columns(args.feature_subset)
    all_df, manifest = load_all_frames(run_dir, feature_columns)
    source_folds = load_source_folds(args.folds_csv)
    _warn_excluded_sources(all_df, source_folds)

    all_folds = sorted(set(source_folds.values()))
    fold_ids = all_folds if not args.folds else sorted(set(args.folds))
    unknown = set(fold_ids) - set(all_folds)
    if unknown:
        raise RuntimeError(f"unknown fold ids: {sorted(unknown)}; available: {all_folds}")

    cv_df = filter_to_fold_sources(all_df, source_folds)
    feature_run_id = str(manifest.get("run_id") or args.feature_run_id or run_dir.name)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-xgb-cv")
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fbeta_key = _fbeta_metric_key(args.f_beta)
    hparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_seed": args.random_seed,
        "f_beta": args.f_beta,
        "tune_threshold": args.tune_threshold,
        "cv_folds": args.cv_folds,
        "feature_subset": args.feature_subset,
    }

    print(f"CV run: {run_id}")
    print(f"feature_run_id: {feature_run_id}")
    print(f"folds: {fold_ids}  folds_csv={args.folds_csv.relative_to(REPO_ROOT)}")
    print(f"output: {out_dir.relative_to(REPO_ROOT)}")
    print(f"hparams: {hparams}")

    fold_results: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        train_df, val_df = split_frames_by_fold(cv_df, fold_id, source_folds)
        val_sources = sorted(
            sid for sid, fid in source_folds.items() if fid == fold_id
        )
        fold_dir = out_dir / f"fold-{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        eval_split_name = f"fold_{fold_id}"

        print(
            f"\n=== fold {fold_id} === "
            f"val sources={val_sources}  "
            f"train clips={train_df['clip_key'].nunique()}  "
            f"val clips={val_df['clip_key'].nunique()}"
        )

        t0 = time.perf_counter()
        report, val_preds, model = train_and_evaluate(
            train_df,
            val_df,
            feature_columns=feature_columns,
            args=args,
            manifest=manifest,
            eval_split_name=eval_split_name,
            split_method="5fold_cv",
        )
        elapsed = round(time.perf_counter() - t0, 2)

        model.save_model(str(fold_dir / "xgb_model.json"))
        report_path = fold_dir / "xgb_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        val_preds.to_parquet(fold_dir / "xgb_val_preds.parquet", index=False)
        val_csv = test_predictions_to_csv(val_preds, split_name=eval_split_name)
        val_csv.to_csv(fold_dir / "xgb_val_preds.csv", index=False)
        oof_frames.append(val_csv)

        fold_metrics = _metrics_for_summary(report, fbeta_key=fbeta_key)
        fold_record = {
            "fold_id": fold_id,
            "val_source_ids": val_sources,
            "n_train_clips": report["n_train_clips"],
            "n_val_clips": report["n_val_clips"],
            "n_train_rows": report["n_train_rows"],
            "n_val_rows": report["n_val_rows"],
            "decision_threshold": report["decision_threshold"],
            "metrics": fold_metrics,
            "checkpoint_dir": str(fold_dir.relative_to(REPO_ROOT)),
            "elapsed_sec": elapsed,
        }
        fold_results.append(fold_record)

        m = fold_metrics
        print(
            f"fold {fold_id} done: {fbeta_key}={m[fbeta_key]:.4f} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"TP={int(m['tp'])} FP={int(m['fp'])} TN={int(m['tn'])} FN={int(m['fn'])} "
            f"({elapsed}s)"
        )

    summary = aggregate_fold_results(fold_results, fbeta_key=fbeta_key)
    oof_preds = pd.concat(oof_frames, ignore_index=True)
    oof_path = out_dir / "xgb_oof_preds.csv"
    oof_preds.to_csv(oof_path, index=False)

    report_doc = {
        "run_id": run_id,
        "feature_run_id": feature_run_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "folds_csv": str(args.folds_csv.resolve().relative_to(REPO_ROOT)),
        "fold_ids": fold_ids,
        "hparams": hparams,
        "folds": fold_results,
        "summary": summary,
        "oof_preds_csv": str(oof_path.relative_to(REPO_ROOT)),
        "n_oof_rows": int(len(oof_preds)),
    }
    summary_path = out_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(report_doc, indent=2) + "\n", encoding="utf-8")
    print_cv_summary(summary, fbeta_key=fbeta_key)
    print(f"\nwrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"wrote {oof_path.relative_to(REPO_ROOT)} ({len(oof_preds)} rows)")
    return report_doc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-fold source-grouped CV for tabular XGBoost")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "feature_extraction" / "_runs",
    )
    p.add_argument("--feature-run-id", type=str, default=None)
    p.add_argument("--folds-csv", type=Path, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--output-dir", type=Path, default=CV_ROOT)
    p.add_argument(
        "--run-id",
        default=None,
        help="Subdirectory under models/tabular_xgb/cv/ (default: UTC timestamp)",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Run only these fold ids (default: all folds in CSV)",
    )
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--feature-subset", choices=("all", "base"), default="all")
    p.add_argument("--f-beta", type=float, default=2.0)
    p.add_argument("--cv-folds", type=int, default=5, help="Threshold-tuning folds on train clips.")
    p.add_argument("--tune-threshold", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decision-threshold", type=float, default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    args = parse_args()
    run_cross_validation(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
