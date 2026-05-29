#!/usr/bin/env python3
"""5-fold source-grouped cross validation for the LSTM playing classifier.

    python models/lstm/cross_validate.py \\
        --device mps \\
        --epochs 10 \\
        --checkpoint-metric f_beta \\
        --early-stop-patience 2 \\
        --run-id cnn-lstm-5fold

Fold assignments: ``data/5_fold_train_test_split.csv`` (columns ``video_id``, ``fold_id``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import (  # noqa: E402
    DEFAULT_FOLDS_CSV,
    DEFAULT_F_BETA,
    DEFAULT_FRAME_LABELS_CSV,
    fold_train_val_clip_ids,
    list_clip_ids,
    load_source_folds,
)
from models.lstm.train import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_CHECKPOINT_METRIC,
    DEFAULT_PRED_THRESHOLD,
    EPOCHS,
    HEAD_DROPOUT,
    LR,
    WEIGHT_DECAY,
    train,
)

CV_ROOT = REPO_ROOT / "models" / "lstm" / "checkpoints" / "cv"

SUMMARY_SCALAR_KEYS = ("f_beta", "precision", "recall")
SUMMARY_COUNT_KEYS = ("tp", "fp", "tn", "fn")


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_fold_results(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean ± std across fold-level metrics."""
    summary: dict[str, Any] = {"n_folds": len(folds)}
    for key in SUMMARY_SCALAR_KEYS + SUMMARY_COUNT_KEYS:
        mean, std = mean_std([float(f["metrics"][key]) for f in folds])
        summary[key] = {"mean": mean, "std": std}

    best_epochs = [int(f["best_epoch"]) for f in folds]
    epoch_mean, epoch_std = mean_std([float(e) for e in best_epochs])
    summary["best_epoch"] = {"mean": epoch_mean, "std": epoch_std, "per_fold": best_epochs}

    return summary


def format_summary_line(key: str, mean: float, std: float, *, digits: int = 3) -> str:
    if key in SUMMARY_COUNT_KEYS:
        return f"{key:>10s}: {mean:8.0f} ± {std:6.0f}"
    return f"{key:>10s}: {mean:{digits}.{digits}f} ± {std:{digits}.{digits}f}"


def print_cv_summary(summary: dict[str, Any], *, f_beta: float) -> None:
    print(f"\n=== {summary['n_folds']}-fold CV summary (β={f_beta:g}) ===")
    for key in SUMMARY_SCALAR_KEYS:
        entry = summary[key]
        print(format_summary_line(key, entry["mean"], entry["std"]))

    epoch = summary["best_epoch"]
    print(
        f"{'best_epoch':>10s}: {epoch['mean']:.1f} ± {epoch['std']:.1f}  "
        f"(per fold: {epoch['per_fold']})"
    )

    for key in SUMMARY_COUNT_KEYS:
        entry = summary[key]
        print(format_summary_line(key, entry["mean"], entry["std"]))


def run_cross_validation(args: argparse.Namespace) -> dict[str, Any]:
    labeled_clip_ids = list_clip_ids(DEFAULT_FRAME_LABELS_CSV)
    source_folds = load_source_folds(args.folds_csv)
    all_folds = sorted(set(source_folds.values()))
    fold_ids = all_folds if not args.folds else sorted(set(args.folds))

    unknown = set(fold_ids) - set(all_folds)
    if unknown:
        raise RuntimeError(f"unknown fold ids: {sorted(unknown)}; available: {all_folds}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-cv")
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pred_threshold": args.pred_threshold,
        "f_beta": args.f_beta,
        "head_dropout": args.head_dropout,
        "checkpoint_metric": args.checkpoint_metric,
        "boundary_margin": args.boundary_margin,
        "train_frame_stride": args.train_frame_stride,
        "early_stop_patience": args.early_stop_patience,
    }

    print(f"CV run: {run_id}")
    print(f"folds: {fold_ids}  folds_csv={args.folds_csv.relative_to(REPO_ROOT)}")
    print(f"output: {out_dir.relative_to(REPO_ROOT)}")
    print(f"hparams: {hparams}")

    fold_results: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        train_ids, val_ids = fold_train_val_clip_ids(
            fold_id,
            folds_csv=args.folds_csv,
            labeled_clip_ids=labeled_clip_ids,
        )
        val_sources = sorted(
            sid for sid, fid in source_folds.items() if fid == fold_id
        )
        fold_dir = out_dir / f"fold-{fold_id}"
        print(
            f"\n=== fold {fold_id} === "
            f"val sources={val_sources}  train clips={len(train_ids)}  val clips={len(val_ids)}"
        )

        t0 = time.perf_counter()
        result = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            pred_threshold=args.pred_threshold,
            f_beta=args.f_beta,
            head_dropout=args.head_dropout,
            checkpoint_metric=args.checkpoint_metric,
            checkpoint_dir=fold_dir,
            boundary_margin=args.boundary_margin,
            train_frame_stride=args.train_frame_stride,
            early_stop_patience=args.early_stop_patience,
            train_clip_ids=train_ids,
            val_clip_ids=val_ids,
            quiet=args.quiet,
        )
        elapsed = round(time.perf_counter() - t0, 2)

        fold_record = {
            "fold_id": fold_id,
            "val_source_ids": val_sources,
            "n_train_clips": len(train_ids),
            "n_val_clips": len(val_ids),
            "train_clip_ids": train_ids,
            "val_clip_ids": val_ids,
            "best_epoch": result["best_epoch"],
            "metrics": result["metrics"],
            "checkpoint_dir": result["checkpoint_dir"],
            "elapsed_sec": elapsed,
        }
        fold_results.append(fold_record)

        m = result["metrics"]
        print(
            f"fold {fold_id} done: best_epoch={result['best_epoch']} "
            f"f_beta={m['f_beta']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"TP={int(m['tp'])} FP={int(m['fp'])} TN={int(m['tn'])} FN={int(m['fn'])} "
            f"({elapsed}s)"
        )

    summary = aggregate_fold_results(fold_results)
    report = {
        "run_id": run_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "folds_csv": str(args.folds_csv.resolve().relative_to(REPO_ROOT)),
        "fold_ids": fold_ids,
        "hparams": hparams,
        "folds": fold_results,
        "summary": summary,
    }

    summary_path = out_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_cv_summary(summary, f_beta=args.f_beta)
    print(f"\nwrote {summary_path.relative_to(REPO_ROOT)}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-fold source-grouped CV for LSTM train.py")
    p.add_argument("--folds-csv", type=Path, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--output-dir", type=Path, default=CV_ROOT)
    p.add_argument(
        "--run-id",
        default=None,
        help="Subdirectory under checkpoints/cv/ (default: UTC timestamp)",
    )
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Run only these fold ids (default: all folds in CSV)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--pred-threshold", type=float, default=DEFAULT_PRED_THRESHOLD)
    p.add_argument("--f-beta", type=float, default=DEFAULT_F_BETA)
    p.add_argument("--head-dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument(
        "--checkpoint-metric",
        choices=("loss", "recall", "cost", "f_beta"),
        default=DEFAULT_CHECKPOINT_METRIC,
    )
    p.add_argument("--boundary-margin", type=int, default=0)
    p.add_argument("--train-frame-stride", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_cross_validation(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
