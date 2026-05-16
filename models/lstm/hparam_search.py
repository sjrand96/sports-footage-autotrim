#!/usr/bin/env python3
"""Random search over LSTM hyperparameters (see SEARCH_SPACE below).

    python models/lstm/hparam_search.py --device mps --n-trials 30 --quiet

Tversky class weights (beta from inverse frequency) are computed from training counts in train.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.train import (  # noqa: E402
    DEFAULT_PRED_THRESHOLD,
    EPOCHS,
    HEAD_DROPOUT,
    RANDOM_SEED,
    train,
)

SEARCH_DIR = REPO_ROOT / "models" / "lstm" / "checkpoints" / "hparam_search"

SEARCH_SPACE: dict[str, list[Any]] = {
    "lr": [5e-5, 1e-4, 3e-4],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "train_frame_stride": [1, 5, 8],
    "boundary_margin": [0, 15, 30],
    "pred_threshold": [0.25, 0.35, 0.45],
}


def sample_hparams(rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def objective_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "loss":
        return float(metrics["loss"])
    if objective == "cost":
        return float(metrics["cost"])
    return -float(metrics["recall"])


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def completed_trial_ids(results_path: Path) -> set[int]:
    if not results_path.is_file():
        return set()
    ids: set[int] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok":
                ids.add(int(row["trial_id"]))
    return ids


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = [
        "trial_id",
        "score",
        "lr",
        "weight_decay",
        "train_frame_stride",
        "boundary_margin",
        "pred_threshold",
        "pos_weight",
        "test_loss",
        "cost",
        "recall",
        "precision",
        "f1",
        "best_epoch",
        "elapsed_sec",
        "checkpoint_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            hp = row.get("hparams") or {}
            m = row.get("metrics") or {}
            w.writerow(
                {
                    "trial_id": row["trial_id"],
                    "score": row["score"],
                    "lr": hp.get("lr"),
                    "weight_decay": hp.get("weight_decay"),
                    "train_frame_stride": hp.get("train_frame_stride"),
                    "boundary_margin": hp.get("boundary_margin"),
                    "pred_threshold": hp.get("pred_threshold"),
                    "pos_weight": row.get("pos_weight"),
                    "test_loss": m.get("loss"),
                    "cost": m.get("cost"),
                    "recall": m.get("recall"),
                    "precision": m.get("precision"),
                    "f1": m.get("f1"),
                    "best_epoch": row.get("best_epoch"),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "checkpoint_dir": row.get("checkpoint_dir"),
                }
            )


def run_search(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = completed_trial_ids(results_path) if args.resume else set()

    print(f"random search: {args.n_trials} trials  objective={args.objective}")
    print(f"space: {SEARCH_SPACE}")

    for trial_id in range(1, args.n_trials + 1):
        if trial_id in done:
            print(f"skip trial_{trial_id:04d}")
            continue

        hp = sample_hparams(rng)
        tid = f"trial_{trial_id:04d}"
        trial_dir = out_dir / tid
        print(f"\n=== {tid} === {hp}")

        t0 = time.perf_counter()
        record: dict[str, Any] = {"trial_id": trial_id, "hparams": hp, "objective": args.objective}
        try:
            result = train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                split_mode=args.split_mode,
                test_size=args.test_size,
                lr=float(hp["lr"]),
                weight_decay=float(hp["weight_decay"]),
                pred_threshold=float(hp["pred_threshold"]),
                head_dropout=args.head_dropout,
                checkpoint_metric=args.checkpoint_metric,
                checkpoint_dir=trial_dir,
                boundary_margin=int(hp["boundary_margin"]),
                train_frame_stride=int(hp["train_frame_stride"]),
                early_stop_patience=args.early_stop_patience,
                quiet=args.quiet,
                skip_final_eval=True,
            )
            metrics = result["metrics"]
            score = objective_score(metrics, args.objective)
            record.update(
                {
                    "status": "ok",
                    "score": score,
                    "metrics": metrics,
                    "pos_weight": result["hparams"].get("pos_weight"),
                    "best_epoch": result["best_epoch"],
                    "checkpoint_dir": result["checkpoint_dir"],
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )
            print(
                f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
                f"cost={metrics['cost']:.0f} loss={metrics['loss']:.4f} "
                f"pos_weight={result['hparams'].get('pos_weight', 0):.3f}"
            )
        except Exception as e:
            record.update({"status": "error", "error": str(e)})
            print(f"ERROR: {e}", file=sys.stderr)

        append_jsonl(results_path, record)

    ok: list[dict[str, Any]] = []
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    ok.append(row)
    ok.sort(key=lambda r: r["score"])
    write_summary_csv(ok, out_dir / "summary.csv")

    if not ok:
        return

    best = ok[0]
    print("\nbest:", best["hparams"], "recall=", best["metrics"]["recall"])
    link = out_dir / "best"
    src = Path(best["checkpoint_dir"])
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random hyperparameter search for train.py")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--objective", choices=("loss", "cost", "recall"), default="loss")
    p.add_argument("--checkpoint-metric", choices=("loss", "recall", "cost"), default="loss")
    p.add_argument("--output-dir", type=Path, default=SEARCH_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-dropout", type=float, default=HEAD_DROPOUT)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--split-mode", default="stratified_by_source")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument(
        "--pred-threshold",
        type=float,
        default=None,
        help=f"Fixed threshold for all trials (default: sample from SEARCH_SPACE; "
        f"train default alone is {DEFAULT_PRED_THRESHOLD})",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pred_threshold is not None:
        SEARCH_SPACE["pred_threshold"] = [args.pred_threshold]
    run_search(args)
