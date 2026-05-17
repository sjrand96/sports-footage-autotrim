#!/usr/bin/env python3
"""Train pooled XGBoost from a feature-extraction run (train/ + test/ parquets)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.core.feature_columns import (  # noqa: E402
    active_feature_columns,
    float_fillna_cols_for_features,
)

_DEFAULT_RUNS_ROOT = REPO_ROOT / "feature_extraction" / "_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost on feature_extraction/{run_id}/train, evaluate on test/."
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to one run (contains manifest.json, train/, test/).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=_DEFAULT_RUNS_ROOT,
        help=f"Root for --feature-run-id (default: {_DEFAULT_RUNS_ROOT})",
    )
    p.add_argument(
        "--feature-run-id",
        type=str,
        default=None,
        help="Run id under --runs-root (alternative to --run-dir).",
    )
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--random-seed", type=int, default=42, help="XGBoost random_state only.")
    p.add_argument(
        "--feature-subset",
        choices=("all", "base"),
        default="all",
        help="'base' = 7 legacy columns; 'all' = base + Chunk 1 spatial.",
    )
    p.add_argument(
        "--save-test-preds",
        type=Path,
        default=None,
        help="Optional parquet for held-out test predictions.",
    )
    p.add_argument("--save-report-json", type=Path, default=None)
    p.add_argument("--save-model", type=Path, default=None, help="XGBoost model JSON/UBJ path.")
    return p.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None and args.feature_run_id is not None:
        raise SystemExit("use either --run-dir or --feature-run-id, not both")
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
    elif args.feature_run_id is not None:
        run_dir = args.runs_root.expanduser().resolve() / args.feature_run_id
    else:
        raise SystemExit("pass --run-dir or --feature-run-id")
    if not (run_dir / "manifest.json").is_file():
        raise SystemExit(f"manifest not found: {run_dir / 'manifest.json'}")
    return run_dir


def _clip_key_frame(source_id: str, clip_index: int) -> str:
    return f"{source_id}_{int(clip_index):03d}"


def _load_split_parquets(split_dir: Path, *, split_name: str, feature_columns: list[str]) -> pd.DataFrame:
    if not split_dir.is_dir():
        raise RuntimeError(f"missing split directory: {split_dir}")

    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquets in {split_dir}")

    chunks: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_parquet(path)
        missing = [c for c in feature_columns + ["is_playing", "source_id", "clip_index", "frame_idx"] if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {missing}")
        df = df.copy()
        df["split"] = split_name
        df["clip_key"] = _clip_key_frame(str(df["source_id"].iloc[0]), int(df["clip_index"].iloc[0]))
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def load_train_test_frames(run_dir: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    train_df = _load_split_parquets(run_dir / "train", split_name="train", feature_columns=feature_columns)
    test_df = _load_split_parquets(run_dir / "test", split_name="test", feature_columns=feature_columns)
    return train_df, test_df, manifest


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, XGBClassifier]:
    fill_cols = float_fillna_cols_for_features(feature_columns)

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    for col in fill_cols:
        X_train[col] = X_train[col].fillna(-1.0)
        X_test[col] = X_test[col].fillna(-1.0)

    y_train = train_df["is_playing"].astype(int).to_numpy()
    y_test = test_df["is_playing"].astype(int).to_numpy()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_seed,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train.to_numpy(dtype=np.float32), y_train)
    y_pred = model.predict(X_test.to_numpy(dtype=np.float32)).astype(int)
    y_prob = model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    train_clips = sorted(train_df["clip_key"].unique().tolist())
    test_clips = sorted(test_df["clip_key"].unique().tolist())

    report: dict[str, Any] = {
        "feature_extraction_run_id": manifest.get("run_id"),
        "extractor_version": manifest.get("extractor_version"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "split_method": manifest.get("split_method"),
        "feature_subset": args.feature_subset,
        "n_features": len(feature_columns),
        "feature_columns": list(feature_columns),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_clips": len(train_clips),
        "n_test_clips": len(test_clips),
        "train_clip_keys": train_clips,
        "test_clip_keys": test_clips,
        "class_balance_train": {"negative": neg_count, "positive": pos_count},
        "scale_pos_weight": scale_pos_weight,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }

    test_out = test_df[["source_id", "clip_index", "clip_key", "frame_idx", "timestamp_sec", "is_playing"]].copy()
    test_out["pred_playing"] = y_pred.astype(bool)
    test_out["pred_prob_playing"] = y_prob.astype(float)
    return report, test_out, model


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    feature_columns = active_feature_columns(args.feature_subset)

    train_df, test_df, manifest = load_train_test_frames(run_dir, feature_columns)
    if train_df.empty or test_df.empty:
        raise SystemExit("train or test split is empty; need parquets in both train/ and test/")

    report, test_preds, model = train_and_evaluate(
        train_df, test_df, feature_columns=feature_columns, args=args, manifest=manifest
    )

    print("=== Tabular XGBoost (feature-extraction run) ===")
    print(f"run_dir: {run_dir}")
    print(f"feature_run_id: {report.get('feature_extraction_run_id')}")
    print(f"feature_subset: {args.feature_subset} ({len(feature_columns)} columns)")
    print(f"train rows / clips: {report['n_train_rows']} / {report['n_train_clips']}")
    print(f"test rows / clips:  {report['n_test_rows']} / {report['n_test_clips']}")
    print("")
    print("metrics:")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print("")
    cm = report["confusion_matrix"]
    print("confusion matrix (rows=true, cols=pred) labels=[0,1]:")
    print(f"  [{cm['matrix'][0][0]}, {cm['matrix'][0][1]}]")
    print(f"  [{cm['matrix'][1][0]}, {cm['matrix'][1][1]}]")

    if args.save_test_preds is not None:
        args.save_test_preds.parent.mkdir(parents=True, exist_ok=True)
        test_preds.to_parquet(args.save_test_preds, index=False)
        print(f"\nsaved test predictions: {args.save_test_preds}")

    if args.save_report_json is not None:
        args.save_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved report: {args.save_report_json}")

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.save_model))
        print(f"saved model: {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
