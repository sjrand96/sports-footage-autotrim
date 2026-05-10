#!/usr/bin/env python3
"""Train a pooled XGBoost model from cached per-clip parquet files.

Reads paired files from a cache directory:
- *_features.parquet
- *_predictions.parquet  (used for ground-truth `is_playing`)

Performs a clip-level split (no frame leakage), trains one pooled model, and
reports accuracy / precision / recall / F1 and confusion matrix on test clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "n_players_total",
    "n_front_row",
    "n_back_row",
    "n_camera_side",
    "n_opposite_side",
    "median_nearest_neighbor_dist",
    "hands_above_head_count",
]
JOIN_COLUMNS = ["source_id", "clip_index", "frame_idx"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pooled XGBoost from cached clip parquets.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cv-pipeline/simplified_e2e_flow/cache"),
        help="Directory containing *_features.parquet and *_predictions.parquet files.",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of clips held out for test.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--min-clips", type=int, default=2, help="Minimum paired clips required.")
    p.add_argument(
        "--save-test-preds",
        type=Path,
        default=None,
        help="Optional parquet path for held-out test predictions.",
    )
    p.add_argument(
        "--save-report-json",
        type=Path,
        default=None,
        help="Optional JSON path for metrics + confusion matrix.",
    )
    p.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional output path for XGBoost model (JSON/UBJ).",
    )
    return p.parse_args()


def _clip_stem_from_filename(name: str, suffix: str) -> str | None:
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def load_paired_clip_rows(cache_dir: Path) -> pd.DataFrame:
    features_files = sorted(cache_dir.glob("*_features.parquet"))
    predictions_files = sorted(cache_dir.glob("*_predictions.parquet"))

    feature_by_stem: dict[str, Path] = {}
    pred_by_stem: dict[str, Path] = {}
    for f in features_files:
        stem = _clip_stem_from_filename(f.name, "_features.parquet")
        if stem:
            feature_by_stem[stem] = f
    for f in predictions_files:
        stem = _clip_stem_from_filename(f.name, "_predictions.parquet")
        if stem:
            pred_by_stem[stem] = f

    shared_stems = sorted(set(feature_by_stem) & set(pred_by_stem))
    if not shared_stems:
        raise RuntimeError(
            f"No paired feature/prediction parquets in {cache_dir}. Need matching *_features.parquet and *_predictions.parquet."
        )

    chunks: list[pd.DataFrame] = []
    for stem in shared_stems:
        df_feat = pd.read_parquet(feature_by_stem[stem])
        df_pred = pd.read_parquet(pred_by_stem[stem])

        missing_feat = [c for c in JOIN_COLUMNS + FEATURE_COLUMNS if c not in df_feat.columns]
        if missing_feat:
            raise RuntimeError(f"{feature_by_stem[stem].name} missing columns: {missing_feat}")
        if "is_playing" not in df_pred.columns:
            raise RuntimeError(f"{pred_by_stem[stem].name} missing required label column `is_playing`")

        df_join = df_feat.merge(
            df_pred[JOIN_COLUMNS + ["is_playing"]],
            on=JOIN_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if df_join.empty:
            raise RuntimeError(f"{stem}: feature/prediction join returned 0 rows")

        df_join["clip_key"] = df_join["source_id"].astype(str) + "_" + df_join["clip_index"].astype(int).map(
            lambda x: f"{x:03d}"
        )
        chunks.append(df_join)

    return pd.concat(chunks, ignore_index=True)


def train_and_evaluate(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, XGBClassifier]:
    clip_keys = np.array(sorted(df["clip_key"].unique().tolist()))
    if clip_keys.size < args.min_clips:
        raise RuntimeError(f"Need at least {args.min_clips} paired clips; found {clip_keys.size}.")

    train_keys, test_keys = train_test_split(
        clip_keys,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    train_df = df[df["clip_key"].isin(train_keys)].copy()
    test_df = df[df["clip_key"].isin(test_keys)].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train/test split produced an empty partition; adjust --test-size or clip count.")

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    X_train["median_nearest_neighbor_dist"] = X_train["median_nearest_neighbor_dist"].fillna(-1.0)
    X_test["median_nearest_neighbor_dist"] = X_test["median_nearest_neighbor_dist"].fillna(-1.0)
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

    report = {
        "n_total_rows": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_clips": int(len(train_keys)),
        "n_test_clips": int(len(test_keys)),
        "train_clip_keys": sorted(train_keys.tolist()),
        "test_clip_keys": sorted(test_keys.tolist()),
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
    df = load_paired_clip_rows(args.cache_dir)
    report, test_preds, model = train_and_evaluate(df, args)

    print("=== Pooled XGBoost Evaluation ===")
    print(f"cache_dir: {args.cache_dir}")
    print(f"total rows: {report['n_total_rows']}")
    print(f"train rows / clips: {report['n_train_rows']} / {report['n_train_clips']}")
    print(f"test rows / clips:  {report['n_test_rows']} / {report['n_test_clips']}")
    print("")
    print("metrics:")
    print(f"  accuracy:  {report['metrics']['accuracy']:.4f}")
    print(f"  precision: {report['metrics']['precision']:.4f}")
    print(f"  recall:    {report['metrics']['recall']:.4f}")
    print(f"  f1:        {report['metrics']['f1']:.4f}")
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
        args.save_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved report JSON: {args.save_report_json}")

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.save_model))
        print(f"saved model: {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

