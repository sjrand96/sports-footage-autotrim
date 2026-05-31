#!/usr/bin/env python3
"""Train pooled XGBoost from a feature-extraction run (single parquet/ folder)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.core.feature_columns import (  # noqa: E402
    active_feature_columns,
    float_fillna_cols_for_features,
)
from feature_extraction.wandb_publish import (  # noqa: E402
    FEATURE_ARTIFACT_NAME,
    wandb_entity,
    wandb_group_for_feature_run,
    wandb_project,
)

_TABULAR_DIR = Path(__file__).resolve().parent
if str(_TABULAR_DIR) not in sys.path:
    sys.path.insert(0, str(_TABULAR_DIR))
from cv_threshold import (  # noqa: E402
    apply_threshold,
    classification_metrics,
    threshold_tune_to_dict,
    tune_threshold_clip_cv,
)

_DEFAULT_RUNS_ROOT = REPO_ROOT / "feature_extraction" / "_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train XGBoost on feature_extraction/{run_id}/parquet with split metadata."
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
    p.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit split JSON. Supports train/test clip IDs or clip keys via "
            "{train_clip_ids,test_clip_ids} or {train_clip_keys,test_clip_keys}."
        ),
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
        "--f-beta",
        type=float,
        default=2.0,
        help="F-beta for threshold tuning and primary test metric (default 2 = recall-heavy).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="StratifiedGroupKFold splits on train clips for OOF threshold tuning.",
    )
    p.add_argument(
        "--tune-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune P(playing) threshold on train clips via grouped CV (default: on).",
    )
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Fixed threshold (skips CV tuning). Default: 0.5 if --no-tune-threshold.",
    )
    p.add_argument(
        "--save-test-preds",
        type=Path,
        default=None,
        help="Optional parquet for held-out test predictions (wide).",
    )
    p.add_argument(
        "--save-test-csv",
        type=Path,
        default=None,
        help="Concise CSV of all test frames: clip, uri, prob, pred, label.",
    )
    p.add_argument("--save-report-json", type=Path, default=None)
    p.add_argument("--save-model", type=Path, default=None, help="XGBoost model JSON/UBJ path.")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log training run to Weights & Biases (requires WANDB_API_KEY).",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=f"W&B project (default: WANDB_PROJECT or {wandb_project()})",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help=f"W&B entity (default: WANDB_ENTITY or {wandb_entity()})",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run display name (default: xgb-{feature_run_id}).",
    )
    p.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional stable W&B run id (resume/overwrite same row). Default: new id per invocation.",
    )
    p.add_argument(
        "--wandb-log-threshold-sweep",
        action="store_true",
        help="Log full OOF threshold-vs-Fβ curve chart (default: scalars only).",
    )
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


def _load_all_parquets(run_dir: Path, *, feature_columns: list[str]) -> pd.DataFrame:
    parquet_dir = run_dir / "parquet"
    if not parquet_dir.is_dir():
        raise RuntimeError(f"missing parquet directory: {parquet_dir}")

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquets in {parquet_dir}")

    chunks: list[pd.DataFrame] = []
    required = feature_columns + ["is_playing", "source_id", "clip_index", "frame_idx", "clip_id"]
    for path in files:
        df = pd.read_parquet(path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path.name} missing columns: {missing}")
        df = df.copy()
        df["clip_key"] = _clip_key_frame(str(df["source_id"].iloc[0]), int(df["clip_index"].iloc[0]))
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def _split_sets_from_manifest(manifest: dict[str, Any]) -> tuple[set[int], set[int]]:
    train_ids = {int(v) for v in (manifest.get("train_clip_ids") or [])}
    test_ids = {int(v) for v in (manifest.get("test_clip_ids") or [])}
    if train_ids or test_ids:
        return train_ids, test_ids

    # Fallback for manifests where split IDs are only present in run_report successes.
    successes = (((manifest.get("run_report") or {}).get("successes")) or [])
    for row in successes:
        split = str(row.get("split") or "")
        clip_id = row.get("clip_id")
        if clip_id is None:
            continue
        if split == "train":
            train_ids.add(int(clip_id))
        elif split == "test":
            test_ids.add(int(clip_id))
    return train_ids, test_ids


def _split_sets_from_json(path: Path) -> tuple[set[int], set[int], set[str], set[str]]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    train_ids = {int(v) for v in (data.get("train_clip_ids") or [])}
    test_ids = {int(v) for v in (data.get("test_clip_ids") or [])}
    train_keys = {str(v) for v in (data.get("train_clip_keys") or [])}
    test_keys = {str(v) for v in (data.get("test_clip_keys") or [])}
    if not (train_ids or test_ids or train_keys or test_keys):
        raise RuntimeError(f"split JSON has no train/test IDs or keys: {path}")
    return train_ids, test_ids, train_keys, test_keys


def load_all_frames(
    run_dir: Path,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    all_df = _load_all_parquets(run_dir, feature_columns=feature_columns)
    return all_df, manifest


def load_train_test_frames(
    run_dir: Path,
    feature_columns: list[str],
    *,
    split_json: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    all_df, manifest = load_all_frames(run_dir, feature_columns)

    if split_json is not None:
        train_ids, test_ids, train_keys, test_keys = _split_sets_from_json(split_json)
        train_mask = all_df["clip_id"].astype(int).isin(train_ids) | all_df["clip_key"].astype(str).isin(train_keys)
        test_mask = all_df["clip_id"].astype(int).isin(test_ids) | all_df["clip_key"].astype(str).isin(test_keys)
    else:
        train_ids, test_ids = _split_sets_from_manifest(manifest)
        if not train_ids or not test_ids:
            raise RuntimeError(
                "manifest does not contain train/test clip IDs. Pass --split-json with explicit split assignment."
            )
        train_mask = all_df["clip_id"].astype(int).isin(train_ids)
        test_mask = all_df["clip_id"].astype(int).isin(test_ids)

    overlap = train_mask & test_mask
    if overlap.any():
        overlap_keys = sorted(all_df.loc[overlap, "clip_key"].astype(str).unique().tolist())
        raise RuntimeError(f"split assignment overlap for clips: {overlap_keys}")

    train_df = all_df.loc[train_mask].copy()
    test_df = all_df.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"empty split after assignment (train_rows={len(train_df)}, test_rows={len(test_df)}). "
            "Check split metadata or --split-json."
        )
    return train_df, test_df, manifest


def _fbeta_metric_key(beta: float) -> str:
    return f"f{beta:g}"


def test_predictions_to_csv(test_out: pd.DataFrame, *, split_name: str = "test") -> pd.DataFrame:
    """Narrow frame for eval viz: one row per eval frame, sorted by clip then frame."""
    col_map = {
        "source_id": "source_id",
        "clip_index": "clip_index",
        "clip_key": "clip_key",
        "clip_s3_uri": "clip_s3_uri",
        "frame_idx": "frame_idx",
        "timestamp_sec": "timestamp_sec",
        "pred_prob_playing": "prob_playing",
        "pred_playing": "pred_playing",
        "is_playing": "is_playing",
    }
    missing = [src for src in col_map if src not in test_out.columns]
    if missing:
        raise ValueError(f"test predictions missing columns for CSV export: {missing}")

    out = test_out[list(col_map.keys())].rename(columns=col_map)
    out["pred_playing"] = out["pred_playing"].astype(int)
    out["is_playing"] = out["is_playing"].astype(int)
    out["model_name"] = "xgboost"
    out["split_name"] = split_name
    return out.sort_values(["source_id", "clip_index", "frame_idx"], kind="stable").reset_index(drop=True)


def build_xgb_classifier(args: argparse.Namespace, *, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
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


def _resolve_decision_threshold(
    args: argparse.Namespace,
    *,
    estimator: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, dict[str, Any] | None]:
    if args.decision_threshold is not None:
        return float(args.decision_threshold), None
    if not args.tune_threshold:
        return 0.5, None

    tune_result = tune_threshold_clip_cv(
        estimator,
        X_train,
        y_train,
        groups,
        beta=args.f_beta,
        n_splits=args.cv_folds,
        random_state=args.random_seed,
    )
    return tune_result.best_threshold, threshold_tune_to_dict(tune_result)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    args: argparse.Namespace,
    manifest: dict[str, Any],
    wandb_run: Any | None = None,
    log_threshold_sweep: bool = False,
    eval_split_name: str = "test",
    split_method: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, XGBClassifier]:
    fill_cols = float_fillna_cols_for_features(feature_columns)

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()
    for col in fill_cols:
        X_train[col] = X_train[col].fillna(-1.0)
        X_test[col] = X_test[col].fillna(-1.0)

    y_train = train_df["is_playing"].astype(int).to_numpy()
    y_test = test_df["is_playing"].astype(int).to_numpy()
    groups = train_df["clip_key"].to_numpy()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    estimator = build_xgb_classifier(args, scale_pos_weight=scale_pos_weight)
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)

    decision_threshold, threshold_tuning = _resolve_decision_threshold(
        args,
        estimator=estimator,
        X_train=X_train_np,
        y_train=y_train,
        groups=groups,
    )

    model = build_xgb_classifier(args, scale_pos_weight=scale_pos_weight)
    model.fit(X_train_np, y_train)

    y_prob = model.predict_proba(X_test_np)[:, 1]
    y_pred = apply_threshold(y_prob, decision_threshold)
    y_pred_default = apply_threshold(y_prob, 0.5)

    fbeta_key = _fbeta_metric_key(args.f_beta)
    metrics = classification_metrics(y_test, y_pred, beta=args.f_beta)
    metrics_default = classification_metrics(y_test, y_pred_default, beta=args.f_beta)

    train_clips = sorted(train_df["clip_key"].unique().tolist())
    eval_clips = sorted(test_df["clip_key"].unique().tolist())
    eval_is_test = eval_split_name == "test"

    report: dict[str, Any] = {
        "feature_extraction_run_id": manifest.get("run_id"),
        "extractor_version": manifest.get("extractor_version"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "split_method": split_method if split_method is not None else manifest.get("split_method"),
        "eval_split_name": eval_split_name,
        "feature_subset": args.feature_subset,
        "n_features": len(feature_columns),
        "feature_columns": list(feature_columns),
        "n_train_rows": int(len(train_df)),
        "n_train_clips": len(train_clips),
        "train_clip_keys": train_clips,
        "class_balance_train": {"negative": neg_count, "positive": pos_count},
        "scale_pos_weight": scale_pos_weight,
        "f_beta": args.f_beta,
        "decision_threshold": decision_threshold,
        "threshold_tuning": threshold_tuning,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            fbeta_key: metrics[fbeta_key],
        },
        "metrics_at_threshold_0_5": {
            "accuracy": metrics_default["accuracy"],
            "precision": metrics_default["precision"],
            "recall": metrics_default["recall"],
            "f1": metrics_default["f1"],
            fbeta_key: metrics_default[fbeta_key],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_at_threshold_0_5": metrics_default["confusion_matrix"],
    }
    if eval_is_test:
        report["n_test_rows"] = int(len(test_df))
        report["n_test_clips"] = len(eval_clips)
        report["test_clip_keys"] = eval_clips
    else:
        report["n_val_rows"] = int(len(test_df))
        report["n_val_clips"] = len(eval_clips)
        report["val_clip_keys"] = eval_clips

    if wandb_run is not None:
        import wandb

        wandb_run.config.update(
            {
                "f_beta": args.f_beta,
                "decision_threshold": decision_threshold,
                "tune_threshold": args.tune_threshold,
                "cv_folds": args.cv_folds if args.tune_threshold and args.decision_threshold is None else None,
            }
        )
        wandb_run.log(
            {
                "test/accuracy": metrics["accuracy"],
                "test/precision": metrics["precision"],
                "test/recall": metrics["recall"],
                "test/f1": metrics["f1"],
                f"test/{fbeta_key}": metrics[fbeta_key],
                "test/fbeta_at_threshold_0_5": metrics_default[fbeta_key],
                "decision_threshold": decision_threshold,
            }
        )
        if threshold_tuning is not None:
            wandb_run.log({"threshold_tuning/oof_fbeta": threshold_tuning["oof_fbeta"]})
            if log_threshold_sweep:
                sweep = threshold_tuning["threshold_sweep"]
                wandb_run.log(
                    {
                        "threshold_sweep": wandb.plot.line(
                            wandb.Table(
                                data=[[row["threshold"], row["fbeta"]] for row in sweep],
                                columns=["threshold", f"oof_f{args.f_beta:g}"],
                            ),
                            "threshold",
                            f"oof_f{args.f_beta:g}",
                            title=f"OOF F{args.f_beta:g} vs threshold (train clips)",
                        ),
                    }
                )
        wandb_run.log(
            {
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_test.tolist(),
                    preds=y_pred.tolist(),
                    class_names=["downtime", "playing"],
                )
            }
        )

    id_cols = ["source_id", "clip_index", "clip_key", "frame_idx", "timestamp_sec", "is_playing"]
    if "clip_s3_uri" in test_df.columns:
        id_cols.insert(3, "clip_s3_uri")
    test_out = test_df[id_cols].copy()
    test_out["pred_playing"] = y_pred.astype(int)
    test_out["pred_prob_playing"] = y_prob.astype(float)
    test_out["decision_threshold"] = decision_threshold
    return report, test_out, model


def _log_model_artifact(wandb_run: Any, model: XGBClassifier, *, feature_run_id: str) -> None:
    import wandb

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "xgb_model.json"
        model.save_model(str(path))
        artifact = wandb.Artifact(
            name=f"xgb-playing-{feature_run_id}",
            type="model",
            description="XGBClassifier for is_playing",
        )
        artifact.add_file(str(path), name="model.json")
        wandb_run.log_artifact(artifact)


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    args = parse_args()
    run_dir = resolve_run_dir(args)
    feature_columns = active_feature_columns(args.feature_subset)

    train_df, test_df, manifest = load_train_test_frames(
        run_dir,
        feature_columns,
        split_json=args.split_json,
    )
    if train_df.empty or test_df.empty:
        raise SystemExit("train or test split is empty; need parquets in both train/ and test/")

    feature_run_id = str(manifest.get("run_id") or args.feature_run_id or run_dir.name)
    fbeta_key = _fbeta_metric_key(args.f_beta)
    wb_run: Any | None = None

    if args.wandb:
        import wandb

        entity = args.wandb_entity or wandb_entity()
        project = args.wandb_project or wandb_project()
        run_name = args.wandb_run_name or f"xgb-{feature_run_id}"

        init_kwargs: dict[str, Any] = {
            "entity": entity,
            "project": project,
            "group": wandb_group_for_feature_run(feature_run_id),
            "job_type": "train",
            "name": run_name,
            "tags": ["xgboost", "playing-detection", f"feature_run:{feature_run_id}"],
            "config": {
                "feature_run_id": feature_run_id,
                "feature_subset": args.feature_subset,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "random_seed": args.random_seed,
                "f_beta": args.f_beta,
                "tune_threshold": args.tune_threshold,
                "cv_folds": args.cv_folds,
                "extractor_version": manifest.get("extractor_version"),
                "feature_schema_version": manifest.get("feature_schema_version"),
                "split_method": manifest.get("split_method"),
            },
        }
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        wb_run = wandb.init(**init_kwargs)
        try:
            artifact = wb_run.use_artifact(f"{FEATURE_ARTIFACT_NAME}:{feature_run_id}", type="dataset")
            wb_run.config.update({"input_feature_artifact": artifact.qualified_name})
        except Exception as exc:  # noqa: BLE001
            wandb.termwarn(f"could not link input artifact {FEATURE_ARTIFACT_NAME}:{feature_run_id}: {exc}")

    report, test_preds, model = train_and_evaluate(
        train_df,
        test_df,
        feature_columns=feature_columns,
        args=args,
        manifest=manifest,
        wandb_run=wb_run,
        log_threshold_sweep=args.wandb_log_threshold_sweep,
    )

    print("=== Tabular XGBoost (feature-extraction run) ===")
    print(f"run_dir: {run_dir}")
    print(f"feature_run_id: {report.get('feature_extraction_run_id')}")
    print(f"feature_subset: {args.feature_subset} ({len(feature_columns)} columns)")
    print(f"train rows / clips: {report['n_train_rows']} / {report['n_train_clips']}")
    print(f"test rows / clips:  {report['n_test_rows']} / {report['n_test_clips']}")
    print(f"decision_threshold: {report['decision_threshold']:.4f} (F{args.f_beta:g} tuning on train clips)")
    if report.get("threshold_tuning"):
        print(f"  OOF F{args.f_beta:g} at chosen threshold: {report['threshold_tuning']['oof_fbeta']:.4f}")
    print("")
    print("metrics (test, tuned threshold):")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print("")
    print(f"metrics (test, threshold=0.5) — {fbeta_key}: {report['metrics_at_threshold_0_5'][fbeta_key]:.4f}")
    print("")
    cm = report["confusion_matrix"]
    print("confusion matrix (rows=true, cols=pred) labels=[0,1]:")
    print(f"  [{cm['matrix'][0][0]}, {cm['matrix'][0][1]}]")
    print(f"  [{cm['matrix'][1][0]}, {cm['matrix'][1][1]}]")

    if args.save_test_preds is not None:
        args.save_test_preds.parent.mkdir(parents=True, exist_ok=True)
        test_preds.to_parquet(args.save_test_preds, index=False)
        print(f"\nsaved test predictions: {args.save_test_preds}")

    if args.save_test_csv is not None:
        args.save_test_csv.parent.mkdir(parents=True, exist_ok=True)
        test_csv = test_predictions_to_csv(test_preds)
        test_csv.to_csv(args.save_test_csv, index=False)
        print(f"saved test predictions CSV: {args.save_test_csv} ({len(test_csv)} rows)")

    if args.save_report_json is not None:
        args.save_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved report: {args.save_report_json}")

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.save_model))
        print(f"saved model: {args.save_model}")

    if wb_run is not None:
        _log_model_artifact(wb_run, model, feature_run_id=feature_run_id)
        wb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
