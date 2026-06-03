#!/usr/bin/env python3
"""SHAP analysis for 5-fold tabular XGBoost ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.core.feature_columns import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_BASE,
    FEATURE_COLUMNS_CHUNK1_SPATIAL,
    FEATURE_COLUMNS_FLOW_MOTION,
    active_feature_columns,
    float_fillna_cols_for_features,
)
from models.tabular_xgb.train import (  # noqa: E402
    _load_all_parquets,
    load_all_frames,
    resolve_run_dir,
)

DEFAULT_CV_RUN = REPO_ROOT / "models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold"
DEFAULT_FEATURE_RUN = REPO_ROOT / "feature_extraction/_runs/all_clips_features_v2"
DEFAULT_FOLDS_CSV = REPO_ROOT / "data/5_fold_train_test_split.csv"
DEFAULT_OUT = DEFAULT_CV_RUN / "shap"

FEATURE_GROUPS: dict[str, list[str]] = {
    "player_counts": [
        "n_players_total",
        "n_front_row",
        "n_back_row",
        "n_camera_side",
        "n_opposite_side",
        "n_pose_instances_raw",
    ],
    "pose_action": [
        "median_nearest_neighbor_dist",
        "hands_above_head_count",
        "knee_angle_mean_deg",
        "knee_angle_min_deg",
        "squat_count",
        "squat_ratio",
        "wrists_above_shoulder_count",
        "high_five_pair_count",
    ],
    "spatial": list(FEATURE_COLUMNS_CHUNK1_SPATIAL),
    "flow_motion": list(FEATURE_COLUMNS_FLOW_MOTION),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-run", type=Path, default=DEFAULT_CV_RUN)
    p.add_argument("--feature-run-dir", type=Path, default=DEFAULT_FEATURE_RUN)
    p.add_argument("--folds-csv", type=Path, default=DEFAULT_FOLDS_CSV)
    p.add_argument("--feature-subset", choices=("all", "base"), default="all")
    p.add_argument("--sample-size", type=int, default=4000, help="Stratified sample per dataset.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--ood-source",
        type=str,
        default="RCbQVAISMcU",
        help="Optional held-out source for OOD SHAP comparison (empty to skip).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    fill_cols = float_fillna_cols_for_features(feature_columns)
    X = df[feature_columns].copy()
    for col in fill_cols:
        X[col] = X[col].fillna(-1.0)
    return X.to_numpy(dtype=np.float32)


def load_fold_models(cv_run: Path) -> list[XGBClassifier]:
    paths = sorted(cv_run.glob("fold-*/xgb_model.json"))
    if not paths:
        raise FileNotFoundError(f"no fold models under {cv_run}")
    models: list[XGBClassifier] = []
    for path in paths:
        model = XGBClassifier()
        model.load_model(str(path))
        models.append(model)
    return models


def cv_source_ids(folds_csv: Path) -> set[str]:
    df = pd.read_csv(folds_csv)
    return set(df["video_id"].astype(str).unique())


def stratified_sample(
    df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    label_col: str = "is_playing",
) -> pd.DataFrame:
    if len(df) <= n:
        return df.sort_values(["clip_key", "frame_idx"], kind="stable").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    labels = df[label_col].astype(int)
    n_classes = labels.nunique()
    per_class = max(1, n // n_classes)
    for value in sorted(labels.unique()):
        chunk = df[labels == value]
        take = min(len(chunk), per_class)
        parts.append(chunk.sample(n=take, random_state=seed + int(value)))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        remaining = df.drop(out.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(out), len(remaining)), random_state=seed)
        out = pd.concat([out, extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ensemble_shap_values(models: list[XGBClassifier], X: np.ndarray) -> np.ndarray:
    """Mean SHAP across fold models (matches ensemble proba averaging)."""
    explainers = [shap.TreeExplainer(m) for m in models]
    stacked = np.stack([e.shap_values(X) for e in explainers], axis=0)
    if stacked.ndim == 4:
        # binary: (n_models, n_samples, n_features, 2) -> take positive class
        stacked = stacked[:, :, :, 1]
    return stacked.mean(axis=0)


def xgb_gain_importance(models: list[XGBClassifier], feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for i, model in enumerate(models, start=1):
        score = model.get_booster().get_score(importance_type="gain")
        for feat_idx, gain in score.items():
            if feat_idx.startswith("f"):
                idx = int(feat_idx[1:])
                name = feature_names[idx]
            else:
                name = feat_idx
            rows.append({"fold": i, "feature": name, "gain": float(gain)})
    wide = pd.DataFrame(rows)
    summary = (
        wide.groupby("feature", as_index=False)["gain"]
        .agg(mean_gain="mean", std_gain="std")
        .sort_values("mean_gain", ascending=False)
    )
    return summary


def importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    models: list[XGBClassifier],
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    gain = xgb_gain_importance(models, feature_names).rename(
        columns={"mean_gain": "xgb_gain_mean", "std_gain": "xgb_gain_std"}
    )
    return df.merge(gain, on="feature", how="left")


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for group, cols in FEATURE_GROUPS.items():
        sub = importance[importance["feature"].isin(cols)]
        rows.append(
            {
                "group": group,
                "n_features": len(cols),
                "mean_abs_shap_sum": float(sub["mean_abs_shap"].sum()),
                "mean_abs_shap_mean": float(sub["mean_abs_shap"].mean()) if len(sub) else 0.0,
                "xgb_gain_sum": float(sub["xgb_gain_mean"].fillna(0).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap_sum", ascending=False)
    out["shap_frac"] = out["mean_abs_shap_sum"] / out["mean_abs_shap_sum"].sum()
    return out


def plot_importance_bar(importance: pd.DataFrame, *, out_path: Path, dpi: int, title: str) -> None:
    top = importance.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4A7FD4", edgecolor="white")
    ax.set_xlabel("mean |SHAP| (5-fold ensemble)")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_group_bar(groups: pd.DataFrame, *, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = groups.sort_values("mean_abs_shap_sum", ascending=True)
    ax.barh(order["group"], order["mean_abs_shap_sum"], color="#E8B923", edgecolor="white")
    ax.set_xlabel("sum of mean |SHAP| across features in group")
    ax.set_title("SHAP by feature group", fontsize=12, fontweight="bold", loc="left")
    for i, (_, row) in enumerate(order.iterrows()):
        ax.text(row["mean_abs_shap_sum"] + 0.002, i, f"{row['shap_frac']:.0%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    *,
    out_path: Path,
    dpi: int,
    title: str,
    max_display: int = 20,
) -> None:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        plot_size=None,
    )
    plt.title(title, fontsize=12, fontweight="bold", loc="left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_playing_vs_downtime(
    shap_values: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    out_path: Path,
    dpi: int,
) -> None:
    top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:12]
    top_names = [feature_names[i] for i in top_idx]
    playing = y.astype(bool)
    rows: list[dict[str, float | str]] = []
    for i in top_idx:
        name = feature_names[i]
        rows.append({"feature": name, "label": "playing", "mean_abs_shap": float(np.abs(shap_values[playing, i]).mean())})
        rows.append(
            {"feature": name, "label": "downtime", "mean_abs_shap": float(np.abs(shap_values[~playing, i]).mean())}
        )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="feature", columns="label", values="mean_abs_shap").loc[top_names]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(top_names))
    w = 0.36
    ax.bar(x - w / 2, pivot["playing"], width=w, label="GT playing", color="#2ca02c")
    ax.bar(x + w / 2, pivot["downtime"], width=w, label="GT downtime", color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("Top features: SHAP magnitude by ground-truth label", fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ood_comparison(
    in_importance: pd.DataFrame,
    ood_importance: pd.DataFrame,
    *,
    out_path: Path,
    dpi: int,
    top_n: int = 15,
) -> None:
    top_feats = in_importance.head(top_n)["feature"].tolist()
    a = in_importance.set_index("feature").reindex(top_feats)["mean_abs_shap"].fillna(0)
    b = ood_importance.set_index("feature").reindex(top_feats)["mean_abs_shap"].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(top_feats))
    w = 0.36
    ax.bar(x - w / 2, a, width=w, label="CV sources (in-distribution)", color="#4A7FD4")
    ax.bar(x + w / 2, b, width=w, label="OOD source", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean |SHAP|")
    ax.set_title("Top feature SHAP: in-distribution vs OOD", fontsize=11, fontweight="bold", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = active_feature_columns(args.feature_subset)
    models = load_fold_models(args.cv_run.resolve())
    print(f"loaded {len(models)} fold models from {args.cv_run}")

    all_df, _ = load_all_frames(args.feature_run_dir.resolve(), feature_columns)
    cv_sources = cv_source_ids(args.folds_csv)
    in_df = all_df[all_df["source_id"].astype(str).isin(cv_sources)].copy()
    print(f"CV frames: {len(in_df):,} from {in_df['source_id'].nunique()} sources")

    sample = stratified_sample(in_df, n=args.sample_size, seed=args.random_seed)
    X = prepare_feature_matrix(sample, feature_columns)
    y = sample["is_playing"].astype(int).to_numpy()
    print(f"computing ensemble SHAP on {len(sample):,} stratified CV frames...")
    shap_cv = ensemble_shap_values(models, X)
    imp_cv = importance_table(shap_cv, feature_columns, models=models)
    groups_cv = group_importance(imp_cv)

    imp_cv.to_csv(out_dir / "shap_importance_cv.csv", index=False)
    groups_cv.to_csv(out_dir / "shap_group_importance_cv.csv", index=False)

    plot_importance_bar(
        imp_cv,
        out_path=out_dir / "shap_importance_bar.jpg",
        dpi=args.dpi,
        title="XGBoost 5-fold ensemble — mean |SHAP| (CV sample)",
    )
    plot_group_bar(groups_cv, out_path=out_dir / "shap_group_importance.jpg", dpi=args.dpi)
    plot_beeswarm(
        shap_cv,
        X,
        feature_columns,
        out_path=out_dir / "shap_beeswarm_cv.jpg",
        dpi=args.dpi,
        title="SHAP beeswarm — CV sample (5-fold ensemble)",
    )
    plot_playing_vs_downtime(
        shap_cv,
        X,
        y,
        feature_columns,
        out_path=out_dir / "shap_playing_vs_downtime.jpg",
        dpi=args.dpi,
    )

    report: dict[str, object] = {
        "cv_run": str(args.cv_run),
        "feature_run_dir": str(args.feature_run_dir),
        "n_features": len(feature_columns),
        "sample_size_cv": len(sample),
        "top_features_cv": imp_cv.head(15).to_dict(orient="records"),
        "feature_groups_cv": groups_cv.to_dict(orient="records"),
    }

    if args.ood_source:
        ood_df = all_df[all_df["source_id"].astype(str) == args.ood_source]
        if ood_df.empty:
            pq_root = args.feature_run_dir / "parquet"
            files = sorted(pq_root.glob(f"{args.ood_source}_*.parquet"))
            if files:
                ood_df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if not ood_df.empty:
            ood_sample = stratified_sample(ood_df, n=min(args.sample_size, len(ood_df)), seed=args.random_seed + 1)
            X_ood = prepare_feature_matrix(ood_sample, feature_columns)
            print(f"computing ensemble SHAP on {len(ood_sample):,} {args.ood_source} frames...")
            shap_ood = ensemble_shap_values(models, X_ood)
            imp_ood = importance_table(shap_ood, feature_columns, models=models)
            imp_ood.to_csv(out_dir / f"shap_importance_{args.ood_source}.csv", index=False)
            plot_importance_bar(
                imp_ood,
                out_path=out_dir / f"shap_importance_bar_{args.ood_source}.jpg",
                dpi=args.dpi,
                title=f"XGBoost 5-fold ensemble — mean |SHAP| ({args.ood_source} OOD)",
            )
            plot_beeswarm(
                shap_ood,
                X_ood,
                feature_columns,
                out_path=out_dir / f"shap_beeswarm_{args.ood_source}.jpg",
                dpi=args.dpi,
                title=f"SHAP beeswarm — {args.ood_source} OOD",
            )
            plot_ood_comparison(
                imp_cv,
                imp_ood,
                out_path=out_dir / f"shap_in_vs_ood_{args.ood_source}.jpg",
                dpi=args.dpi,
            )
            report["top_features_ood"] = imp_ood.head(15).to_dict(orient="records")
            report["ood_source"] = args.ood_source
            report["sample_size_ood"] = len(ood_sample)

    (out_dir / "shap_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTop 10 features (mean |SHAP| on CV sample):")
    for _, row in imp_cv.head(10).iterrows():
        print(f"  {row['feature']:40s}  {row['mean_abs_shap']:.4f}")

    print("\nFeature group contribution (sum mean |SHAP|):")
    for _, row in groups_cv.iterrows():
        print(f"  {row['group']:14s}  {row['mean_abs_shap_sum']:.4f}  ({row['shap_frac']:.0%})")

    print(f"\nWrote plots and CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
