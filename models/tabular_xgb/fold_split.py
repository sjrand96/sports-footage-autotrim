"""Source-grouped fold assignments for tabular XGB CV (video-level holdout)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS_CSV = REPO_ROOT / "data" / "5_fold_train_test_split.csv"


def load_source_folds(
    csv_path: Path,
    *,
    expected_folds: int = 5,
) -> dict[str, int]:
    """Return ``video_id`` (source) → ``fold_id`` from a fold assignment CSV."""
    if not csv_path.is_file():
        raise RuntimeError(f"missing fold assignment CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("video_id", "fold_id"):
        if col not in df.columns:
            raise RuntimeError(f"{csv_path} must have columns video_id and fold_id")

    source_folds: dict[str, int] = {}
    for row in df.itertuples(index=False):
        source_id = str(row.video_id).strip()
        fold_id = int(row.fold_id)
        if not source_id:
            continue
        if source_id in source_folds:
            raise RuntimeError(f"duplicate video_id in {csv_path}: {source_id!r}")
        if fold_id < 1 or fold_id > expected_folds:
            raise RuntimeError(
                f"fold_id must be 1..{expected_folds} in {csv_path}; got {fold_id} for {source_id!r}"
            )
        source_folds[source_id] = fold_id

    if not source_folds:
        raise RuntimeError(f"no video_id rows in {csv_path}")

    fold_ids = set(source_folds.values())
    if len(fold_ids) != expected_folds:
        raise RuntimeError(
            f"expected folds 1..{expected_folds} all present in {csv_path}; got {sorted(fold_ids)}"
        )
    return source_folds


def fold_val_sources(fold_id: int, source_folds: dict[str, int]) -> set[str]:
    val_sources = {sid for sid, fid in source_folds.items() if fid == fold_id}
    if not val_sources:
        raise RuntimeError(f"no sources assigned to fold {fold_id}")
    return val_sources


def fold_train_sources(fold_id: int, source_folds: dict[str, int]) -> set[str]:
    train_sources = {sid for sid, fid in source_folds.items() if fid != fold_id}
    if not train_sources:
        raise RuntimeError(f"no training sources left when holding out fold {fold_id}")
    return train_sources


def filter_to_fold_sources(all_df: pd.DataFrame, source_folds: dict[str, int]) -> pd.DataFrame:
    """Keep only rows whose source_id appears in the fold CSV."""
    allowed = set(source_folds)
    out = all_df[all_df["source_id"].astype(str).isin(allowed)].copy()
    if out.empty:
        raise RuntimeError("no frames left after filtering to fold CSV sources")
    return out


def split_frames_by_fold(
    all_df: pd.DataFrame,
    fold_id: int,
    source_folds: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For fold ``k``, return ``(train_df, val_df)`` grouped by source video."""
    val_sources = fold_val_sources(fold_id, source_folds)
    train_sources = fold_train_sources(fold_id, source_folds)

    train_df = all_df[all_df["source_id"].astype(str).isin(train_sources)].copy()
    val_df = all_df[all_df["source_id"].astype(str).isin(val_sources)].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError(
            f"empty train/val for fold {fold_id} "
            f"(train_rows={len(train_df)}, val_rows={len(val_df)})"
        )

    train_clips = set(train_df["clip_key"].astype(str).unique())
    val_clips = set(val_df["clip_key"].astype(str).unique())
    overlap = train_clips & val_clips
    if overlap:
        raise RuntimeError(f"train/val clip overlap for fold {fold_id}: {sorted(overlap)[:5]}")
    return train_df, val_df
