"""Datasets for LSTM training on cached per-frame features."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
DEFAULT_FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"

WINDOW_SIZE = 30
WINDOW_RADIUS = 15  # T-15 .. T+14 inclusive
DEFAULT_BOUNDARY_MARGIN = 0  # frames to drop from loss on each side of a 0/1 transition
DEFAULT_FRAME_STRIDE = 1  # train: use every Nth frame; test/eval always stride 1


def train_label_counts(
    train_clip_ids: list[str],
    labels_by_clip: dict[str, np.ndarray],
    *,
    boundary_margin: int = 0,
    frame_stride: int = 1,
) -> tuple[int, int]:
    """Count positive / negative frames in the training set (same stride + boundary mask as training)."""
    n_pos = 0
    n_neg = 0
    for clip_id in train_clip_ids:
        labels = labels_by_clip[clip_id]
        mask = loss_mask_for_labels(labels, boundary_margin)
        for frame_idx in range(0, len(labels), frame_stride):
            if not mask[frame_idx]:
                continue
            if labels[frame_idx]:
                n_pos += 1
            else:
                n_neg += 1
    return n_pos, n_neg


def bce_pos_weight_from_counts(n_pos: int, n_neg: int) -> float:
    """``BCEWithLogitsLoss(pos_weight)`` = n_neg / n_pos (inverse class frequency for playing)."""
    if n_pos <= 0:
        raise ValueError("no positive (playing) frames in training count")
    return float(n_neg) / float(n_pos)


def loss_mask_for_labels(labels: np.ndarray, margin: int) -> np.ndarray:
    """Per-frame mask: True = include in training loss, False = ignore near label transitions.

    For each transition between frames ``t`` and ``t+1`` (``labels[t] != labels[t+1]``),
    frames with indices in ``[t - margin + 1, t + margin]`` inclusive are excluded. With
    ``margin=15``, the last 15 frames before and first 15 after the boundary are ignored
    (30 frames total), matching the model's temporal window.
    """
    n = int(len(labels))
    if margin <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    include = np.ones(n, dtype=bool)
    for t in np.flatnonzero(labels[:-1] != labels[1:]):
        lo = max(0, int(t) - margin + 1)
        hi = min(n, int(t) + margin + 1)
        include[lo:hi] = False
    return include


def load_labels_by_clip(csv_path: Path = DEFAULT_FRAME_LABELS_CSV) -> dict[str, np.ndarray]:
    """Return ``clip_id`` → ``uint8`` label vector indexed by ``frame_idx``."""
    df = pd.read_csv(csv_path)
    out: dict[str, np.ndarray] = {}
    for clip_id, group in df.groupby("clip_id", sort=False):
        group = group.sort_values("frame_idx")
        frame_idx = group["frame_idx"].to_numpy()
        expected = np.arange(len(frame_idx))
        if not np.array_equal(frame_idx, expected):
            raise ValueError(f"non-contiguous frame_idx for clip_id={clip_id!r}")
        out[str(clip_id)] = group["is_playing"].to_numpy(dtype=np.uint8)
    return out


def list_clip_ids(csv_path: Path = DEFAULT_FRAME_LABELS_CSV) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["clip_id"])
    return sorted(df["clip_id"].astype(str).unique())


def clip_to_source_map(csv_path: Path = DEFAULT_FRAME_LABELS_CSV) -> dict[str, str]:
    df = pd.read_csv(csv_path, usecols=["clip_id", "source_id"])
    rows = df.drop_duplicates(subset=["clip_id"])
    return {str(r.clip_id): str(r.source_id) for r in rows.itertuples(index=False)}


def group_clips_by_source(
    clip_ids: list[str], source_map: dict[str, str]
) -> dict[str, list[str]]:
    """Map ``source_id`` → sorted ``clip_id`` list for clips in ``clip_ids``."""
    by_src: dict[str, list[str]] = {}
    for cid in clip_ids:
        src = source_map.get(cid)
        if src is None:
            raise KeyError(f"clip_id missing from labels/source map: {cid!r}")
        by_src.setdefault(str(src), []).append(cid)
    for src in by_src:
        by_src[src].sort()
    return by_src


def split_clip_ids_by_source(
    clip_ids: list[str],
    source_map: dict[str, str],
    *,
    fixed_test_sources: tuple[str, ...] | None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Split clips so no ``source_id`` appears in both train and test.

    - If ``fixed_test_sources`` is non-empty: test = all clips from those sources;
      train = remaining clips (no proportional holdout).
    - If ``fixed_test_sources`` is ``None`` or empty: ``train_test_split`` is applied to
      **distinct source ids** (not clips), then clips are assigned by source.

    Returns ``(train_clip_ids, test_clip_ids, info)`` where ``info`` documents the split.
    """
    if not clip_ids:
        raise ValueError("empty clip_ids")
    by_src = group_clips_by_source(clip_ids, source_map)
    all_sources = sorted(by_src)

    if fixed_test_sources:
        missing = [s for s in fixed_test_sources if s not in by_src]
        if missing:
            raise RuntimeError(
                "fixed test source(s) have no clips in this run: "
                + ", ".join(repr(m) for m in missing)
            )
        test_sources = sorted(fixed_test_sources)
        train_sources = sorted(s for s in all_sources if s not in set(test_sources))
        if not train_sources:
            raise RuntimeError(
                "train set empty after holding out test sources "
                + ", ".join(test_sources)
            )
        mode = "fixed_test_sources"
    else:
        if len(all_sources) < 2:
            raise RuntimeError(
                f"need at least 2 distinct source_id values for a random source split; "
                f"found {len(all_sources)}"
            )
        train_sources, test_sources = train_test_split(
            all_sources,
            test_size=test_size,
            random_state=random_state,
        )
        train_sources = sorted(train_sources)
        test_sources = sorted(test_sources)
        mode = "random_by_source"

    train_ids: list[str] = []
    test_ids: list[str] = []
    for s in train_sources:
        train_ids.extend(by_src[s])
    for s in test_sources:
        test_ids.extend(by_src[s])
    info: dict[str, Any] = {
        "mode": mode,
        "train_sources": train_sources,
        "test_sources": test_sources,
        "n_sources_train": len(train_sources),
        "n_sources_test": len(test_sources),
    }
    return sorted(train_ids), sorted(test_ids), info


def _per_source_test_count(n_clips: int, test_size: float) -> int:
    """Clips to hold out for one source; keeps >=1 train clip when n_clips > 1."""
    if n_clips <= 1:
        return 0
    n_test = max(1, int(round(n_clips * test_size)))
    return min(n_test, n_clips - 1)


def _source_split_seed(source_id: str, base_seed: int) -> int:
    return (base_seed + zlib.crc32(source_id.encode("utf-8"))) & 0x7FFFFFFF


def split_clip_ids_stratified_by_source(
    clip_ids: list[str],
    source_map: dict[str, str],
    *,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Split clips so each ``source_id`` contributes ~``test_size`` of its clips to test.

    Every video with 2+ clips has at least one train and one test clip. Single-clip
    sources stay in train only.
    """
    if not clip_ids:
        raise ValueError("empty clip_ids")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    by_src = group_clips_by_source(clip_ids, source_map)
    train_ids: list[str] = []
    test_ids: list[str] = []
    per_source: dict[str, dict[str, Any]] = {}

    for source_id in sorted(by_src):
        clips = by_src[source_id]
        n = len(clips)
        n_test = _per_source_test_count(n, test_size)
        if n_test == 0:
            train_ids.extend(clips)
            per_source[source_id] = {
                "n_clips": n,
                "n_test": 0,
                "test_clip_ids": [],
            }
            continue
        rng = np.random.RandomState(_source_split_seed(source_id, random_state))
        perm = rng.permutation(n)
        test_clips = [clips[i] for i in perm[:n_test]]
        train_clips = [clips[i] for i in perm[n_test:]]
        test_ids.extend(test_clips)
        train_ids.extend(train_clips)
        per_source[source_id] = {
            "n_clips": n,
            "n_test": n_test,
            "test_clip_ids": sorted(test_clips),
        }

    if not train_ids:
        raise RuntimeError("train set empty after stratified split")
    if not test_ids:
        raise RuntimeError(
            "test set empty after stratified split; every source may have only one clip"
        )

    info: dict[str, Any] = {
        "mode": "stratified_by_source",
        "test_size": test_size,
        "random_state": random_state,
        "n_sources": len(by_src),
        "per_source": per_source,
    }
    return sorted(train_ids), sorted(test_ids), info


def load_clip_features(
    clip_id: str,
    *,
    backbone: str,
    features_root: Path = DEFAULT_FEATURES_ROOT,
) -> torch.Tensor:
    """Load ``(num_frames, feat_dim)`` float tensor from cache."""
    path = features_root / backbone / f"{clip_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing feature cache: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    features = data["features"]
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    return features.to(torch.float32)


def build_window_indices(num_frames: int) -> np.ndarray:
    """Shape ``(num_frames, WINDOW_SIZE)`` global frame indices (negative if padded)."""
    idx = np.arange(num_frames, dtype=np.int64)
    offsets = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS, dtype=np.int64)
    return idx[:, None] + offsets[None, :]


class FeatureWindowDataset(Dataset):
    """One sample per (clip, target frame): 30-step feature window + center label."""

    def __init__(
        self,
        clip_ids: list[str],
        *,
        backbone: str,
        labels_by_clip: dict[str, np.ndarray] | None = None,
        features_root: Path = DEFAULT_FEATURES_ROOT,
        labels_csv: Path = DEFAULT_FRAME_LABELS_CSV,
        boundary_margin: int = DEFAULT_BOUNDARY_MARGIN,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
    ) -> None:
        self.backbone = backbone
        self.features_root = features_root
        self.labels_by_clip = labels_by_clip or load_labels_by_clip(labels_csv)
        self.boundary_margin = int(boundary_margin)
        self.frame_stride = int(frame_stride)
        if self.boundary_margin < 0:
            raise ValueError(f"boundary_margin must be >= 0, got {boundary_margin}")
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        self._features: dict[str, torch.Tensor] = {}
        self._window_idx: dict[str, np.ndarray] = {}
        self._loss_mask: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        for clip_id in clip_ids:
            if clip_id not in self.labels_by_clip:
                raise KeyError(f"no labels for clip_id={clip_id!r}")
            labels = self.labels_by_clip[clip_id]
            feats = load_clip_features(clip_id, backbone=backbone, features_root=features_root)
            if feats.shape[0] != len(labels):
                raise ValueError(
                    f"length mismatch {clip_id}: features={feats.shape[0]} labels={len(labels)}"
                )
            self._features[clip_id] = feats
            self._window_idx[clip_id] = build_window_indices(feats.shape[0])
            self._loss_mask[clip_id] = loss_mask_for_labels(labels, self.boundary_margin)
            for frame_idx in range(0, feats.shape[0], self.frame_stride):
                self.samples.append((clip_id, frame_idx))

        if not self.samples:
            raise ValueError("empty dataset")

        self.feat_dim = int(next(iter(self._features.values())).shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    def loss_frame_counts(self) -> tuple[int, int]:
        """Return ``(n_included, n_total)`` frames with loss mask True."""
        included = 0
        for clip_id, frame_idx in self.samples:
            if self._loss_mask[clip_id][frame_idx]:
                included += 1
        return included, len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip_id, frame_idx = self.samples[index]
        feats = self._features[clip_id]
        win_idx = self._window_idx[clip_id][frame_idx]
        label = int(self.labels_by_clip[clip_id][frame_idx])

        seq = torch.zeros(WINDOW_SIZE, self.feat_dim, dtype=torch.float32)
        for i, src_idx in enumerate(win_idx):
            if 0 <= src_idx < feats.shape[0]:
                seq[i] = feats[int(src_idx)]

        loss_mask = bool(self._loss_mask[clip_id][frame_idx])
        return {
            "seq": seq,
            "label": torch.tensor(label, dtype=torch.float32),
            "loss_mask": torch.tensor(1.0 if loss_mask else 0.0, dtype=torch.float32),
            "clip_id": clip_id,
            "frame_idx": frame_idx,
        }
