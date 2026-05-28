"""Datasets for LSTM training on cached per-frame features."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.lstm.model import WINDOW_RADIUS, WINDOW_SIZE

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
DEFAULT_TRAIN_CLIPS_CSV = REPO_ROOT / "data" / "train_clips.csv"
DEFAULT_TEST_CLIPS_CSV = REPO_ROOT / "data" / "test_clips.csv"
DEFAULT_FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"
DEFAULT_BOUNDARY_MARGIN = 0  # frames to drop from loss on each side of a 0/1 transition
DEFAULT_FRAME_STRIDE = 1  # train: use every Nth frame; test/eval always stride 1
# F-beta and Tversky: beta>1 weights recall over precision (beta=2 => F2).
DEFAULT_F_BETA = 2.0


def f_beta_score(precision: float, recall: float, beta: float = DEFAULT_F_BETA) -> float:
    """F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)."""
    b2 = float(beta) ** 2
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def tversky_coefficients_from_f_beta(f_beta: float = DEFAULT_F_BETA) -> tuple[float, float]:
    """Tversky FP/FN weights aligned with F_beta: alpha=1/(1+beta^2), beta_fn=beta^2/(1+beta^2)."""
    b2 = float(f_beta) ** 2
    denom = 1.0 + b2
    return 1.0 / denom, b2 / denom


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


def class_weight_ratio_from_counts(n_pos: int, n_neg: int) -> float:
    """Inverse class frequency n_neg / n_pos (playing vs inactive)."""
    if n_pos <= 0:
        raise ValueError("no positive (playing) frames in training count")
    return float(n_neg) / float(n_pos)


def bce_pos_weight_from_counts(n_pos: int, n_neg: int) -> float:
    """Deprecated alias for :func:`class_weight_ratio_from_counts`."""
    return class_weight_ratio_from_counts(n_pos, n_neg)


def tversky_coefficients_from_counts(n_pos: int, n_neg: int) -> tuple[float, float]:
    """Tversky (alpha=FP, beta=FN) with ``alpha + beta == 1`` from class frequencies.

    Prefer :func:`tversky_coefficients_from_f_beta` when matching evaluation F_beta.
    """
    total = n_pos + n_neg
    if total <= 0:
        raise ValueError("no labeled frames in training count")
    if n_pos <= 0:
        raise ValueError("no positive (playing) frames in training count")
    alpha = float(n_pos) / float(total)
    beta = float(n_neg) / float(total)
    return alpha, beta


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


def _read_clip_ids_csv(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            clip_id = row[0].strip()
            if clip_id:
                ids.append(clip_id)
    if not ids:
        raise RuntimeError(f"no clip ids in {path}")
    return ids


def load_train_test_clip_ids(
    *,
    train_csv: Path = DEFAULT_TRAIN_CLIPS_CSV,
    test_csv: Path = DEFAULT_TEST_CLIPS_CSV,
    labeled_clip_ids: set[str] | list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Load train/test clip ids from ``data/train_clips.csv`` and ``data/test_clips.csv``."""
    for path, name in ((train_csv, "train"), (test_csv, "test")):
        if not path.is_file():
            raise RuntimeError(
                f"missing {name} clip list: {path}\n"
                "Run: python data/train_test_split.py"
            )

    train_ids = _read_clip_ids_csv(train_csv)
    test_ids = _read_clip_ids_csv(test_csv)

    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(
            "train and test clip lists overlap: " + ", ".join(sorted(overlap)[:5])
            + (" ..." if len(overlap) > 5 else "")
        )
    if len(train_ids) != len(train_set):
        raise RuntimeError(f"duplicate clip ids in {train_csv}")
    if len(test_ids) != len(test_set):
        raise RuntimeError(f"duplicate clip ids in {test_csv}")

    if labeled_clip_ids is not None:
        labeled = set(labeled_clip_ids)
        missing = (train_set | test_set) - labeled
        if missing:
            bad = sorted(missing)[:5]
            raise RuntimeError(
                "clip ids in train/test CSV missing from frame labels: "
                + ", ".join(bad)
                + (" ..." if len(missing) > 5 else "")
            )
        unassigned = labeled - train_set - test_set
        if unassigned:
            sample = sorted(unassigned)[:5]
            raise RuntimeError(
                f"{len(unassigned)} labeled clip(s) not in train or test CSV "
                f"(e.g. {', '.join(sample)})"
            )

    return sorted(train_ids), sorted(test_ids)


def clip_to_source_map(csv_path: Path = DEFAULT_FRAME_LABELS_CSV) -> dict[str, str]:
    df = pd.read_csv(csv_path, usecols=["clip_id", "source_id"])
    rows = df.drop_duplicates(subset=["clip_id"])
    return {str(r.clip_id): str(r.source_id) for r in rows.itertuples(index=False)}


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
