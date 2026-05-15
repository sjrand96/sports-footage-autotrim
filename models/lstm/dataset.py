"""Datasets for LSTM training on cached per-frame features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
DEFAULT_FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"

WINDOW_SIZE = 30
WINDOW_RADIUS = 15  # T-15 .. T+14 inclusive


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
    ) -> None:
        self.backbone = backbone
        self.features_root = features_root
        self.labels_by_clip = labels_by_clip or load_labels_by_clip(labels_csv)

        self._features: dict[str, torch.Tensor] = {}
        self._window_idx: dict[str, np.ndarray] = {}
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
            for frame_idx in range(feats.shape[0]):
                self.samples.append((clip_id, frame_idx))

        if not self.samples:
            raise ValueError("empty dataset")

        self.feat_dim = int(next(iter(self._features.values())).shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip_id, frame_idx = self.samples[index]
        feats = self._features[clip_id]
        win_idx = self._window_idx[clip_id][frame_idx]
        label = int(self.labels_by_clip[clip_id][frame_idx])

        seq = torch.zeros(WINDOW_SIZE, self.feat_dim, dtype=torch.float32)
        for i, src_idx in enumerate(win_idx):
            if 0 <= src_idx < feats.shape[0]:
                seq[i] = feats[int(src_idx)]

        return {
            "seq": seq,
            "label": torch.tensor(label, dtype=torch.float32),
            "clip_id": clip_id,
            "frame_idx": frame_idx,
        }
