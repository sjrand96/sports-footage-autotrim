"""Dataset for volleyball playtime vs downtime windows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSample:
    clip_id: str
    clip_path: str
    window_start_sec: float
    window_end_sec: float
    label: int
    source_id: Optional[str]
    match_id: Optional[str]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_indices(timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.where((timestamps >= start) & (timestamps <= end))[0]


def _safe_npz_load(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _ensure_length(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] == 0:
        return np.zeros((length,) + arr.shape[1:], dtype=arr.dtype)
    if arr.shape[0] > length:
        return arr[:length]
    pad = np.repeat(arr[-1:], length - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _load_frames(clip_path: str, start: float, end: float, num_frames: int) -> np.ndarray:
    try:
        from decord import VideoReader  # type: ignore
    except ImportError as exc:
        raise ImportError("decord is required for raw-frame loading") from exc

    vr = VideoReader(clip_path)
    fps = vr.get_avg_fps()
    start_idx = int(start * fps)
    end_idx = int(end * fps)
    end_idx = max(start_idx + 1, end_idx)
    indices = np.linspace(start_idx, end_idx - 1, num=num_frames)
    indices = np.clip(indices.astype(int), 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()
    return frames


class VideoWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        features_dir: Optional[str] = None,
        pose_dir: Optional[str] = None,
        num_frames: int = 16,
        use_raw_frames: bool = False,
    ) -> None:
        self.rows = _load_jsonl(manifest_path)
        self.features_dir = features_dir
        self.pose_dir = pose_dir
        self.num_frames = num_frames
        self.use_raw_frames = use_raw_frames

    def __len__(self) -> int:
        return len(self.rows)

    def _load_feature_window(self, row: Dict[str, Any]) -> torch.Tensor:
        clip_id = row.get("clip_id") or os.path.splitext(os.path.basename(row["clip_path"]))[0]
        feat_path = os.path.join(self.features_dir or "", f"{clip_id}.npz")
        data = _safe_npz_load(feat_path)
        timestamps = data["timestamps_sec"].astype(float)
        features = data["features"].astype(np.float32)
        indices = _sample_indices(timestamps, row["window_start_sec"], row["window_end_sec"])
        window_feats = features[indices] if len(indices) > 0 else features[:1]
        window_feats = _ensure_length(window_feats, self.num_frames)

        if self.pose_dir:
            pose_path = os.path.join(self.pose_dir, f"{clip_id}.npz")
            pose_data = _safe_npz_load(pose_path)
            pose_feats = pose_data["pose"].astype(np.float32)
            pose_timestamps = pose_data["timestamps_sec"].astype(float)
            pose_indices = _sample_indices(pose_timestamps, row["window_start_sec"], row["window_end_sec"])
            pose_window = pose_feats[pose_indices] if len(pose_indices) > 0 else pose_feats[:1]
            pose_window = _ensure_length(pose_window, self.num_frames)
            pose_window = pose_window.reshape(pose_window.shape[0], -1)
            window_feats = np.concatenate([window_feats, pose_window], axis=-1)

        return torch.from_numpy(window_feats)

    def _load_raw_window(self, row: Dict[str, Any]) -> torch.Tensor:
        frames = _load_frames(
            row["clip_path"],
            float(row["window_start_sec"]),
            float(row["window_end_sec"]),
            self.num_frames,
        )
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.from_numpy(frames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.rows[index]
        if self.use_raw_frames:
            features = self._load_raw_window(row)
        else:
            if not self.features_dir:
                raise ValueError("features_dir is required when use_raw_frames is False")
            features = self._load_feature_window(row)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        meta = {
            "clip_id": row.get("clip_id"),
            "clip_path": row.get("clip_path"),
            "source_id": row.get("source_id"),
            "match_id": row.get("match_id"),
            "window_start_sec": row.get("window_start_sec"),
            "window_end_sec": row.get("window_end_sec"),
        }
        return features, label, meta


def collate_windows(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    features, labels, metas = zip(*batch)
    return torch.stack(features, dim=0), torch.stack(labels, dim=0), list(metas)
