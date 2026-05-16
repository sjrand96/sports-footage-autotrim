"""Dataset for volleyball playtime vs downtime windows."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.e2e_feature_columns import active_feature_columns
from src.data.s3_cache import download_s3_uri, is_s3_uri


@dataclass
class WindowSample:
    clip_id: str
    clip_path: str
    window_start_sec: float
    window_end_sec: float
    label: int
    source_id: Optional[str]
    match_id: Optional[str]


def _load_jsonl(path: str, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if is_s3_uri(path):
        path = download_s3_uri(path, cache_dir or ".s3_cache")
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
    except ImportError:
        return _load_frames_opencv(clip_path, start, end, num_frames)

    vr = VideoReader(clip_path)
    fps = vr.get_avg_fps()
    start_idx = int(start * fps)
    end_idx = int(end * fps)
    end_idx = max(start_idx + 1, end_idx)
    indices = np.linspace(start_idx, end_idx - 1, num=num_frames)
    indices = np.clip(indices.astype(int), 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def _load_frames_opencv(clip_path: str, start: float, end: float, num_frames: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Either decord or opencv-python is required for raw-frame loading") from exc

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video clip: {clip_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_idx = int(start * fps)
        end_idx = max(start_idx + 1, int(end * fps))
        if frame_count > 0:
            start_idx = min(max(0, start_idx), frame_count - 1)
            end_idx = min(max(start_idx + 1, end_idx), frame_count)
        indices = np.linspace(start_idx, end_idx - 1, num=num_frames)
        indices = np.clip(indices.astype(int), 0, max(0, frame_count - 1)) if frame_count > 0 else indices.astype(int)

        frames: List[np.ndarray] = []
        last_frame: Optional[np.ndarray] = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if last_frame is None:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    frame = last_frame
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = frame
            frames.append(frame)
        return np.stack(frames, axis=0)
    finally:
        cap.release()


def _safe_clip_id(row: Dict[str, Any]) -> str:
    clip_id = row.get("clip_id")
    if clip_id:
        return str(clip_id)
    base = os.path.basename(str(row["clip_path"]))
    return os.path.splitext(base)[0]


def _candidate_feature_names(clip_id: str) -> List[str]:
    return [f"{clip_id}_features.parquet", f"{clip_id}.parquet"]


def _clip_index_from_clip_id(clip_id: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", clip_id)
    if not match:
        return None
    return int(match.group(1))


class VideoWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        features_dir: Optional[str] = None,
        pose_dir: Optional[str] = None,
        e2e_features_dir: Optional[str] = None,
        e2e_feature_subset: str = "all",
        s3_cache_dir: Optional[str] = None,
        clip_cache_dir: Optional[str] = None,
        clip_duration_sec: float = 60.0,
        num_frames: int = 16,
        use_raw_frames: bool = False,
    ) -> None:
        self.s3_cache_dir = s3_cache_dir or os.path.join(os.getcwd(), "data", "s3_cache")
        self.clip_cache_dir = clip_cache_dir or os.path.join(self.s3_cache_dir, "clips")
        self.rows = _load_jsonl(manifest_path, self.s3_cache_dir)
        self.features_dir = features_dir
        self.pose_dir = pose_dir
        self.e2e_features_dir = e2e_features_dir
        self.e2e_feature_columns = active_feature_columns(e2e_feature_subset)
        self.clip_duration_sec = clip_duration_sec
        self.num_frames = num_frames
        self.use_raw_frames = use_raw_frames
        self._e2e_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_feature_window(self, row: Dict[str, Any]) -> torch.Tensor:
        clip_id = _safe_clip_id(row)
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

    def _feature_parquet_path(self, clip_id: str) -> str:
        if not self.e2e_features_dir:
            raise ValueError("e2e_features_dir is not configured")
        root = self.e2e_features_dir.rstrip("/")
        for name in _candidate_feature_names(clip_id):
            if is_s3_uri(root):
                return download_s3_uri(f"{root}/{name}", self.s3_cache_dir)
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
        return os.path.join(root, _candidate_feature_names(clip_id)[0])

    def _load_e2e_dataframe(self, row: Dict[str, Any]) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas and a parquet engine are required for E2E parquet features") from exc

        clip_id = _safe_clip_id(row)
        if clip_id not in self._e2e_cache:
            path = self._feature_parquet_path(clip_id)
            df = pd.read_parquet(path)
            missing = [col for col in self.e2e_feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"{path} is missing expected E2E feature columns: {missing}")
            self._e2e_cache[clip_id] = df
        return self._e2e_cache[clip_id]

    def _load_e2e_frame_features(self, row: Dict[str, Any]) -> torch.Tensor:
        clip_id = _safe_clip_id(row)
        df = self._load_e2e_dataframe(row)
        if "timestamp_sec" not in df.columns:
            raise ValueError(f"E2E feature parquet for {clip_id} is missing timestamp_sec")

        timestamps = df["timestamp_sec"].to_numpy(dtype=float)
        start = float(row["window_start_sec"])
        end = float(row["window_end_sec"])

        if timestamps.size and timestamps.max(initial=0.0) > self.clip_duration_sec + 1.0:
            clip_index = row.get("clip_index")
            if clip_index is None and "clip_index" in df.columns and len(df["clip_index"].dropna()) > 0:
                clip_index = int(df["clip_index"].dropna().iloc[0])
            if clip_index is None:
                clip_index = _clip_index_from_clip_id(clip_id)
            if clip_index is not None:
                timestamps = timestamps - (int(clip_index) - 1) * self.clip_duration_sec

        values_df = df[self.e2e_feature_columns].copy()
        for col in self.e2e_feature_columns:
            fill = 0.0 if col.startswith("n_") or col.endswith("_count") else -1.0
            values_df[col] = values_df[col].fillna(fill)
        values = values_df.to_numpy(dtype=np.float32)

        indices = _sample_indices(timestamps, start, end)
        window_feats = values[indices] if len(indices) > 0 else values[:1]
        window_feats = _ensure_length(window_feats, self.num_frames)
        return torch.from_numpy(window_feats)

    def _clip_path_for_row(self, row: Dict[str, Any]) -> str:
        clip_path = row.get("clip_path") or row.get("clip_s3_uri")
        if not clip_path and self.e2e_features_dir:
            df = self._load_e2e_dataframe(row)
            if "clip_s3_uri" in df.columns and len(df["clip_s3_uri"].dropna()) > 0:
                clip_path = str(df["clip_s3_uri"].dropna().iloc[0])
        if not clip_path:
            raise ValueError(
                f"Manifest row for clip_id={row.get('clip_id')!r} has no clip_path, "
                "and the feature parquet did not provide clip_s3_uri."
            )
        return str(clip_path)

    def _load_raw_window(self, row: Dict[str, Any]) -> torch.Tensor:
        clip_path = self._clip_path_for_row(row)
        if is_s3_uri(clip_path):
            clip_path = download_s3_uri(clip_path, self.clip_cache_dir)
        frames = _load_frames(
            clip_path,
            float(row["window_start_sec"]),
            float(row["window_end_sec"]),
            self.num_frames,
        )
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.from_numpy(frames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor | Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        row = self.rows[index]
        tensors: Dict[str, torch.Tensor] = {}
        if self.use_raw_frames:
            tensors["video"] = self._load_raw_window(row)
        else:
            if not self.features_dir:
                raise ValueError("features_dir is required when use_raw_frames is False")
            tensors["video"] = self._load_feature_window(row)

        if self.e2e_features_dir:
            tensors["e2e"] = self._load_e2e_frame_features(row)

        features: torch.Tensor | Dict[str, torch.Tensor]
        features = tensors["video"] if len(tensors) == 1 else tensors

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


def collate_windows(batch: List[Tuple[torch.Tensor | Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor | Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
    features, labels, metas = zip(*batch)
    if isinstance(features[0], dict):
        keys = features[0].keys()
        stacked = {key: torch.stack([item[key] for item in features], dim=0) for key in keys}  # type: ignore[index]
        return stacked, torch.stack(labels, dim=0), list(metas)
    return torch.stack(features, dim=0), torch.stack(labels, dim=0), list(metas)  # type: ignore[arg-type]
