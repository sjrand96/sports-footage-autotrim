"""Dataset for frame-centered raw-video windows using parquet frame labels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.s3_cache import download_s3_uri, is_s3_uri, parse_s3_uri

try:
    from models.lstm.dataset import loss_mask_for_labels
except ModuleNotFoundError:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from models.lstm.dataset import loss_mask_for_labels


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    clip_s3_uri: str
    source_id: str | None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_local_parquets(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"parquet path not found: {path}")
    return sorted(str(p) for p in Path(path).glob("*.parquet"))


def _list_s3_parquets(prefix_uri: str) -> List[str]:
    parsed = parse_s3_uri(prefix_uri)
    import boto3

    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=parsed.bucket, Prefix=parsed.key):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key.endswith(".parquet"):
                keys.append(key)
    return [f"s3://{parsed.bucket}/{key}" for key in sorted(keys)]


def _resolve_parquet_sources(sources: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    for source in sources:
        if is_s3_uri(source):
            resolved.extend(_list_s3_parquets(source))
        else:
            resolved.extend(_list_local_parquets(source))
    if not resolved:
        raise RuntimeError("no parquet files found")
    return resolved


def _load_parquet(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and a parquet engine are required") from exc
    return pd.read_parquet(path)


def _read_parquets(paths: Sequence[str], cache_dir: str) -> "pd.DataFrame":
    frames: List["pd.DataFrame"] = []
    for path in paths:
        local_path = download_s3_uri(path, cache_dir) if is_s3_uri(path) else path
        frames.append(_load_parquet(local_path))
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required") from exc
    return pd.concat(frames, ignore_index=True)


def _build_window_offsets(window_radius: int) -> np.ndarray:
    if window_radius <= 0:
        raise ValueError("window_radius must be > 0")
    return np.arange(-window_radius, window_radius, dtype=np.int64)


def _sample_window_indices(window_indices: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_frames == window_indices.size:
        return window_indices
    picks = np.linspace(0, window_indices.size - 1, num=num_frames)
    return window_indices[picks.round().astype(int)]


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    if frames.size == 0:
        return frames
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    try:
        import cv2  # type: ignore

        resized = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) for frame in frames]
        return np.stack(resized, axis=0)
    except ImportError:
        from PIL import Image

        resized = [np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR)) for frame in frames]
        return np.stack(resized, axis=0)


def _load_frames_by_indices(clip_path: str, indices: Sequence[int], image_size: int) -> np.ndarray:
    try:
        from decord import VideoReader  # type: ignore

        vr = VideoReader(clip_path)
        n = len(vr)
        frames = np.zeros((len(indices), image_size, image_size, 3), dtype=np.uint8)
        valid_positions: List[int] = []
        valid_indices: List[int] = []
        for pos, idx in enumerate(indices):
            if 0 <= idx < n:
                valid_positions.append(pos)
                valid_indices.append(int(idx))
        if valid_indices:
            batch = vr.get_batch(valid_indices).asnumpy()
            batch = _resize_frames(batch, image_size)
            for pos, frame in zip(valid_positions, batch):
                frames[pos] = frame
        return frames
    except ImportError:
        return _load_frames_by_indices_opencv(clip_path, indices, image_size)


def _load_frames_by_indices_opencv(clip_path: str, indices: Sequence[int], image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Either decord or opencv-python is required for raw-frame loading") from exc

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video clip: {clip_path}")
    try:
        frames = np.zeros((len(indices), image_size, image_size, 3), dtype=np.uint8)
        for pos, idx in enumerate(indices):
            if idx < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[pos] = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        return frames
    finally:
        cap.release()


class RawFrameWindowDataset(Dataset):
    def __init__(
        self,
        parquet_sources: Sequence[str],
        *,
        s3_cache_dir: str,
        clip_cache_dir: str,
        window_radius: int = 15,
        num_frames: int = 16,
        image_size: int = 224,
        boundary_margin: int = 0,
        frame_stride: int = 1,
    ) -> None:
        self.s3_cache_dir = s3_cache_dir
        self.clip_cache_dir = clip_cache_dir
        _ensure_dir(self.s3_cache_dir)
        _ensure_dir(self.clip_cache_dir)
        self.window_radius = int(window_radius)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.boundary_margin = int(boundary_margin)
        self.frame_stride = int(frame_stride)
        if self.window_radius <= 0:
            raise ValueError("window_radius must be > 0")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        parquet_paths = _resolve_parquet_sources(parquet_sources)
        df = _read_parquets(parquet_paths, self.s3_cache_dir)
        required = {"clip_id", "clip_s3_uri", "frame_idx", "is_playing"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing parquet columns: {sorted(missing)}")

        df = df.sort_values(["clip_id", "frame_idx"])
        self.clip_meta: Dict[str, ClipMeta] = {}
        self.labels_by_clip: Dict[str, np.ndarray] = {}
        self._loss_mask: Dict[str, np.ndarray] = {}
        self.samples: List[Tuple[str, int]] = []
        self._window_offsets = _build_window_offsets(self.window_radius)

        for clip_id, group in df.groupby("clip_id", sort=False):
            clip_id_str = str(clip_id)
            frame_idx = group["frame_idx"].to_numpy(dtype=int)
            expected = np.arange(frame_idx.size)
            if not np.array_equal(frame_idx, expected):
                raise ValueError(f"non-contiguous frame_idx for clip_id={clip_id_str}")
            labels = group["is_playing"].to_numpy(dtype=np.uint8)
            self.labels_by_clip[clip_id_str] = labels
            clip_uri = str(group["clip_s3_uri"].dropna().iloc[0])
            source_id = None
            if "source_id" in group.columns:
                source_id = str(group["source_id"].dropna().iloc[0]) if group["source_id"].notna().any() else None
            self.clip_meta[clip_id_str] = ClipMeta(clip_id=clip_id_str, clip_s3_uri=clip_uri, source_id=source_id)
            self._loss_mask[clip_id_str] = loss_mask_for_labels(labels, self.boundary_margin)
            for idx in range(0, labels.size, self.frame_stride):
                self.samples.append((clip_id_str, idx))

        if not self.samples:
            raise ValueError("empty dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def clip_ids(self) -> List[str]:
        return sorted(self.clip_meta.keys())

    def _clip_path(self, clip_id: str) -> str:
        clip_uri = self.clip_meta[clip_id].clip_s3_uri
        if is_s3_uri(clip_uri):
            return download_s3_uri(clip_uri, self.clip_cache_dir)
        return clip_uri

    def __getitem__(self, index: int) -> Dict[str, Any]:
        clip_id, frame_idx = self.samples[index]
        labels = self.labels_by_clip[clip_id]
        window_indices = frame_idx + self._window_offsets
        window_indices = _sample_window_indices(window_indices, self.num_frames)

        clip_path = self._clip_path(clip_id)
        frames = _load_frames_by_indices(clip_path, window_indices.tolist(), self.image_size)
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))

        loss_mask = bool(self._loss_mask[clip_id][frame_idx])
        return {
            "video": torch.from_numpy(frames),
            "label": torch.tensor(float(labels[frame_idx]), dtype=torch.float32),
            "loss_mask": torch.tensor(1.0 if loss_mask else 0.0, dtype=torch.float32),
            "clip_id": clip_id,
            "frame_idx": frame_idx,
        }


def collate_raw_frame_windows(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    batch_list = list(batch)
    return {
        "video": torch.stack([item["video"] for item in batch_list], dim=0),
        "label": torch.stack([item["label"] for item in batch_list], dim=0),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch_list], dim=0),
        "clip_id": [item["clip_id"] for item in batch_list],
        "frame_idx": torch.tensor([item["frame_idx"] for item in batch_list], dtype=torch.long),
    }
