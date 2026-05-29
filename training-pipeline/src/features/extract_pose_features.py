#!/usr/bin/env python3
"""Extract pose features for each clip using MediaPipe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from src.data.s3_cache import download_s3_uri, is_s3_uri


def _load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_clip_id(row: Dict[str, Any]) -> str:
    clip_id = row.get("clip_id")
    if clip_id:
        return str(clip_id)
    base = os.path.basename(row["clip_path"])
    return os.path.splitext(base)[0]


def _load_frames(clip_path: str, fps: float, *, s3_cache_dir: str) -> np.ndarray:
    if is_s3_uri(clip_path):
        clip_path = download_s3_uri(clip_path, s3_cache_dir)
    try:
        from decord import VideoReader  # type: ignore

        vr = VideoReader(clip_path)
        native_fps = vr.get_avg_fps()
        stride = max(1, int(round(native_fps / fps)))
        indices = np.arange(0, len(vr), stride)
        return vr.get_batch(indices).asnumpy()
    except Exception:
        return _load_frames_opencv(clip_path, fps)


def _load_frames_opencv(clip_path: str, fps: float) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python is required when decord cannot read a clip") from exc

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video clip: {clip_path}")
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(native_fps / fps)))
        frames: list[np.ndarray] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(frame[:, :, ::-1])
            idx += 1
        return np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
    finally:
        cap.release()


def _extract_pose(frames: np.ndarray) -> np.ndarray:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError("mediapipe is required for pose extraction") from exc

    mp_solutions = getattr(mp, "solutions", None)
    if mp_solutions is None:
        try:
            import mediapipe.python.solutions as mp_solutions  # type: ignore
        except Exception as exc:
            raise ImportError("mediapipe solutions module is unavailable") from exc

    mp_pose = mp_solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    outputs = []
    for frame in frames:
        rgb = frame[:, :, ::-1]
        result = pose.process(rgb)
        if not result.pose_landmarks:
            outputs.append(np.zeros((33, 4), dtype=np.float32))
            continue
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z, lm.visibility])
        outputs.append(np.array(keypoints, dtype=np.float32))

    pose.close()
    return np.stack(outputs, axis=0) if outputs else np.zeros((0, 33, 4), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose features for each clip.")
    parser.add_argument("--input", required=True, help="Canonical clip manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for pose feature .npz files.")
    parser.add_argument("--fps", type=float, default=6.0, help="Sampling fps for pose.")
    parser.add_argument("--s3-cache-dir", default="data/s3_cache", help="Local cache for S3 clips.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for row in _load_jsonl(args.input):
        clip_id = _safe_clip_id(row)
        out_path = os.path.join(args.output_dir, f"{clip_id}.npz")
        if os.path.exists(out_path):
            continue

        frames = _load_frames(row["clip_path"], args.fps, s3_cache_dir=args.s3_cache_dir)
        timestamps_sec = np.arange(frames.shape[0]) / args.fps
        pose = _extract_pose(frames)
        np.savez_compressed(out_path, timestamps_sec=timestamps_sec, pose=pose)
        print(f"Saved pose: {out_path}")


if __name__ == "__main__":
    main()
