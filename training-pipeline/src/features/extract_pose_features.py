#!/usr/bin/env python3
"""Extract pose features for each clip using MediaPipe."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable

import numpy as np


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


def _load_frames(clip_path: str, fps: float) -> np.ndarray:
    try:
        from decord import VideoReader  # type: ignore
    except ImportError as exc:
        raise ImportError("decord is required for pose extraction") from exc

    vr = VideoReader(clip_path)
    native_fps = vr.get_avg_fps()
    stride = max(1, int(round(native_fps / fps)))
    indices = np.arange(0, len(vr), stride)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def _extract_pose(frames: np.ndarray) -> np.ndarray:
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError("mediapipe is required for pose extraction") from exc

    mp_pose = mp.solutions.pose
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for row in _load_jsonl(args.input):
        clip_id = _safe_clip_id(row)
        out_path = os.path.join(args.output_dir, f"{clip_id}.npz")
        if os.path.exists(out_path):
            continue

        frames = _load_frames(row["clip_path"], args.fps)
        timestamps_sec = np.arange(frames.shape[0]) / args.fps
        pose = _extract_pose(frames)
        np.savez_compressed(out_path, timestamps_sec=timestamps_sec, pose=pose)
        print(f"Saved pose: {out_path}")


if __name__ == "__main__":
    main()
