#!/usr/bin/env python3
"""Extract frozen visual embeddings for each clip."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms


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
        raise ImportError("decord is required for embedding extraction") from exc

    vr = VideoReader(clip_path)
    native_fps = vr.get_avg_fps()
    stride = max(1, int(round(native_fps / fps)))
    indices = np.arange(0, len(vr), stride)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def _build_model(device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract visual embeddings for each clip.")
    parser.add_argument("--input", required=True, help="Canonical clip manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for .npz embeddings.")
    parser.add_argument("--fps", type=float, default=6.0, help="Sampling fps for embeddings.")
    parser.add_argument("--batch-size", type=int, default=32, help="Frame batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = _build_model(device)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for row in _load_jsonl(args.input):
        clip_id = _safe_clip_id(row)
        out_path = os.path.join(args.output_dir, f"{clip_id}.npz")
        if os.path.exists(out_path):
            continue

        frames = _load_frames(row["clip_path"], args.fps)
        timestamps_sec = np.arange(frames.shape[0]) / args.fps

        features: List[np.ndarray] = []
        for i in range(0, frames.shape[0], args.batch_size):
            batch = frames[i : i + args.batch_size]
            batch_tensors = torch.stack([preprocess(frame) for frame in batch]).to(device)
            with torch.no_grad():
                emb = model(batch_tensors).cpu().numpy()
            features.append(emb.astype(np.float32))

        features_np = np.concatenate(features, axis=0) if features else np.zeros((0, 512), dtype=np.float32)
        np.savez_compressed(out_path, timestamps_sec=timestamps_sec, features=features_np)
        print(f"Saved embeddings: {out_path}")


if __name__ == "__main__":
    main()
