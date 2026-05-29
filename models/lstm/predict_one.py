#!/usr/bin/env python3
"""Predict per-frame playing probabilities for a single MP4 clip.

Run from repo root:

    python models/lstm/predict_one.py \
      --clip-path data/test/NSsm4av7AF8/$CLIP.mp4 \
      --checkpoint models/lstm/checkpoints/2026-05-27-14:47-cnn-lstm-3sec-context/best.pt \
      --output-csv models/lstm/predictions/2026-05-27-14:47-cnn-lstm-3sec-context/"$CLIP"_predictions.csv \
      --device mps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import WINDOW_RADIUS, WINDOW_SIZE, build_window_indices  # noqa: E402
from models.lstm.encoders import get_encoder, resolve_device  # noqa: E402
from models.lstm.extract_features import encode_video_streaming  # noqa: E402
from models.lstm.train import pred_threshold_from_config, temporal_model_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--device", default=None, help="mps, cuda, cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    clip_path: Path = args.clip_path
    checkpoint: Path = args.checkpoint
    output_csv: Path = args.output_csv
    device: str | None = args.device
    batch_size = int(args.batch_size)

    if not clip_path.is_file():
        raise SystemExit(f"Missing clip: {clip_path}")
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")

    clip_id = clip_path.stem  # expected e.g. AWDaW5Sfu7I_001

    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    config = ckpt.get("config") or {}

    backbone = str(config.get("backbone", "efficientnet_v2_m"))
    feat_dim = int(config.get("feat_dim", 0))
    if feat_dim <= 0:
        raise SystemExit(f"Checkpoint missing feat_dim: {checkpoint}")
    pred_thr = pred_threshold_from_config(config)

    encoder = get_encoder(backbone, device=dev)
    if int(getattr(encoder, "feat_dim", feat_dim)) != feat_dim:
        # Not always present, but if it is, it should match.
        raise SystemExit(f"Encoder feat_dim mismatch: encoder={encoder.feat_dim} checkpoint={feat_dim}")

    print(f"clip={clip_path}")
    print(f"clip_id={clip_id}")
    print(f"checkpoint={checkpoint}")
    print(f"device={dev} backbone={backbone} feat_dim={feat_dim} batch_size={batch_size}")
    print(f"window_size={WINDOW_SIZE} (radius={WINDOW_RADIUS}) pred_threshold={pred_thr:.4f}")

    # ------------------------------------------------------------
    # 1) CNN feature extraction (directly from the MP4)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    features = encode_video_streaming(encoder, clip_path, batch_size=batch_size, show_frames=True)
    extract_s = time.perf_counter() - t0
    num_frames = int(features.shape[0])
    if int(features.shape[1]) != feat_dim:
        raise SystemExit(f"Feature dim mismatch: got {features.shape[1]} expected {feat_dim}")
    print(f"extracted_frames={num_frames} extract_fps={num_frames / extract_s:.1f}")

    # ------------------------------------------------------------
    # 2) Temporal model inference
    # ------------------------------------------------------------
    model = temporal_model_from_config(feat_dim, config).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    window_idx = build_window_indices(num_frames)
    frame_idx_out: list[int] = []
    prob_out: list[float] = []

    model_s = 0.0
    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            seq = torch.zeros((end - start, WINDOW_SIZE, feat_dim), dtype=torch.float32)
            for j, frame_idx in enumerate(range(start, end)):
                idxs = window_idx[frame_idx]
                for i, src_idx in enumerate(idxs):
                    if 0 <= src_idx < num_frames:
                        seq[j, i] = features[int(src_idx)]

            t1 = time.perf_counter()
            logits = model(seq.to(dev))
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            model_s += time.perf_counter() - t1

            frame_idx_out.extend(range(start, end))
            prob_out.extend(probs.tolist())

    prob = np.asarray(prob_out, dtype=np.float64)
    pred = (prob >= float(pred_thr)).astype(np.int64)
    print(f"infer_frames={num_frames} model_fps={num_frames / model_s:.1f}")

    # ------------------------------------------------------------
    # 4) Write CSV
    # ------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "clip_key": clip_id,
            "frame_idx": np.asarray(frame_idx_out, dtype=np.int64),
            "prob_playing": prob,
            "pred_playing": pred,
        }
    )

    df.to_csv(output_csv, index=False, lineterminator="\n")
    print(f"wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

