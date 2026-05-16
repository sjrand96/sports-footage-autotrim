#!/usr/bin/env python3
"""Export per-frame LSTM predictions for one clip to the repo root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import FeatureWindowDataset, load_labels_by_clip  # noqa: E402
from models.lstm.encoders import resolve_device  # noqa: E402
from models.lstm.train import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    FRAME_LABELS_CSV,
    NUM_WORKERS,
    collate_batch,
    pred_threshold_from_config,
    temporal_model_from_config,
)


def export_clip_predictions(
    clip_id: str,
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = REPO_ROOT,
    device: str | None = None,
    batch_size: int = 64,
) -> tuple[Path, Path]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    config = ckpt.get("config") or {}
    feat_dim = int(config.get("feat_dim", 0))
    if feat_dim <= 0:
        raise RuntimeError(f"checkpoint missing feat_dim: {checkpoint}")

    backbone = str(config.get("backbone", "efficientnet_v2_m"))
    pred_thr = pred_threshold_from_config(config)

    labels_by_clip = load_labels_by_clip(FRAME_LABELS_CSV)
    ds = FeatureWindowDataset([clip_id], backbone=backbone, labels_by_clip=labels_by_clip)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )

    model = temporal_model_from_config(feat_dim, config).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frame_indices: list[int] = []
    y_true: list[int] = []
    y_prob: list[float] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["seq"].to(dev))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            frame_indices.extend(batch["frame_idx"].tolist())
            y_true.extend(batch["label"].cpu().numpy().ravel().astype(int).tolist())
            y_prob.extend(probs.tolist())

    order = np.argsort(frame_indices)
    frame_idx = np.asarray(frame_indices, dtype=np.int64)[order]
    label = np.asarray(y_true, dtype=np.int64)[order]
    prob = np.asarray(y_prob, dtype=np.float64)[order]
    pred = (prob >= pred_thr).astype(np.int64)

    df = pd.DataFrame(
        {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "prob_playing": prob,
            "pred_playing": pred.astype(bool),
            "label_playing": label.astype(bool),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{clip_id}_predictions.csv"
    out_json = output_dir / f"{clip_id}_predictions.json"
    df.to_csv(out_csv, index=False)

    summary = {
        "clip_id": clip_id,
        "checkpoint": str(checkpoint.resolve()),
        "pred_threshold": pred_thr,
        "backbone": backbone,
        "n_frames": int(len(df)),
        "n_pred_playing": int(pred.sum()),
        "n_label_playing": int(label.sum()),
    }
    out_json.write_text(
        json.dumps({"summary": summary, "frames": df.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    return out_csv, out_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-id", required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory for output files (default: repo root)",
    )
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_csv, out_json = export_clip_predictions(
        args.clip_id,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
