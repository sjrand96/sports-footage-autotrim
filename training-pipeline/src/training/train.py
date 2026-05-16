#!/usr/bin/env python3
"""Train transformer classifier for playtime vs downtime windows."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.video_window_dataset import VideoWindowDataset, collate_windows
from src.models.transformer_classifier import (
    FusionTransformerConfig,
    LateFusionTransformerClassifier,
    TransformerClassifier,
    TransformerConfig,
)
from src.training.wandb_logger import WandbConfig, WandbLogger


@dataclass
class TrainConfig:
    manifest_path: str
    features_dir: str | None
    pose_dir: str | None
    e2e_features_dir: str | None
    output_dir: str
    split_field: str = "source_id"
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 13
    batch_size: int = 32
    num_frames: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    use_raw_frames: bool = False
    fusion: str = "none"
    e2e_feature_subset: str = "all"
    s3_cache_dir: str | None = None
    clip_cache_dir: str | None = None
    clip_duration_sec: float = 60.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _group_split(rows: List[Dict[str, Any]], field: str, ratios: Tuple[float, float, float], seed: int) -> Tuple[List[int], List[int], List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        key = row.get(field) or row.get("match_id") or row.get("source_id") or "unknown"
        groups.setdefault(str(key), []).append(idx)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train : n_train + n_val])

    train_idx, val_idx, test_idx = [], [], []
    for key, indices in groups.items():
        if key in train_keys:
            train_idx.extend(indices)
        elif key in val_keys:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)
    return train_idx, val_idx, test_idx


def _compute_class_weights(labels: List[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    return torch.tensor(weights, dtype=torch.float32)


def _normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 1, 3, 1, 1)
    return (frames - mean) / std


def _encode_frames(
    frames: torch.Tensor,
    encoder: nn.Module,
) -> torch.Tensor:
    batch, timesteps, channels, height, width = frames.shape
    frames = _normalize_frames(frames)
    flat = frames.view(batch * timesteps, channels, height, width)
    with torch.no_grad():
        feats = encoder(flat)
    return feats.view(batch, timesteps, -1)


def _move_features_to_device(features: torch.Tensor | Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor | Dict[str, torch.Tensor]:
    if isinstance(features, dict):
        return {key: value.to(device) for key, value in features.items()}
    return features.to(device)


def _prepare_model_inputs(
    features: torch.Tensor | Dict[str, torch.Tensor],
    fusion: str,
    frame_encoder: nn.Module | None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    if isinstance(features, dict):
        video = features["video"]
        if frame_encoder is not None:
            video = _encode_frames(video, frame_encoder)
        if fusion == "early":
            return torch.cat([video, features["e2e"]], dim=-1)
        if fusion == "late":
            return {"video": video, "e2e": features["e2e"]}
        return video
    if frame_encoder is not None:
        return _encode_frames(features, frame_encoder)
    return features


def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    frame_encoder: nn.Module | None = None,
) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            feats, labels, _ = batch
            feats = _move_features_to_device(feats, device)
            model_inputs = _prepare_model_inputs(feats, getattr(model, "fusion_mode", "none"), frame_encoder)
            logits = model(model_inputs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    return y_true, y_pred


def _precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer classifier.")
    parser.add_argument("--config", default=None, help="Optional JSON config path.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--e2e-features-dir", default=None, help="Local dir or s3:// prefix containing *_features.parquet files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-raw-frames", action="store_true", help="Train on raw frames with a frozen ResNet encoder.")
    parser.add_argument("--fusion", choices=["none", "early", "late"], default=None, help="How to fuse E2E parquet features with video features.")
    parser.add_argument("--e2e-feature-subset", choices=["all", "base"], default=None)
    parser.add_argument("--s3-cache-dir", default=None, help="Local cache for S3 manifests/features.")
    parser.add_argument("--clip-cache-dir", default=None, help="Local cache for S3 video clips.")
    parser.add_argument("--wandb-project", default="volleyball-playtime")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        pose_dir=args.pose_dir,
        e2e_features_dir=args.e2e_features_dir,
        output_dir=args.output_dir,
        use_raw_frames=args.use_raw_frames,
        fusion=args.fusion or "none",
        e2e_feature_subset=args.e2e_feature_subset or "all",
        s3_cache_dir=args.s3_cache_dir,
        clip_cache_dir=args.clip_cache_dir,
    )
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        merged = {**cfg.__dict__, **payload}
        merged.update(
            {
                "manifest_path": args.manifest,
                "features_dir": args.features_dir if args.features_dir is not None else merged.get("features_dir"),
                "pose_dir": args.pose_dir if args.pose_dir is not None else merged.get("pose_dir"),
                "e2e_features_dir": args.e2e_features_dir if args.e2e_features_dir is not None else merged.get("e2e_features_dir"),
                "output_dir": args.output_dir,
                "s3_cache_dir": args.s3_cache_dir if args.s3_cache_dir is not None else merged.get("s3_cache_dir"),
                "clip_cache_dir": args.clip_cache_dir if args.clip_cache_dir is not None else merged.get("clip_cache_dir"),
            }
        )
        if args.use_raw_frames:
            merged["use_raw_frames"] = True
        if args.fusion is not None:
            merged["fusion"] = args.fusion
        if args.e2e_feature_subset is not None:
            merged["e2e_feature_subset"] = args.e2e_feature_subset
        cfg = TrainConfig(**merged)

    _set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if not cfg.use_raw_frames and not cfg.features_dir:
        raise ValueError("--features-dir is required unless --use-raw-frames is set")
    if cfg.fusion in {"early", "late"} and not cfg.e2e_features_dir:
        raise ValueError("--e2e-features-dir is required for early or late fusion")

    dataset = VideoWindowDataset(
        cfg.manifest_path,
        features_dir=cfg.features_dir,
        pose_dir=cfg.pose_dir,
        e2e_features_dir=cfg.e2e_features_dir,
        e2e_feature_subset=cfg.e2e_feature_subset,
        s3_cache_dir=cfg.s3_cache_dir,
        clip_cache_dir=cfg.clip_cache_dir,
        clip_duration_sec=cfg.clip_duration_sec,
        num_frames=cfg.num_frames,
        use_raw_frames=cfg.use_raw_frames,
    )

    train_idx, val_idx, _ = _group_split(dataset.rows, cfg.split_field, cfg.split_ratios, cfg.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
    )

    sample_batch = next(iter(train_loader))
    sample_features = sample_batch[0]
    e2e_dim = None
    if cfg.use_raw_frames:
        video_input_dim = 512
    else:
        video_tensor = sample_features["video"] if isinstance(sample_features, dict) else sample_features
        video_input_dim = video_tensor.shape[-1]

    if isinstance(sample_features, dict):
        e2e_dim = int(sample_features["e2e"].shape[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.fusion == "late":
        if e2e_dim is None:
            raise ValueError("Late fusion requires E2E parquet features.")
        model_cfg = FusionTransformerConfig(video_input_dim=video_input_dim, feature_input_dim=e2e_dim, max_len=cfg.num_frames)
        model = LateFusionTransformerClassifier(model_cfg).to(device)
    else:
        input_dim = video_input_dim + (e2e_dim or 0) if cfg.fusion == "early" else video_input_dim
        model_cfg = TransformerConfig(input_dim=input_dim, max_len=cfg.num_frames)
        model = TransformerClassifier(model_cfg).to(device)
    model.fusion_mode = cfg.fusion  # type: ignore[attr-defined]

    labels = [dataset.rows[i]["label"] for i in train_idx]
    class_weights = _compute_class_weights([int(x) for x in labels]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    wandb_logger = WandbLogger(
        WandbConfig(project=args.wandb_project, entity=args.wandb_entity, run_name=args.wandb_run, enabled=True),
        config={**cfg.__dict__, "video_input_dim": video_input_dim, "e2e_input_dim": e2e_dim},
    )

    best_f1 = -1.0
    best_path = os.path.join(cfg.output_dir, "best.pt")

    frame_encoder = None
    if cfg.use_raw_frames:
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        encoder.fc = nn.Identity()
        encoder.eval()
        encoder.to(device)
        frame_encoder = encoder

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for feats, labels, _ in train_loader:
            feats = _move_features_to_device(feats, device)
            labels = labels.to(device)
            model_inputs = _prepare_model_inputs(feats, cfg.fusion, frame_encoder)
            optimizer.zero_grad()
            logits = model(model_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / max(1, len(train_loader.dataset))
        y_true, y_pred = _eval_loader(model, val_loader, device, frame_encoder=frame_encoder)
        precision, recall, f1 = _precision_recall_f1(y_true, y_pred)

        wandb_logger.log(
            {
                "train_loss": avg_loss,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "epoch": epoch,
            },
            step=epoch,
        )
        wandb_logger.log_confusion_matrix(y_true, y_pred, labels=["downtime", "playtime"])

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "config": model_cfg.__dict__, "fusion": cfg.fusion}, best_path)

    wandb_logger.finish()
    print(f"Best model saved to {best_path} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
