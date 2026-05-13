#!/usr/bin/env python3
"""Evaluate window and segment-level performance."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.video_window_dataset import VideoWindowDataset, collate_windows
from src.models.transformer_classifier import TransformerClassifier, TransformerConfig
from src.training.wandb_logger import WandbConfig, WandbLogger


@dataclass
class EvalConfig:
    manifest_path: str
    features_dir: str
    pose_dir: str | None
    checkpoint_path: str
    batch_size: int = 32
    num_frames: int = 16
    num_workers: int = 4


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def _window_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def _merge_windows(windows: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda x: x[0])
    merged = [windows[0]]
    for start, end in windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _segment_metrics(y_true: List[int], y_pred: List[int], meta: List[Dict[str, Any]], gap: float, iou: float) -> Dict[str, float]:
    by_clip: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta):
        clip = row.get("clip_id") or row.get("clip_path")
        by_clip.setdefault(str(clip), []).append(idx)

    tp = fp = fn = 0
    for clip, indices in by_clip.items():
        gt_windows = [(meta[i]["window_start_sec"], meta[i]["window_end_sec"]) for i in indices if y_true[i] == 1]
        pred_windows = [(meta[i]["window_start_sec"], meta[i]["window_end_sec"]) for i in indices if y_pred[i] == 1]
        gt_segments = _merge_windows(gt_windows, gap)
        pred_segments = _merge_windows(pred_windows, gap)

        matched_gt = set()
        for pred in pred_segments:
            best_iou = 0.0
            best_idx = None
            for idx, gt in enumerate(gt_segments):
                if idx in matched_gt:
                    continue
                inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
                union = max(pred[1], gt[1]) - min(pred[0], gt[0])
                score = inter / union if union > 0 else 0.0
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_iou >= iou and best_idx is not None:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn += max(0, len(gt_segments) - len(matched_gt))

    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def _try_make_gif(
    clip_path: str,
    start: float,
    end: float,
    out_path: str,
    fps: float = 6.0,
    max_frames: int = 24,
) -> bool:
    try:
        from decord import VideoReader  # type: ignore
        import imageio.v2 as imageio  # type: ignore
    except ImportError:
        return False

    vr = VideoReader(clip_path)
    native_fps = vr.get_avg_fps()
    start_idx = int(start * native_fps)
    end_idx = max(start_idx + 1, int(end * native_fps))
    indices = np.linspace(start_idx, end_idx - 1, num=min(max_frames, end_idx - start_idx))
    indices = np.clip(indices.astype(int), 0, len(vr) - 1)
    frames = vr.get_batch(indices).asnumpy()
    imageio.mimsave(out_path, frames, fps=fps)
    return True


def _try_plot_timeline(
    rows: List[Dict[str, Any]],
    probs: List[float],
    out_path: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False

    starts = [row["window_start_sec"] for row in rows]
    ends = [row["window_end_sec"] for row in rows]
    labels = [row["label"] for row in rows]
    centers = [(s + e) / 2 for s, e in zip(starts, ends)]

    plt.figure(figsize=(10, 3))
    plt.step(centers, probs, where="mid", label="pred_prob")
    plt.fill_between(centers, [0] * len(labels), labels, step="mid", alpha=0.2, label="gt")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("time (sec)")
    plt.ylabel("playtime")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _load_checkpoint(path: str, input_dim: int, max_len: int) -> TransformerClassifier:
    data = torch.load(path, map_location="cpu")
    cfg = TransformerConfig(input_dim=input_dim, max_len=max_len)
    if "config" in data:
        cfg = TransformerConfig(**{**cfg.__dict__, **data["config"]})
    model = TransformerClassifier(cfg)
    model.load_state_dict(data["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on window and segment metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gap-sec", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--media-samples", type=int, default=3)
    parser.add_argument("--wandb-project", default="volleyball-playtime")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    cfg = EvalConfig(
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        pose_dir=args.pose_dir,
        checkpoint_path=args.checkpoint,
    )

    dataset = VideoWindowDataset(
        cfg.manifest_path,
        features_dir=cfg.features_dir,
        pose_dir=cfg.pose_dir,
        num_frames=cfg.num_frames,
        use_raw_frames=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
    )

    sample_batch = next(iter(loader))
    input_dim = sample_batch[0].shape[-1]
    model = _load_checkpoint(cfg.checkpoint_path, input_dim, cfg.num_frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        for feats, labels, batch_meta in loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
            y_prob.extend(probs)
            meta.extend(batch_meta)

    window_metrics = _window_metrics(y_true, y_pred)
    segment_metrics = _segment_metrics(y_true, y_pred, meta, args.gap_sec, args.iou)
    results = {
        "window": window_metrics,
        "segment": segment_metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    wandb_logger = WandbLogger(
        WandbConfig(project=args.wandb_project, run_name=args.wandb_run, enabled=True),
        config=cfg.__dict__,
    )
    wandb_logger.log({"window_precision": window_metrics["precision"], "window_recall": window_metrics["recall"], "window_f1": window_metrics["f1"]})
    wandb_logger.log({"segment_precision": segment_metrics["precision"], "segment_recall": segment_metrics["recall"], "segment_f1": segment_metrics["f1"]})
    wandb_logger.log_confusion_matrix(y_true, y_pred, labels=["downtime", "playtime"])

    sample_rows = []
    for idx, row in enumerate(meta):
        sample_rows.append(
            [
                row.get("clip_id"),
                row.get("window_start_sec"),
                row.get("window_end_sec"),
                int(y_true[idx]),
                int(y_pred[idx]),
                float(y_prob[idx]),
                int(y_true[idx] == y_pred[idx]),
            ]
        )
    random.shuffle(sample_rows)
    wandb_logger.log_table(
        "validation_windows",
        ["clip_id", "start_sec", "end_sec", "label", "pred", "prob_play", "correct"],
        sample_rows[:200],
    )

    wandb_module = wandb_logger.get()
    media_dir = os.path.join(os.path.dirname(args.output), "media")
    os.makedirs(media_dir, exist_ok=True)
    if wandb_module:
        correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
        incorrect_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        random.shuffle(correct_indices)
        random.shuffle(incorrect_indices)

        for tag, indices in [("correct", correct_indices), ("incorrect", incorrect_indices)]:
            for idx in indices[: args.media_samples]:
                row = meta[idx]
                clip_id = row.get("clip_id") or f"clip_{idx}"
                gif_path = os.path.join(media_dir, f"{clip_id}_{tag}_{idx}.gif")
                if _try_make_gif(
                    row["clip_path"],
                    float(row["window_start_sec"]),
                    float(row["window_end_sec"]),
                    gif_path,
                ):
                    wandb_logger.log_media(
                        f"window_{tag}",
                        wandb_module.Image(gif_path, caption=f"{clip_id} {tag}"),
                    )

        by_clip: Dict[str, List[int]] = {}
        for idx, row in enumerate(meta):
            clip_key = row.get("clip_id") or row.get("clip_path")
            by_clip.setdefault(str(clip_key), []).append(idx)

        for clip_key in list(by_clip.keys())[: args.media_samples]:
            indices = by_clip[clip_key]
            rows = [meta[i] for i in indices]
            probs = [y_prob[i] for i in indices]
            plot_path = os.path.join(media_dir, f"timeline_{clip_key}.png")
            if _try_plot_timeline(rows, probs, plot_path):
                wandb_logger.log_media(
                    "timeline_debug",
                    wandb_module.Image(plot_path, caption=f"{clip_key} timeline"),
                )
    wandb_logger.finish()

    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
