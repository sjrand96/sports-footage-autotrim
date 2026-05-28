#!/usr/bin/env python3
"""Evaluate window, segment, and optional frame-level performance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import random

import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.s3_cache import download_s3_uri, is_s3_uri
from src.data.video_window_dataset import VideoWindowDataset, collate_windows
from src.models.transformer_classifier import (
    FusionTransformerConfig,
    LateFusionTransformerClassifier,
    TransformerClassifier,
    TransformerConfig,
)
from src.training.wandb_logger import WandbConfig, WandbLogger


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


@dataclass
class EvalConfig:
    manifest_path: str
    features_dir: str | None
    pose_dir: str | None
    e2e_features_dir: str | None
    e2e_feature_subset: str
    checkpoint_path: str
    batch_size: int = 32
    num_frames: int = 16
    num_workers: int = 4
    use_raw_frames: bool = False
    fusion: str = "none"
    s3_cache_dir: str | None = None
    clip_cache_dir: str | None = None
    e2e_only: bool = False


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def _f_beta(precision: float, recall: float, beta: float) -> float:
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)


def _binary_metrics(
    y_true: List[int],
    y_pred: List[int],
    *,
    pos_weight: float,
    beta: float,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    n = tp + fp + tn + fn
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_beta": _f_beta(precision, recall, beta),
        "accuracy": (tp + tn) / max(n, 1),
        "cost": pos_weight * fn + fp,
        "n_frames": float(n),
    }


def _format_metrics_row(clip_id: str, m: Dict[str, float]) -> str:
    return (
        f"{clip_id:20s}  "
        f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
        f"f1={m['f1']:.3f}  fbeta={m['f_beta']:.3f}  "
        f"acc={m['accuracy']:.3f}  cost={m['cost']:.0f}  "
        f"TP={int(m['tp']):4d} FP={int(m['fp']):4d} TN={int(m['tn']):4d} FN={int(m['fn']):4d}  "
        f"n={int(m['n_frames'])}"
    )


def _table_metrics_row(model_name: str, m: Dict[str, float]) -> Dict[str, float | str]:
    return {
        "model": model_name,
        "tp": int(m["tp"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
        "tn": int(m["tn"]),
        "precision": m["precision"],
        "recall": m["recall"],
        "f2_score": m["f_beta"],
        "n_frames": int(m["n_frames"]),
    }


def _window_metrics(y_true: List[int], y_pred: List[int], *, beta: float) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1, "f_beta": _f_beta(precision, recall, beta)}


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


def _segment_metrics(
    y_true: List[int],
    y_pred: List[int],
    meta: List[Dict[str, Any]],
    gap: float,
    iou: float,
    *,
    beta: float,
) -> Dict[str, float]:
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
    return {"precision": precision, "recall": recall, "f1": f1, "f_beta": _f_beta(precision, recall, beta)}


def _try_make_gif(
    clip_path: str,
    start: float,
    end: float,
    out_path: str,
    fps: float = 6.0,
    max_frames: int = 24,
    cache_dir: str | None = None,
) -> bool:
    try:
        from decord import VideoReader  # type: ignore
        import imageio.v2 as imageio  # type: ignore
    except ImportError:
        return False

    if is_s3_uri(clip_path):
        if not cache_dir:
            return False
        clip_path = download_s3_uri(clip_path, cache_dir)

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


def _spans_from_binary(values: List[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(values) - 1):
            end = idx + 1 if value else idx
            spans.append((start, end - start))
            start = None
    return spans


def _plot_source_timelines(
    meta: List[Dict[str, Any]],
    y_true: List[int],
    y_pred: List[int],
    out_path: str,
    *,
    group_field: str,
    max_groups: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return False

    grouped: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta):
        key = row.get(group_field) or row.get("clip_id") or row.get("clip_path") or "unknown"
        grouped.setdefault(str(key), []).append(idx)

    items = sorted(grouped.items(), key=lambda kv: kv[0])[:max_groups]
    if not items:
        return False

    fig_h = max(2.0, 1.3 * len(items))
    fig, axes = plt.subplots(len(items), 1, figsize=(14, fig_h), sharex=False)
    if len(items) == 1:
        axes = [axes]

    for ax, (group, indices) in zip(axes, items):
        ordered = sorted(indices, key=lambda i: (str(meta[i].get("clip_id", "")), meta[i].get("window_start_sec", 0)))
        labels = [int(y_true[i]) for i in ordered]
        preds = [int(y_pred[i]) for i in ordered]

        gt_spans = _spans_from_binary(labels)
        pred_spans = _spans_from_binary(preds)

        ax.broken_barh(gt_spans, (1.1, 0.8), facecolors="tab:green")
        ax.broken_barh(pred_spans, (0.0, 0.8), facecolors="tab:red")
        ax.set_ylim(-0.2, 2.1)
        ax.set_yticks([0.4, 1.5])
        ax.set_yticklabels(["prediction", "ground truth"])
        ax.set_xlim(0, max(len(labels), 1))
        ax.set_title(str(group), fontsize=10, loc="left")

    axes[-1].set_xlabel("window index (concatenated across clips)")
    fig.suptitle(f"Test set — one row per {group_field}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _resolve_frame_parquet(prefix: str, clip_id: str, cache_dir: str) -> str:
    candidates = [f"{clip_id}.parquet", f"{clip_id}_predictions.parquet", f"{clip_id}_features.parquet"]
    if is_s3_uri(prefix):
        for name in candidates:
            uri = f"{prefix.rstrip('/')}/{name}"
            try:
                return download_s3_uri(uri, cache_dir)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                try:
                    from botocore.exceptions import ClientError  # type: ignore
                except ImportError:
                    raise
                if not isinstance(exc, ClientError):
                    raise
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
        raise FileNotFoundError(f"No parquet found for clip_id={clip_id} under {prefix}")

    for name in candidates:
        path = os.path.join(prefix, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No parquet found for clip_id={clip_id} under {prefix}")


def _nearest_center_indices(centers: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros_like(timestamps, dtype=int)
    idx = np.searchsorted(centers, timestamps)
    idx = np.clip(idx, 0, len(centers) - 1)
    prev_idx = np.clip(idx - 1, 0, len(centers) - 1)
    next_idx = idx
    prev_dist = np.abs(timestamps - centers[prev_idx])
    next_dist = np.abs(timestamps - centers[next_idx])
    use_prev = prev_dist <= next_dist
    out = np.where(use_prev, prev_idx, next_idx)
    return out.astype(int)


def _clip_index_from_clip_id(clip_id: str) -> int | None:
    try:
        return int(str(clip_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return None


def _relative_frame_timestamps(
    df: Any,
    *,
    clip_id: str,
    clip_index: Any,
    clip_duration_sec: float,
) -> np.ndarray:
    if "timestamp_sec" not in df.columns:
        raise ValueError(f"Frame label parquet for {clip_id} is missing timestamp_sec")
    timestamps = df["timestamp_sec"].to_numpy(dtype=float)
    if timestamps.size == 0:
        return timestamps
    if float(np.nanmax(timestamps)) <= clip_duration_sec + 1.0:
        return timestamps

    resolved_clip_index = clip_index
    if resolved_clip_index is None and "clip_index" in df.columns and len(df["clip_index"].dropna()) > 0:
        resolved_clip_index = int(df["clip_index"].dropna().iloc[0])
    if resolved_clip_index is None:
        resolved_clip_index = _clip_index_from_clip_id(clip_id)
    if resolved_clip_index is None:
        return timestamps
    return timestamps - (int(resolved_clip_index) - 1) * clip_duration_sec


def _frame_center_metrics(
    meta: List[Dict[str, Any]],
    y_pred: List[int],
    y_prob: List[float],
    *,
    frame_labels_dir: str,
    cache_dir: str,
    clip_duration_sec: float,
    pos_weight: float,
    beta: float,
    collect_predictions: bool,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and a parquet engine are required for frame-level evaluation") from exc

    by_clip: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta):
        clip_key = row.get("clip_id") or row.get("clip_path")
        by_clip.setdefault(str(clip_key), []).append(idx)

    all_true: List[int] = []
    all_pred: List[int] = []
    frame_rows: List[Dict[str, Any]] = []
    per_clip: Dict[str, Dict[str, float]] = {}

    clip_items = sorted(by_clip.items())
    for clip_id, indices in _progress(clip_items, desc="Frame-level eval", unit="clip"):
        ordered = sorted(indices, key=lambda i: float(meta[i].get("window_start_sec", 0.0)))
        centers = np.asarray(
            [
                (float(meta[i]["window_start_sec"]) + float(meta[i]["window_end_sec"])) / 2.0
                for i in ordered
            ],
            dtype=float,
        )
        clip_preds = np.asarray([int(y_pred[i]) for i in ordered], dtype=np.int64)
        clip_probs = np.asarray([float(y_prob[i]) for i in ordered], dtype=float)

        parquet_path = _resolve_frame_parquet(frame_labels_dir, clip_id, cache_dir)
        df = pd.read_parquet(parquet_path)
        if "is_playing" not in df.columns:
            raise ValueError(f"Frame label parquet for {clip_id} is missing is_playing")

        first_meta = meta[ordered[0]]
        timestamps = _relative_frame_timestamps(
            df,
            clip_id=clip_id,
            clip_index=first_meta.get("clip_index"),
            clip_duration_sec=clip_duration_sec,
        )
        labels = df["is_playing"].to_numpy(dtype=np.int64)
        finite = np.isfinite(timestamps)
        in_clip = finite & (timestamps >= 0.0) & (timestamps <= clip_duration_sec)
        timestamps = timestamps[in_clip]
        labels = labels[in_clip]
        if timestamps.size == 0:
            continue

        nearest = _nearest_center_indices(centers, timestamps)
        frame_pred = clip_preds[nearest].astype(np.int64)
        frame_prob = clip_probs[nearest].astype(float)
        frame_true = labels.astype(np.int64)

        all_true.extend(frame_true.tolist())
        all_pred.extend(frame_pred.tolist())
        per_clip[clip_id] = _binary_metrics(
            frame_true.tolist(),
            frame_pred.tolist(),
            pos_weight=pos_weight,
            beta=beta,
        )

        if collect_predictions:
            frame_idx_values = (
                df.loc[in_clip, "frame_idx"].to_numpy(dtype=int).tolist()
                if "frame_idx" in df.columns
                else list(range(len(frame_true)))
            )
            for frame_idx, timestamp, truth, pred, prob, window_i in zip(
                frame_idx_values,
                timestamps.tolist(),
                frame_true.tolist(),
                frame_pred.tolist(),
                frame_prob.tolist(),
                nearest.tolist(),
            ):
                frame_rows.append(
                    {
                        "clip_id": clip_id,
                        "frame_idx": int(frame_idx),
                        "timestamp_sec": float(timestamp),
                        "label": int(truth),
                        "pred": int(pred),
                        "prob_play": float(prob),
                        "assigned_window_start_sec": float(meta[ordered[int(window_i)]]["window_start_sec"]),
                        "assigned_window_end_sec": float(meta[ordered[int(window_i)]]["window_end_sec"]),
                    }
                )

    pooled = _binary_metrics(all_true, all_pred, pos_weight=pos_weight, beta=beta)
    pooled["n_clips"] = float(len(per_clip))
    return pooled, per_clip, frame_rows


def _normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 1, 3, 1, 1)
    return (frames - mean) / std


def _encode_frames(frames: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    batch, timesteps, channels, height, width = frames.shape
    frames = _normalize_frames(frames)
    flat = frames.view(batch * timesteps, channels, height, width)
    with torch.no_grad():
        feats = encoder(flat)
    return feats.view(batch, timesteps, -1)


def _prepare_model_inputs(
    features: torch.Tensor | Dict[str, torch.Tensor],
    fusion: str,
    frame_encoder: torch.nn.Module | None,
    expected_e2e_dim: int | None = None,
) -> torch.Tensor | Dict[str, torch.Tensor]:
    if isinstance(features, dict):
        video = features["video"]
        if frame_encoder is not None:
            video = _encode_frames(video, frame_encoder)
        if expected_e2e_dim is not None:
            e2e = features["e2e"]
            if e2e.shape[-1] > expected_e2e_dim:
                e2e = e2e[..., :expected_e2e_dim]
            elif e2e.shape[-1] < expected_e2e_dim:
                pad = expected_e2e_dim - e2e.shape[-1]
                e2e = torch.nn.functional.pad(e2e, (0, pad))
            features = {**features, "e2e": e2e}
        if fusion == "early":
            return torch.cat([video, features["e2e"]], dim=-1)
        if fusion == "late":
            return {"video": video, "e2e": features["e2e"]}
        return video
    if frame_encoder is not None:
        return _encode_frames(features, frame_encoder)
    return features


def _load_checkpoint(
    path: str,
    input_dim: int,
    max_len: int,
    *,
    fusion: str,
    video_input_dim: int | None = None,
    feature_input_dim: int | None = None,
) -> Tuple[torch.nn.Module, str, int | None, int | None]:
    data = torch.load(path, map_location="cpu")
    fusion_mode = str(data.get("fusion") or fusion or "none")
    cfg_payload = data.get("config") or {}
    if fusion_mode == "late":
        cfg = FusionTransformerConfig(
            video_input_dim=video_input_dim or input_dim,
            feature_input_dim=feature_input_dim or input_dim,
            max_len=max_len,
        )
        if cfg_payload:
            allowed = set(cfg.__dict__.keys())
            filtered = {k: v for k, v in cfg_payload.items() if k in allowed}
            cfg = FusionTransformerConfig(**{**cfg.__dict__, **filtered})
        model = LateFusionTransformerClassifier(cfg)
        expected_input_dim = int(cfg.video_input_dim)
        expected_feature_dim = int(cfg.feature_input_dim)
    else:
        cfg = TransformerConfig(input_dim=input_dim, max_len=max_len)
        if cfg_payload:
            allowed = set(cfg.__dict__.keys())
            filtered = {k: v for k, v in cfg_payload.items() if k in allowed}
            cfg = TransformerConfig(**{**cfg.__dict__, **filtered})
        model = TransformerClassifier(cfg)
        expected_input_dim = int(cfg.input_dim)
        expected_feature_dim = None
    model.load_state_dict(data["model"])
    return model, fusion_mode, expected_input_dim, expected_feature_dim


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on window and segment metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--e2e-features-dir", default=None)
    parser.add_argument(
        "--e2e-feature-subset",
        choices=[
            "all",
            "base",
            "counts",
            "pairwise",
            "net_dist",
            "centroids",
            "mocon",
            "spatial",
            "pose_angles",
            "actions",
            "temporal",
        ],
        default="all",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gap-sec", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--media-samples", type=int, default=3)
    parser.add_argument("--use-raw-frames", action="store_true")
    parser.add_argument("--fusion", choices=["none", "early", "late"], default="none")
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--f-beta", type=float, default=2.0)
    parser.add_argument("--e2e-only", action="store_true", help="Evaluate using only E2E parquet features (no video embeddings).")
    parser.add_argument("--s3-cache-dir", default=None)
    parser.add_argument("--clip-cache-dir", default=None)
    parser.add_argument("--per-clip-output", default=None)
    parser.add_argument("--frame-eval", choices=["none", "center"], default="none")
    parser.add_argument("--frame-labels-dir", default=None)
    parser.add_argument("--frame-output", default=None)
    parser.add_argument("--frame-predictions-output", default=None)
    parser.add_argument("--clip-duration-sec", type=float, default=60.0)
    parser.add_argument("--timeline-output", default=None)
    parser.add_argument("--timeline-group-field", default="source_id")
    parser.add_argument("--timeline-max-groups", type=int, default=12)
    parser.add_argument("--skip-media", action="store_true", help="Skip GIF/timeline media generation.")
    parser.add_argument("--wandb-project", default="volleyball-playtime")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    cfg = EvalConfig(
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        pose_dir=args.pose_dir,
        e2e_features_dir=args.e2e_features_dir,
        e2e_feature_subset=args.e2e_feature_subset,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size if args.batch_size is not None else (4 if args.use_raw_frames else 32),
        num_workers=args.num_workers if args.num_workers is not None else (0 if args.use_raw_frames else 4),
        use_raw_frames=args.use_raw_frames,
        fusion=args.fusion,
        s3_cache_dir=args.s3_cache_dir,
        clip_cache_dir=args.clip_cache_dir,
        e2e_only=args.e2e_only,
    )

    if not cfg.use_raw_frames and not cfg.features_dir and not cfg.e2e_only:
        raise ValueError("--features-dir is required unless --use-raw-frames or --e2e-only is set")
    if cfg.e2e_only and not cfg.e2e_features_dir:
        raise ValueError("--e2e-features-dir is required for --e2e-only")
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
        num_frames=cfg.num_frames,
        use_raw_frames=cfg.use_raw_frames,
        e2e_only=cfg.e2e_only,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_windows,
    )

    sample_batch = next(iter(loader))
    sample_features = sample_batch[0]
    if isinstance(sample_features, dict):
        video_tensor = sample_features["video"]
        input_dim = video_tensor.shape[-1]
        video_input_dim = input_dim
        feature_input_dim = int(sample_features["e2e"].shape[-1])
    else:
        input_dim = sample_features.shape[-1]
        video_input_dim = None
        feature_input_dim = None
    if cfg.use_raw_frames:
        video_input_dim = 512

    model, fusion_mode, expected_input_dim, expected_feature_dim = _load_checkpoint(
        cfg.checkpoint_path,
        input_dim,
        cfg.num_frames,
        fusion=cfg.fusion,
        video_input_dim=video_input_dim,
        feature_input_dim=feature_input_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.fusion_mode = fusion_mode  # type: ignore[attr-defined]

    frame_encoder = None
    if cfg.use_raw_frames:
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        encoder.fc = torch.nn.Identity()
        encoder.eval()
        encoder.to(device)
        frame_encoder = encoder

    expected_e2e_dim = None
    if fusion_mode == "early" and expected_input_dim is not None and video_input_dim is not None:
        expected_e2e_dim = max(0, expected_input_dim - video_input_dim)
    if fusion_mode == "late" and expected_feature_dim is not None:
        expected_e2e_dim = expected_feature_dim

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        for feats, labels, batch_meta in _progress(loader, desc="Window inference", unit="batch", total=len(loader)):
            if isinstance(feats, dict):
                feats = {key: value.to(device) for key, value in feats.items()}
            else:
                feats = feats.to(device)
            model_inputs = _prepare_model_inputs(
                feats,
                fusion_mode,
                frame_encoder,
                expected_e2e_dim=expected_e2e_dim,
            )
            logits = model(model_inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds = [1 if p >= args.pred_threshold else 0 for p in probs]
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
            y_prob.extend(probs)
            meta.extend(batch_meta)

    window_metrics = _window_metrics(y_true, y_pred, beta=args.f_beta)
    segment_metrics = _segment_metrics(y_true, y_pred, meta, args.gap_sec, args.iou, beta=args.f_beta)
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = max(1, len(y_true) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)
    pooled = _binary_metrics(y_true, y_pred, pos_weight=pos_weight, beta=args.f_beta)
    per_clip: Dict[str, Dict[str, float]] = {}
    by_clip: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta):
        clip_key = row.get("clip_id") or row.get("clip_path")
        by_clip.setdefault(str(clip_key), []).append(idx)
    for clip_key, indices in by_clip.items():
        clip_true = [y_true[i] for i in indices]
        clip_pred = [y_pred[i] for i in indices]
        per_clip[clip_key] = _binary_metrics(clip_true, clip_pred, pos_weight=pos_weight, beta=args.f_beta)

    print(f"\n=== test set inference (threshold={args.pred_threshold}) ===")
    for clip_id in sorted(per_clip):
        print(_format_metrics_row(clip_id, per_clip[clip_id]))
    print("\npooled:")
    print(_format_metrics_row("ALL", pooled))

    results = {
        "window": window_metrics,
        "segment": segment_metrics,
        "pooled": pooled,
    }

    frame_pooled: Dict[str, float] | None = None
    frame_per_clip: Dict[str, Dict[str, float]] | None = None
    frame_rows: List[Dict[str, Any]] = []
    if args.frame_eval == "center":
        frame_labels_dir = args.frame_labels_dir or cfg.e2e_features_dir
        if not frame_labels_dir:
            raise ValueError("--frame-labels-dir is required for --frame-eval center when --e2e-features-dir is not set")
        frame_cache_dir = cfg.s3_cache_dir or os.path.join(os.getcwd(), "data", "s3_cache")
        frame_pooled, frame_per_clip, frame_rows = _frame_center_metrics(
            meta,
            y_pred,
            y_prob,
            frame_labels_dir=frame_labels_dir,
            cache_dir=frame_cache_dir,
            clip_duration_sec=args.clip_duration_sec,
            pos_weight=pos_weight,
            beta=args.f_beta,
            collect_predictions=bool(args.frame_predictions_output),
        )
        results["frame_center"] = frame_pooled
        print("\nframe-level center-window assignment:")
        for clip_id in sorted(frame_per_clip):
            print(_format_metrics_row(clip_id, frame_per_clip[clip_id]))
        print("\nframe pooled:")
        print(_format_metrics_row("ALL", frame_pooled))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    per_clip_output = args.per_clip_output
    if per_clip_output is None:
        per_clip_output = os.path.join(os.path.dirname(args.output), "test_clip_metrics.json")
    report: Dict[str, Any] = {"per_clip": per_clip, "pooled": pooled}
    if frame_per_clip is not None and frame_pooled is not None:
        report["frame_center_per_clip"] = frame_per_clip
        report["frame_center_pooled"] = frame_pooled
    with open(per_clip_output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if frame_pooled is not None:
        frame_output = args.frame_output
        if frame_output is None:
            frame_output = os.path.join(os.path.dirname(args.output), "test_frame_metrics.json")
        with open(frame_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "per_clip": frame_per_clip,
                    "pooled": frame_pooled,
                    "table_row": _table_metrics_row("Window Transformer (center-mapped frames)", frame_pooled),
                    "mapping": "center",
                },
                handle,
                indent=2,
            )

    if args.frame_predictions_output:
        with open(args.frame_predictions_output, "w", encoding="utf-8") as handle:
            for row in frame_rows:
                handle.write(json.dumps(row) + "\n")

    if args.timeline_output:
        _plot_source_timelines(
            meta,
            y_true,
            y_pred,
            args.timeline_output,
            group_field=args.timeline_group_field,
            max_groups=args.timeline_max_groups,
        )

    wandb_logger = WandbLogger(
        WandbConfig(project=args.wandb_project, run_name=args.wandb_run, enabled=True),
        config=cfg.__dict__,
    )
    wandb_logger.log(
        {
            "window_precision": window_metrics["precision"],
            "window_recall": window_metrics["recall"],
            "window_f1": window_metrics["f1"],
            "window_f_beta": window_metrics["f_beta"],
        }
    )
    wandb_logger.log(
        {
            "segment_precision": segment_metrics["precision"],
            "segment_recall": segment_metrics["recall"],
            "segment_f1": segment_metrics["f1"],
            "segment_f_beta": segment_metrics["f_beta"],
        }
    )
    if frame_pooled is not None:
        wandb_logger.log(
            {
                "frame_center_precision": frame_pooled["precision"],
                "frame_center_recall": frame_pooled["recall"],
                "frame_center_f1": frame_pooled["f1"],
                "frame_center_f_beta": frame_pooled["f_beta"],
            }
        )
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
    if wandb_module and not args.skip_media:
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
                    cache_dir=cfg.clip_cache_dir or cfg.s3_cache_dir,
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
