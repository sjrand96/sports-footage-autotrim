#!/usr/bin/env python3
"""Run a tiny training job from the repository's sample Label Studio export."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.transformer_classifier import TransformerClassifier, TransformerConfig


@dataclass
class SampleTrainConfig:
    project: str = "volleyball-playtime-sample"
    entity: str | None = None
    run_name: str = "sample-data-transformer-smoke-test"
    sample_export: str = str(PROJECT_ROOT / "data/project-1-at-2026-04-28-04-06-975ac587.json")
    sample_frame: str = str(PROJECT_ROOT / "data/example_frame.png")
    output_dir: str = "outputs/sample_wandb"
    wandb_mode: str = "online"
    seed: int = 13
    sequence_len: int = 16
    input_dim: int = 12
    batch_size: int = 32
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    window_size: float = 3.0
    stride: float = 1.0
    min_overlap_ratio: float = 0.5
    val_ratio: float = 0.25
    play_labels: str = "playing,play,ball_in_play"
    noise: float = 0.08
    log_val_frames: bool = True
    max_val_frames: int = 64
    label_frame_rate: float = 30.0


def _canonical_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_sample_export(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("items") or payload.get("tasks") or []
        return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported sample export shape in {path}")


def _row_video_ref(row: Dict[str, Any]) -> str:
    data = row.get("data")
    if isinstance(data, dict) and isinstance(data.get("video"), str):
        return data["video"]
    return str(row.get("video") or row.get("clip_path") or "")


def _pick_latest_annotation(annotations: Any) -> Dict[str, Any] | None:
    if not isinstance(annotations, list):
        return None
    candidates = [item for item in annotations if isinstance(item, dict) and not item.get("was_cancelled")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))


def _parse_video_label_items(items: Any, label_frame_rate: float) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    segments: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value") if isinstance(item.get("value"), dict) else item
        labels = value.get("timelinelabels") or value.get("labels") or [value.get("label", "")]
        label = str(labels[0] if labels else "")
        for range_item in value.get("ranges", []):
            if not isinstance(range_item, dict):
                continue
            start = range_item.get("start")
            end = range_item.get("end")
            if start is None or end is None:
                continue
            start_sec = float(start) / label_frame_rate
            end_sec = float(end) / label_frame_rate
            segments.append(
                {
                    "start": start_sec,
                    "end": end_sec,
                    "label": label,
                    "label_key": _canonical_label(label),
                }
            )
    return segments


def _parse_segments(row: Dict[str, Any], label_frame_rate: float) -> List[Dict[str, Any]]:
    """Parse Label Studio timeline labels into canonical segments."""
    segments: List[Dict[str, Any]] = []

    segments.extend(_parse_video_label_items(row.get("videoLabels"), label_frame_rate))

    latest = _pick_latest_annotation(row.get("annotations"))
    if latest is not None:
        segments.extend(_parse_video_label_items(latest.get("result"), label_frame_rate))

    for item in row.get("annotations", []):
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", item.get("start", item.get("start_time")))
        end = item.get("end_sec", item.get("end", item.get("end_time")))
        label = str(item.get("label", item.get("category", item.get("name", ""))))
        if start is None or end is None:
            continue
        segments.append({"start": float(start), "end": float(end), "label": label, "label_key": _canonical_label(label)})

    return sorted(segments, key=lambda seg: (seg["start"], seg["end"]))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _make_windows(
    rows: Sequence[Dict[str, Any]],
    cfg: SampleTrainConfig,
) -> List[Dict[str, Any]]:
    play_labels = {_canonical_label(label) for label in cfg.play_labels.split(",") if label.strip()}
    windows: List[Dict[str, Any]] = []
    for row in rows:
        segments = _parse_segments(row, cfg.label_frame_rate)
        if not segments:
            continue
        video = _row_video_ref(row)
        video_duration = _video_duration_sec(video)
        duration = max([seg["end"] for seg in segments] + ([video_duration] if video_duration else []))
        clip_id = str(row.get("id") or row.get("clip_id") or Path(video or "sample").stem)
        max_start = max(0.0, duration - cfg.window_size)
        starts = np.arange(0.0, max_start + 1e-6, cfg.stride)
        for start in starts:
            end = float(start + cfg.window_size)
            play_overlap = sum(
                _overlap(start, end, seg["start"], seg["end"])
                for seg in segments
                if seg["label_key"] in play_labels
            )
            label = int((play_overlap / cfg.window_size) >= cfg.min_overlap_ratio)
            dominant = max(segments, key=lambda seg: _overlap(start, end, seg["start"], seg["end"]))
            windows.append(
                {
                    "clip_id": clip_id,
                    "video": video,
                    "window_start": round(float(start), 3),
                    "window_end": round(end, 3),
                    "label": label,
                    "dominant_label": dominant["label"],
                    "play_overlap_ratio": round(play_overlap / cfg.window_size, 4),
                    "duration": duration,
                    "segments": segments,
                }
            )
    if not windows:
        raise ValueError("No trainable windows could be built from the sample export.")
    return windows


def _feature_at_time(t: float, duration: float, segments: Sequence[Dict[str, Any]], play_labels: set[str]) -> List[float]:
    active = next((seg for seg in segments if seg["start"] <= t <= seg["end"]), None)
    is_play = 1.0 if active and active["label_key"] in play_labels else 0.0
    is_downtime = 1.0 if active and active["label_key"] not in play_labels else 0.0
    if active:
        seg_duration = max(1e-6, active["end"] - active["start"])
        seg_progress = (t - active["start"]) / seg_duration
        dist_start = (t - active["start"]) / max(1.0, duration)
        dist_end = (active["end"] - t) / max(1.0, duration)
    else:
        seg_progress = 0.0
        dist_start = 0.0
        dist_end = 0.0
    normalized_time = t / max(1.0, duration)
    return [
        normalized_time,
        np.sin(2 * np.pi * normalized_time),
        np.cos(2 * np.pi * normalized_time),
        is_play,
        is_downtime,
        seg_progress,
        dist_start,
        dist_end,
    ]


def _window_to_features(window: Dict[str, Any], cfg: SampleTrainConfig, rng: np.random.Generator) -> np.ndarray:
    play_labels = {_canonical_label(label) for label in cfg.play_labels.split(",") if label.strip()}
    start = float(window["window_start"])
    end = float(window["window_end"])
    duration = float(window["duration"])
    timestamps = np.linspace(start, end, cfg.sequence_len, endpoint=False) + ((end - start) / cfg.sequence_len / 2)
    rows = []
    for idx, t in enumerate(timestamps):
        base = _feature_at_time(float(t), duration, window["segments"], play_labels)
        local_progress = idx / max(1, cfg.sequence_len - 1)
        rows.append(
            [
                *base,
                local_progress,
                float(window["play_overlap_ratio"]),
                float(window["window_start"]) / max(1.0, duration),
                float(window["window_end"]) / max(1.0, duration),
            ]
        )
    features = np.asarray(rows, dtype=np.float32)
    features += rng.normal(0.0, cfg.noise, size=features.shape).astype(np.float32)
    return features


def _make_dataset(windows: Sequence[Dict[str, Any]], cfg: SampleTrainConfig, seed_offset: int = 0) -> TensorDataset:
    rng = np.random.default_rng(cfg.seed + seed_offset)
    features = np.stack([_window_to_features(window, cfg, rng) for window in windows])
    labels = np.asarray([int(window["label"]) for window in windows], dtype=np.int64)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def _split_windows(windows: Sequence[Dict[str, Any]], cfg: SampleTrainConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_label: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for window in windows:
        by_label[int(window["label"])].append(dict(window))

    rng = random.Random(cfg.seed)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for label_windows in by_label.values():
        rng.shuffle(label_windows)
        val_count = max(1, int(round(len(label_windows) * cfg.val_ratio))) if len(label_windows) > 1 else 0
        val.extend(label_windows[:val_count])
        train.extend(label_windows[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    if not train or not val:
        raise ValueError("Sample split produced an empty train or validation set.")
    return train, val


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def _precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float, float, float]:
    true_play = y_true == 1
    pred_play = y_pred == 1
    tp = (true_play & pred_play).sum().item()
    fp = (~true_play & pred_play).sum().item()
    fn = (true_play & ~pred_play).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def _class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights


def _import_wandb():
    """Import the installed W&B SDK even when ./wandb run logs exist in cwd."""
    cwd = Path.cwd().resolve()
    original_path = list(sys.path)
    shadow = sys.modules.get("wandb")
    if shadow is not None and not hasattr(shadow, "init"):
        del sys.modules["wandb"]
    try:
        sys.path = [
            path
            for path in original_path
            if path
            and Path(path).resolve() != cwd
            and Path(path).resolve() != (cwd / "wandb")
        ]
        wandb = importlib.import_module("wandb")
    finally:
        sys.path = original_path
    if not hasattr(wandb, "init"):
        raise ImportError("Imported a module named wandb, but it is not the W&B SDK.")
    return wandb


def _local_video_path(video_ref: str) -> Path | None:
    if not video_ref:
        return None
    if video_ref.startswith("file://"):
        path = Path(video_ref[7:])
    elif "://" in video_ref:
        return None
    else:
        path = Path(video_ref)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path if path.is_file() else None


def _video_duration_sec(video_ref: str) -> float | None:
    path = _local_video_path(video_ref)
    if path is None:
        return None

    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frames and frames > 0:
            return float(frames / fps)
        duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        return float(duration_msec / 1000.0) if duration_msec and duration_msec > 0 else None
    finally:
        cap.release()


def _extract_frame_rgb(video_ref: str, timestamp_sec: float) -> np.ndarray | None:
    path = _local_video_path(video_ref)
    if path is None:
        return None

    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _wandb_frame(wandb: Any, video_ref: str, timestamp_sec: float, caption: str) -> Any:
    frame_rgb = _extract_frame_rgb(video_ref, timestamp_sec)
    if frame_rgb is None:
        return None
    return wandb.Image(frame_rgb, caption=caption)


def _init_wandb(cfg: SampleTrainConfig):
    if cfg.wandb_mode == "disabled":
        return None

    try:
        wandb = _import_wandb()
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. Install it with `python -m pip install wandb`, "
            "then run `wandb login`, or pass `--wandb-mode disabled` for a local-only smoke test."
        ) from exc

    os.environ["WANDB_MODE"] = cfg.wandb_mode
    return wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config=asdict(cfg),
        job_type="sample_train",
    )


def _log_confusion_matrix(run, y_true: torch.Tensor, y_pred: torch.Tensor, step: int) -> None:
    if run is None:
        return

    wandb = _import_wandb()

    run.log(
        {
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true.cpu().tolist(),
                preds=y_pred.cpu().tolist(),
                class_names=["downtime", "playtime"],
            )
        },
        step=step,
    )


def _log_sample_dashboard_inputs(run, cfg: SampleTrainConfig, rows: Sequence[Dict[str, Any]], windows: Sequence[Dict[str, Any]]) -> None:
    if run is None:
        return

    wandb = _import_wandb()

    segment_rows = []
    segment_frame_budget = cfg.max_val_frames if cfg.log_val_frames else 0
    for row in rows:
        video = _row_video_ref(row)
        clip_id = str(row.get("id") or row.get("clip_id") or Path(video or "sample").stem)
        for seg_idx, seg in enumerate(_parse_segments(row, cfg.label_frame_rate)):
            midpoint = (float(seg["start"]) + float(seg["end"])) / 2.0
            frame_media = None
            if seg_idx < segment_frame_budget:
                frame_media = _wandb_frame(
                    wandb,
                    video,
                    midpoint,
                    f"{clip_id} segment {seg['start']:.2f}-{seg['end']:.2f}s {seg['label']}",
                )
            segment_rows.append([frame_media, clip_id, video, seg["start"], seg["end"], seg["label"]])

    window_rows = []
    window_frame_budget = cfg.max_val_frames if cfg.log_val_frames else 0
    for idx, window in enumerate(windows):
        midpoint = (float(window["window_start"]) + float(window["window_end"])) / 2.0
        frame_media = None
        if idx < window_frame_budget:
            frame_media = _wandb_frame(
                wandb,
                str(window.get("video", "")),
                midpoint,
                (
                    f"{window['clip_id']} window "
                    f"{window['window_start']}-{window['window_end']}s "
                    f"{'playtime' if window['label'] else 'downtime'}"
                ),
            )
        window_rows.append(
            [
                frame_media,
                window["clip_id"],
                window["window_start"],
                window["window_end"],
                "playtime" if window["label"] else "downtime",
                window["dominant_label"],
                window["play_overlap_ratio"],
            ]
        )
    run.log(
        {
            "data/source_segments": wandb.Table(
                columns=["frame", "clip_id", "video", "start", "end", "label"],
                data=segment_rows,
            ),
            "data/windows": wandb.Table(
                columns=["frame", "clip_id", "window_start", "window_end", "label", "dominant_label", "play_overlap_ratio"],
                data=window_rows,
            ),
            "data/label_counts": wandb.Histogram([int(window["label"]) for window in windows]),
        },
        step=0,
    )

    frame_path = Path(cfg.sample_frame)
    if frame_path.exists():
        run.log({"data/example_frame": wandb.Image(str(frame_path), caption=frame_path.name)}, step=0)


def _log_prediction_table(
    run,
    windows: Sequence[Dict[str, Any]],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    probabilities: torch.Tensor,
    cfg: SampleTrainConfig,
    step: int,
) -> None:
    if run is None:
        return

    wandb = _import_wandb()

    data = []
    frame_budget = cfg.max_val_frames if cfg.log_val_frames else 0
    for idx, (window, truth, pred, probs) in enumerate(zip(windows, y_true.tolist(), y_pred.tolist(), probabilities.tolist())):
        midpoint = (float(window["window_start"]) + float(window["window_end"])) / 2.0
        frame_media = None
        if idx < frame_budget:
            frame_rgb = _extract_frame_rgb(str(window.get("video", "")), midpoint)
            if frame_rgb is not None:
                frame_media = wandb.Image(
                    frame_rgb,
                    caption=(
                        f"{window['clip_id']} "
                        f"{window['window_start']}-{window['window_end']}s "
                        f"true={'playtime' if truth else 'downtime'} "
                        f"pred={'playtime' if pred else 'downtime'}"
                    ),
                )
        data.append(
            [
                frame_media,
                window["clip_id"],
                window["window_start"],
                window["window_end"],
                "playtime" if truth else "downtime",
                "playtime" if pred else "downtime",
                float(probs[1]),
                window["dominant_label"],
                window["play_overlap_ratio"],
            ]
        )
    run.log(
        {
            "val/predictions": wandb.Table(
                columns=[
                    "frame",
                    "clip_id",
                    "window_start",
                    "window_end",
                    "true_label",
                    "pred_label",
                    "playtime_probability",
                    "dominant_label",
                    "play_overlap_ratio",
                ],
                data=data,
            )
        },
        step=step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sample-data W&B training job.")
    parser.add_argument("--project", default=SampleTrainConfig.project)
    parser.add_argument("--entity", default=None, help="Optional W&B team/entity.")
    parser.add_argument("--run-name", default=SampleTrainConfig.run_name)
    parser.add_argument("--sample-export", default=SampleTrainConfig.sample_export)
    parser.add_argument("--sample-frame", default=SampleTrainConfig.sample_frame)
    parser.add_argument("--output-dir", default=SampleTrainConfig.output_dir)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=SampleTrainConfig.wandb_mode)
    parser.add_argument("--epochs", type=int, default=SampleTrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=SampleTrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=SampleTrainConfig.lr)
    parser.add_argument("--seed", type=int, default=SampleTrainConfig.seed)
    parser.add_argument("--window-size", type=float, default=SampleTrainConfig.window_size)
    parser.add_argument("--stride", type=float, default=SampleTrainConfig.stride)
    parser.add_argument("--play-labels", default=SampleTrainConfig.play_labels)
    parser.add_argument("--no-val-frames", action="store_true", help="Do not attach video frames to the W&B validation table.")
    parser.add_argument("--max-val-frames", type=int, default=SampleTrainConfig.max_val_frames)
    parser.add_argument("--label-frame-rate", type=float, default=SampleTrainConfig.label_frame_rate)
    args = parser.parse_args()

    cfg = SampleTrainConfig(
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
        sample_export=args.sample_export,
        sample_frame=args.sample_frame,
        output_dir=args.output_dir,
        wandb_mode=args.wandb_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        window_size=args.window_size,
        stride=args.stride,
        play_labels=args.play_labels,
        log_val_frames=not args.no_val_frames,
        max_val_frames=args.max_val_frames,
        label_frame_rate=args.label_frame_rate,
    )

    _set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = _init_wandb(cfg)

    sample_rows = _load_sample_export(Path(cfg.sample_export))
    windows = _make_windows(sample_rows, cfg)
    train_windows, val_windows = _split_windows(windows, cfg)
    _log_sample_dashboard_inputs(run, cfg, sample_rows, windows)

    train_loader = DataLoader(_make_dataset(train_windows, cfg, seed_offset=1), batch_size=cfg.batch_size, shuffle=True)
    val_features, val_labels = _make_dataset(val_windows, cfg, seed_offset=2).tensors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = TransformerConfig(input_dim=cfg.input_dim, max_len=cfg.sequence_len, model_dim=64, num_layers=2)
    model = TransformerClassifier(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_labels = train_loader.dataset.tensors[1]
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_labels).to(device))

    best_f1 = -1.0
    best_path = output_dir / "sample_best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        examples_seen = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += _accuracy(logits.detach(), labels) * batch_size
            examples_seen += batch_size

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features.to(device))
            val_loss = criterion(val_logits, val_labels.to(device)).item()
            val_preds = torch.argmax(val_logits.cpu(), dim=1)
            val_probs = torch.softmax(val_logits.cpu(), dim=1)
            val_acc = _accuracy(val_logits.cpu(), val_labels)
            precision, recall, f1 = _precision_recall_f1(val_labels, val_preds)

        metrics = {
            "epoch": epoch + 1,
            "train/loss": total_loss / examples_seen,
            "train/accuracy": total_acc / examples_seen,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if run is not None:
            run.log(metrics, step=epoch + 1)

        print(
            f"epoch={epoch + 1:02d} "
            f"train_loss={metrics['train/loss']:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "model_config": asdict(model_cfg)}, best_path)

    final_log_step = cfg.epochs + 1
    _log_confusion_matrix(run, val_labels, val_preds, final_log_step)
    _log_prediction_table(run, val_windows, val_labels, val_preds, val_probs, cfg, final_log_step)

    if run is not None:
        wandb = _import_wandb()

        artifact = wandb.Artifact("sample-data-transformer-checkpoint", type="model")
        artifact.add_file(str(best_path))
        run.log_artifact(artifact)
        run.summary["best_val_f1"] = best_f1
        run.summary["checkpoint_path"] = str(best_path)
        run.summary["sample_export"] = cfg.sample_export
        run.summary["train_windows"] = len(train_windows)
        run.summary["val_windows"] = len(val_windows)
        run.finish()

    print(
        f"Trained on {len(train_windows)} sample windows; validated on {len(val_windows)} windows. "
        f"Best checkpoint saved to {best_path} (F1={best_f1:.4f})"
    )


if __name__ == "__main__":
    main()
