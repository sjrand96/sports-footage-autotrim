#!/usr/bin/env python3
"""Train a transformer on raw frame windows using parquet train/test splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _same_path(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


sys.path = [entry for entry in sys.path if not _same_path(entry, REPO_ROOT)]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from src.data.raw_frame_window_dataset import (  # noqa: E402
    RawFrameWindowDataset,
    collate_raw_frame_windows,
)
from src.models.transformer_classifier import (  # noqa: E402
    TransformerClassifier,
    TransformerConfig,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import (  # noqa: E402
    class_weight_ratio_from_counts,
    tversky_coefficients_from_counts,
    train_label_counts,
)
from models.lstm.encoders import resolve_device  # noqa: E402
from src.training.wandb_logger import WandbConfig, WandbLogger  # noqa: E402


@dataclass
class TrainConfig:
    train_parquet: str
    test_parquet: str
    output_dir: str
    s3_cache_dir: str
    clip_cache_dir: str
    num_frames: int = 30
    image_size: int = 224
    window_radius: int = 15
    boundary_margin: int = 0
    train_frame_stride: int = 1
    batch_size: int = 16
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pred_threshold: float = 0.35
    checkpoint_metric: str = "loss"
    early_stop_patience: int | None = None
    num_workers: int = 2
    log_every: int = 20
    device: str | None = None
    finetune_encoder: bool = False
    encoder_lr: float | None = None
    wandb_project: str = "volleyball-playtime"
    wandb_entity: str | None = "cs348k-sports-footage-autotrim"
    wandb_run: str | None = None
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device).view(1, 1, 3, 1, 1)
    return (frames - mean) / std


def _encode_frames(frames: torch.Tensor, encoder: nn.Module, *, use_grad: bool) -> torch.Tensor:
    batch, timesteps, channels, height, width = frames.shape
    frames = _normalize_frames(frames)
    flat = frames.view(batch * timesteps, channels, height, width)
    if use_grad:
        feats = encoder(flat)
    else:
        with torch.no_grad():
            feats = encoder(flat)
    return feats.view(batch, timesteps, -1)


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    t = targets.to(dtype=probs.dtype)
    tp = (probs * t).sum()
    fp = (probs * (1.0 - t)).sum()
    fn = ((1.0 - probs) * t).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


def masked_tversky_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    m = mask.to(dtype=probs.dtype)
    t = targets.to(dtype=probs.dtype)
    tp = (probs * t * m).sum()
    fp = (probs * (1.0 - t) * m).sum()
    fn = ((1.0 - probs) * t * m).sum()
    denom = m.sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    loss = 1.0 - tversky
    return loss * denom, denom


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, pos_weight: float) -> dict[str, float]:
    yt = y_true.astype(np.int64).ravel()
    yp = y_pred.astype(np.int64).ravel()
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
        "accuracy": (tp + tn) / max(n, 1),
        "cost": pos_weight * fn + fp,
        "n_frames": float(n),
    }


def format_metrics_row(clip_id: str, m: dict[str, float]) -> str:
    return (
        f"{clip_id:20s}  "
        f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  f1={m['f1']:.3f}  "
        f"acc={m['accuracy']:.3f}  cost={m['cost']:.0f}  "
        f"TP={int(m['tp']):4d} FP={int(m['fp']):4d} TN={int(m['tn']):4d} FN={int(m['fn']):4d}  "
        f"n={int(m['n_frames'])}"
    )


def save_loss_history(history: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def plot_loss_curve(history: list[dict[str, Any]], path: Path, *, best_epoch: int | None = None) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    test_loss = [float(h["test_loss"]) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train", marker="o", markersize=4)
    ax.plot(epochs, test_loss, label="test", marker="o", markersize=4)
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Tversky loss")
    ax.set_title("Training loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _is_better_checkpoint(metric: str, metrics: dict[str, float], *, best_loss: float, best_recall: float, best_cost: float) -> bool:
    if metric == "loss":
        return metrics["loss"] < best_loss
    if metric == "cost":
        return metrics["cost"] < best_cost or (metrics["cost"] == best_cost and metrics["recall"] > best_recall)
    return metrics["recall"] > best_recall or (metrics["recall"] == best_recall and metrics["cost"] < best_cost)


def _checkpoint_save_msg(metric: str, metrics: dict[str, float]) -> str:
    if metric == "loss":
        return f"test_loss={metrics['loss']:.4f}"
    if metric == "cost":
        return f"cost={metrics['cost']:.0f} recall={metrics['recall']:.4f}"
    return f"recall={metrics['recall']:.4f} cost={metrics['cost']:.0f}"


def _collect_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    pos_weight: float,
) -> dict[str, float]:
    return binary_metrics(np.asarray(list(y_true)), np.asarray(list(y_pred)), pos_weight=pos_weight)


def evaluate_loader(
    model: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    *,
    tversky_alpha: float,
    tversky_beta: float,
    boundary_margin: int,
    pred_threshold: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], List[int], List[int]]:
    model.eval()
    loss_sum = 0.0
    mask_sum = 0.0
    per_clip: Dict[str, List[Tuple[int, int, int]]] = {}
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            frames = batch["video"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            loss_mask = batch["loss_mask"].to(device).unsqueeze(1)
            feats = _encode_frames(frames, encoder, use_grad=False)
            logits = model(feats)
            pos_logits = logits[:, 1] - logits[:, 0]

            if boundary_margin > 0:
                wsum, msum = masked_tversky_loss_sum(
                    pos_logits, labels, loss_mask, alpha=tversky_alpha, beta=tversky_beta
                )
                loss_sum += float(wsum.item())
                mask_sum += float(msum.item())
            else:
                loss = tversky_loss(pos_logits, labels, alpha=tversky_alpha, beta=tversky_beta)
                n = float(labels.numel())
                loss_sum += float(loss.item()) * n
                mask_sum += n

            probs = torch.sigmoid(pos_logits).cpu().numpy().ravel()
            preds = (probs >= pred_threshold).astype(np.int64)
            ys = labels.long().cpu().numpy().ravel().astype(np.int64)

            all_true.extend(ys.tolist())
            all_pred.extend(preds.tolist())

            for clip_id, frame_idx, y_true, y_pred in zip(
                batch["clip_id"], batch["frame_idx"].tolist(), ys.tolist(), preds.tolist()
            ):
                per_clip.setdefault(str(clip_id), []).append((int(frame_idx), int(y_true), int(y_pred)))

    pooled = _collect_metrics(all_true, all_pred, pos_weight=pos_weight)
    pooled["loss"] = loss_sum / max(mask_sum, 1.0)

    per_clip_metrics: Dict[str, dict[str, float]] = {}
    for clip_id, rows in per_clip.items():
        rows = sorted(rows, key=lambda r: r[0])
        y_true = [r[1] for r in rows]
        y_pred = [r[2] for r in rows]
        per_clip_metrics[clip_id] = _collect_metrics(y_true, y_pred, pos_weight=pos_weight)

    return pooled, per_clip_metrics, all_true, all_pred


def evaluate_test_clips(
    pooled: dict[str, float],
    per_clip: dict[str, dict[str, float]],
    *,
    pred_threshold: float,
    save_report: Path | None = None,
) -> dict[str, Any]:
    print(f"\n=== test set inference (threshold={pred_threshold}) ===")
    for clip_id in sorted(per_clip):
        print(format_metrics_row(clip_id, per_clip[clip_id]))
    print("\npooled:")
    print(format_metrics_row("ALL", pooled))

    report = {"per_clip": per_clip, "pooled": pooled}
    if save_report is not None:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {save_report}")
    return report


def train(cfg: TrainConfig) -> dict[str, Any]:
    _set_seed(42)
    os.makedirs(cfg.output_dir, exist_ok=True)
    dev = resolve_device(cfg.device)

    train_ds = RawFrameWindowDataset(
        [cfg.train_parquet],
        s3_cache_dir=cfg.s3_cache_dir,
        clip_cache_dir=cfg.clip_cache_dir,
        window_radius=cfg.window_radius,
        num_frames=cfg.num_frames,
        image_size=cfg.image_size,
        boundary_margin=cfg.boundary_margin,
        frame_stride=cfg.train_frame_stride,
    )
    test_ds = RawFrameWindowDataset(
        [cfg.test_parquet],
        s3_cache_dir=cfg.s3_cache_dir,
        clip_cache_dir=cfg.clip_cache_dir,
        window_radius=cfg.window_radius,
        num_frames=cfg.num_frames,
        image_size=cfg.image_size,
        boundary_margin=cfg.boundary_margin,
        frame_stride=1,
    )

    n_pos, n_neg = train_label_counts(
        train_ds.clip_ids(),
        train_ds.labels_by_clip,
        boundary_margin=cfg.boundary_margin,
        frame_stride=cfg.train_frame_stride,
    )
    tversky_alpha, tversky_beta = tversky_coefficients_from_counts(n_pos, n_neg)
    pos_weight = class_weight_ratio_from_counts(n_pos, n_neg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_raw_frame_windows,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_raw_frame_windows,
    )

    print(
        "dataset: "
        f"train_samples={len(train_ds)} test_samples={len(test_ds)} "
        f"batch_size={cfg.batch_size} num_workers={cfg.num_workers}"
    )

    encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    encoder.to(dev)
    if cfg.finetune_encoder:
        encoder.train()
    else:
        encoder.eval()

    model_cfg = TransformerConfig(
        input_dim=512,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_len=cfg.num_frames,
    )
    model = TransformerClassifier(model_cfg).to(dev)
    encoder_lr = cfg.encoder_lr if cfg.encoder_lr is not None else cfg.lr
    if cfg.finetune_encoder:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": cfg.lr},
                {"params": encoder.parameters(), "lr": encoder_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    config = {
        "train_parquet": cfg.train_parquet,
        "test_parquet": cfg.test_parquet,
        "s3_cache_dir": cfg.s3_cache_dir,
        "clip_cache_dir": cfg.clip_cache_dir,
        "num_frames": cfg.num_frames,
        "image_size": cfg.image_size,
        "window_radius": cfg.window_radius,
        "boundary_margin": cfg.boundary_margin,
        "train_frame_stride": cfg.train_frame_stride,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "pred_threshold": cfg.pred_threshold,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "pos_weight": pos_weight,
        "model_dim": cfg.model_dim,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "device": cfg.device,
        "finetune_encoder": cfg.finetune_encoder,
        "encoder_lr": cfg.encoder_lr,
        "wandb_project": cfg.wandb_project,
        "wandb_entity": cfg.wandb_entity,
        "wandb_run": cfg.wandb_run,
        "checkpoint_metric": cfg.checkpoint_metric,
        "early_stop_patience": cfg.early_stop_patience,
    }
    Path(cfg.output_dir, "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    wandb_logger = WandbLogger(
        WandbConfig(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run,
            enabled=True,
        ),
        config=config,
    )

    best_loss = float("inf")
    best_recall = -1.0
    best_cost = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    loss_history: list[dict[str, Any]] = []

    for epoch in tqdm(range(cfg.epochs), desc="epochs", dynamic_ncols=True, mininterval=1.0):
        model.train()
        train_loss_sum = 0.0
        train_mask_sum = 0.0
        train_iter = tqdm(
            train_loader,
            desc=f"train {epoch}",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            total=len(train_loader),
        )
        for batch_idx, batch in enumerate(train_iter):
            frames = batch["video"].to(dev)
            labels = batch["label"].to(dev).unsqueeze(1)
            loss_mask = batch["loss_mask"].to(dev).unsqueeze(1)

            feats = _encode_frames(frames, encoder, use_grad=cfg.finetune_encoder)
            logits = model(feats)
            pos_logits = logits[:, 1] - logits[:, 0]

            optimizer.zero_grad(set_to_none=True)
            if cfg.boundary_margin > 0:
                wsum, msum = masked_tversky_loss_sum(
                    pos_logits, labels, loss_mask, alpha=tversky_alpha, beta=tversky_beta
                )
                loss = wsum / msum.clamp(min=1.0)
                train_loss_sum += float(wsum.item())
                train_mask_sum += float(msum.item())
            else:
                loss = tversky_loss(pos_logits, labels, alpha=tversky_alpha, beta=tversky_beta)
                train_loss_sum += float(loss.item()) * labels.numel()
                train_mask_sum += float(labels.numel())
            loss.backward()
            optimizer.step()
            if cfg.log_every > 0 and batch_idx % cfg.log_every == 0:
                train_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / max(train_mask_sum, 1.0)
        pooled, per_clip, y_true, y_pred = evaluate_loader(
            model,
            encoder,
            test_loader,
            dev,
            pos_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            boundary_margin=cfg.boundary_margin,
            pred_threshold=cfg.pred_threshold,
        )
        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": float(pooled["loss"]),
                "recall": float(pooled["recall"]),
                "precision": float(pooled["precision"]),
                "f1": float(pooled["f1"]),
                "cost": float(pooled["cost"]),
            }
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"test_loss={pooled['loss']:.4f} recall={pooled['recall']:.4f} "
            f"precision={pooled['precision']:.4f} f1={pooled['f1']:.4f} cost={pooled['cost']:.0f}"
        )

        wandb_logger.log(
            {
                "train_loss": train_loss,
                "test_loss": float(pooled["loss"]),
                "test_precision": float(pooled["precision"]),
                "test_recall": float(pooled["recall"]),
                "test_f1": float(pooled["f1"]),
                "test_cost": float(pooled["cost"]),
                "epoch": epoch,
            },
            step=epoch,
        )
        wandb_logger.log_confusion_matrix(y_true, y_pred, labels=["downtime", "playtime"])

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": pooled,
            "config": config,
        }
        torch.save(ckpt, Path(cfg.output_dir) / "last.pt")

        improved = _is_better_checkpoint(
            cfg.checkpoint_metric,
            pooled,
            best_loss=best_loss,
            best_recall=best_recall,
            best_cost=best_cost,
        )
        if improved:
            best_loss = min(best_loss, pooled["loss"])
            best_recall = max(best_recall, pooled["recall"])
            best_cost = min(best_cost, pooled["cost"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(ckpt, Path(cfg.output_dir) / "best.pt")
            print(f"  saved best.pt ({_checkpoint_save_msg(cfg.checkpoint_metric, pooled)})")
        else:
            epochs_without_improvement += 1

        if cfg.early_stop_patience and epochs_without_improvement >= cfg.early_stop_patience:
            print(
                f"  early stop: no {cfg.checkpoint_metric} improvement for {cfg.early_stop_patience} epoch(s) "
                f"(best epoch {best_epoch})"
            )
            break

    loss_history_path = Path(cfg.output_dir) / "loss_history.json"
    loss_curve_path = Path(cfg.output_dir) / "loss_curve.png"
    save_loss_history(loss_history, loss_history_path)
    plot_loss_curve(loss_history, loss_curve_path, best_epoch=best_epoch)
    print(f"  wrote {loss_history_path.name} and {loss_curve_path.name}")

    best_ckpt = torch.load(Path(cfg.output_dir) / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    pooled, per_clip, y_true, y_pred = evaluate_loader(
        model,
        encoder,
        test_loader,
        dev,
        pos_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        boundary_margin=cfg.boundary_margin,
        pred_threshold=cfg.pred_threshold,
    )
    evaluate_test_clips(
        pooled,
        per_clip,
        pred_threshold=cfg.pred_threshold,
        save_report=Path(cfg.output_dir) / "test_clip_metrics.json",
    )
    wandb_logger.log_confusion_matrix(y_true, y_pred, labels=["downtime", "playtime"])
    per_clip_rows = [
        [clip_id, metrics["recall"], metrics["precision"], metrics["f1"], metrics["accuracy"], metrics["cost"]]
        for clip_id, metrics in sorted(per_clip.items())
    ]
    wandb_logger.log_table(
        "test/per_clip_metrics",
        ["clip_id", "recall", "precision", "f1", "accuracy", "cost"],
        per_clip_rows,
    )
    wandb_logger.finish()

    return {
        "best_epoch": int(best_ckpt["epoch"]),
        "metrics": dict(best_ckpt["metrics"]),
        "checkpoint_metric": cfg.checkpoint_metric,
        "checkpoint_dir": str(cfg.output_dir),
        "loss_history_path": str(loss_history_path),
        "loss_curve_path": str(loss_curve_path),
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train transformer on raw frames from parquet splits.")
    p.add_argument("--train-parquet", required=True, help="S3 prefix or local path with train parquet files")
    p.add_argument("--test-parquet", required=True, help="S3 prefix or local path with test parquet files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--s3-cache-dir", default=str(REPO_ROOT / "data" / "s3_cache"))
    p.add_argument("--clip-cache-dir", default=str(REPO_ROOT / "data" / "s3_cache" / "clips"))
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--window-radius", type=int, default=15)
    p.add_argument("--boundary-margin", type=int, default=0)
    p.add_argument("--train-frame-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pred-threshold", type=float, default=0.35)
    p.add_argument("--checkpoint-metric", choices=("loss", "recall", "cost"), default="loss")
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    p.add_argument("--finetune-encoder", action="store_true", help="Allow ResNet encoder weights to update")
    p.add_argument("--encoder-lr", type=float, default=None, help="Optional LR for encoder params")
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=20, help="Batch interval for progress updates")
    p.add_argument("--wandb-project", default="volleyball-playtime")
    p.add_argument("--wandb-entity", default="cs348k-sports-footage-autotrim")
    p.add_argument("--wandb-run", default=None)
    args = p.parse_args()
    return TrainConfig(
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        output_dir=args.output_dir,
        s3_cache_dir=args.s3_cache_dir,
        clip_cache_dir=args.clip_cache_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        window_radius=args.window_radius,
        boundary_margin=args.boundary_margin,
        train_frame_stride=args.train_frame_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pred_threshold=args.pred_threshold,
        checkpoint_metric=args.checkpoint_metric,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=args.device,
        finetune_encoder=args.finetune_encoder,
        encoder_lr=args.encoder_lr,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run=args.wandb_run,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
