#!/usr/bin/env python3
"""Train BiLSTM on cached features; print per-clip test metrics when done.

Prerequisites::

    python data/preprocess_labels.py
    python models/lstm/extract_features.py
    python models/lstm/train.py

Re-run test evaluation only::

    python models/lstm/train.py --eval-only --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import (  # noqa: E402
    DEFAULT_BOUNDARY_MARGIN,
    DEFAULT_FRAME_STRIDE,
    FeatureWindowDataset,
    WINDOW_RADIUS,
    class_weight_ratio_from_counts,
    tversky_coefficients_from_counts,
    clip_to_source_map,
    list_clip_ids,
    load_labels_by_clip,
    split_clip_ids_by_source,
    split_clip_ids_stratified_by_source,
    train_label_counts,
)
from models.lstm.encoders import default_backbone, get_encoder, resolve_device  # noqa: E402
from models.lstm.extract_features import ensure_features_for_clips  # noqa: E402
from models.lstm.model import TemporalPlayingClassifier  # noqa: E402

FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"
CHECKPOINT_DIR = REPO_ROOT / "models" / "lstm" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

BACKBONE = default_backbone()
# Default: ~10% of each video's clips in test (stratified by source_id).
TEST_SIZE = 0.1
DEFAULT_SPLIT_MODE = "stratified_by_source"
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4  # AdamW L2; set 0 to disable
NUM_WORKERS = 0
# Playing vs inactive at inference; lower => higher recall (default 0.35).
DEFAULT_PRED_THRESHOLD = 0.35

# Temporal head (smaller + dropout to reduce overfitting)
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.0  # only used when LSTM_NUM_LAYERS > 1 (between LSTM layers)
HEAD_DROPOUT = 0.3  # on BiLSTM output before the linear head (active in train mode)
DEFAULT_CHECKPOINT_METRIC = "loss"  # loss | recall | cost
TVERSKY_SMOOTH = 1e-6


def temporal_model_from_config(feat_dim: int, config: dict) -> TemporalPlayingClassifier:
    """Build the head with sizes from ``config``; defaults match pre-refactor checkpoints."""
    return TemporalPlayingClassifier(
        feat_dim,
        hidden_size=int(config.get("lstm_hidden_size", 128)),
        num_layers=int(config.get("lstm_num_layers", 1)),
        dropout=float(config.get("lstm_dropout", 0.0)),
        head_dropout=float(config.get("head_dropout", 0.0)),
    )


def pos_weight_from_config(config: dict) -> float:
    """Cost FN weight / Tversky beta from checkpoint config (legacy keys supported)."""
    if "tversky_beta" in config:
        return float(config["tversky_beta"])
    if "pos_weight" in config:
        return float(config["pos_weight"])
    if "fn_fp_weight_ratio" in config:
        return float(config["fn_fp_weight_ratio"])
    if "pos_weight_positive" in config:
        return float(config["pos_weight_positive"])
    fn_c, fp_c = config.get("fn_cost"), config.get("fp_cost")
    if fn_c is not None and fp_c is not None:
        fp = float(fp_c)
        if fp > 0:
            return float(fn_c) / fp
    return 1.0


def tversky_coefficients_from_config(config: dict) -> tuple[float, float]:
    if "tversky_alpha" in config and "tversky_beta" in config:
        alpha = float(config["tversky_alpha"])
        beta = float(config["tversky_beta"])
    else:
        ratio = pos_weight_from_config(config)
        alpha, beta = 1.0, ratio
    total = alpha + beta
    if total <= 0:
        return 0.5, 0.5
    return alpha / total, beta / total


def pred_threshold_from_config(config: dict) -> float:
    if "pred_threshold" in config:
        return float(config["pred_threshold"])
    return DEFAULT_PRED_THRESHOLD


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    smooth: float = TVERSKY_SMOOTH,
) -> torch.Tensor:
    """``1 - Tversky`` on sigmoid probabilities (batch-level scalar)."""
    probs = torch.sigmoid(logits)
    t = targets.to(dtype=probs.dtype)
    tp = (probs * t).sum()
    fp = (probs * (1.0 - t)).sum()
    fn = ((1.0 - probs) * t).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


def masked_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    smooth: float = TVERSKY_SMOOTH,
) -> torch.Tensor:
    """Tversky loss over samples with ``mask > 0`` (boundary frames excluded)."""
    weighted, denom = masked_tversky_loss_sum(
        logits, targets, mask, alpha=alpha, beta=beta, smooth=smooth
    )
    if denom <= 0:
        return weighted.sum() * 0.0
    return weighted / denom


def masked_tversky_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    smooth: float = TVERSKY_SMOOTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss * mask_sum, mask_sum)`` for global epoch/loader averaging."""
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


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pos_weight: float,
) -> dict[str, float]:
    """Thresholded frame metrics; ``cost = pos_weight * FN + FP`` (matches Tversky beta on FN)."""
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


def load_feature_meta(backbone: str) -> dict:
    meta_path = FEATURES_ROOT / backbone / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(
            f"missing {meta_path}; run: python models/lstm/extract_features.py"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_mode_from_config(config: dict) -> str:
    if mode := config.get("split_mode"):
        return str(mode)
    if config.get("random_source_split"):
        return "random_by_source"
    if config.get("fixed_test_sources"):
        return "hold_out_sources"
    return DEFAULT_SPLIT_MODE


def _test_size_from_config(config: dict) -> float:
    return float(config.get("test_size", config.get("test_size_sources", TEST_SIZE)))


def split_train_test_clips(
    clip_ids: list[str],
    source_map: dict[str, str],
    *,
    split_mode: str = DEFAULT_SPLIT_MODE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    hold_out_sources: tuple[str, ...] = (),
) -> tuple[list[str], list[str], dict]:
    if split_mode == "stratified_by_source":
        return split_clip_ids_stratified_by_source(
            clip_ids,
            source_map,
            test_size=test_size,
            random_state=random_state,
        )
    if split_mode == "hold_out_sources":
        if not hold_out_sources:
            raise RuntimeError("hold_out_sources split requires at least one source_id")
        return split_clip_ids_by_source(
            clip_ids,
            source_map,
            fixed_test_sources=hold_out_sources,
            test_size=test_size,
            random_state=random_state,
        )
    if split_mode == "random_by_source":
        return split_clip_ids_by_source(
            clip_ids,
            source_map,
            fixed_test_sources=None,
            test_size=test_size,
            random_state=random_state,
        )
    raise ValueError(f"unknown split_mode: {split_mode!r}")


def resolved_test_clip_ids_from_config(config: dict) -> list[str]:
    """Rebuild test clip ids when a checkpoint omits ``test_clip_ids`` but stored split settings."""
    explicit = config.get("test_clip_ids")
    if explicit:
        return list(explicit)
    sm = clip_to_source_map(FRAME_LABELS_CSV)
    all_ids = list_clip_ids(FRAME_LABELS_CSV)
    hold_out = tuple(str(x) for x in (config.get("hold_out_sources") or config.get("fixed_test_sources") or ()))
    _, test_ids, _ = split_train_test_clips(
        all_ids,
        sm,
        split_mode=_split_mode_from_config(config),
        test_size=_test_size_from_config(config),
        random_state=int(config.get("random_seed", RANDOM_SEED)),
        hold_out_sources=hold_out,
    )
    return test_ids


def collate_batch(batch: list[dict]) -> dict:
    return {
        "seq": torch.stack([b["seq"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch], dim=0),
        "clip_id": [b["clip_id"] for b in batch],
        "frame_idx": torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long),
    }


def evaluate_loader(
    model: TemporalPlayingClassifier,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    *,
    tversky_alpha: float,
    tversky_beta: float,
    boundary_margin: int = 0,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mask_sum = 0.0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(seq)
            if boundary_margin > 0:
                loss_mask = batch["loss_mask"].to(device).unsqueeze(1)
                wsum, msum = masked_tversky_loss_sum(
                    logits,
                    labels,
                    loss_mask,
                    alpha=tversky_alpha,
                    beta=tversky_beta,
                )
                loss_sum += float(wsum.item())
                mask_sum += float(msum.item())
            else:
                loss = tversky_loss(
                    logits, labels, alpha=tversky_alpha, beta=tversky_beta
                )
                n = float(labels.numel())
                loss_sum += float(loss.item()) * n
                mask_sum += n
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs >= pred_threshold).astype(np.int64)
            y_true.append(labels.long().cpu().numpy().ravel())
            y_pred.append(preds)

    metrics = binary_metrics(
        np.concatenate(y_true), np.concatenate(y_pred), pos_weight=pos_weight
    )
    metrics["loss"] = loss_sum / max(mask_sum, 1.0)
    return metrics


def predict_clip(
    model: TemporalPlayingClassifier,
    clip_id: str,
    *,
    backbone: str,
    labels_by_clip: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    ds = FeatureWindowDataset([clip_id], backbone=backbone, labels_by_clip=labels_by_clip)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )

    frame_indices: list[int] = []
    y_true: list[float] = []
    y_prob: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["seq"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            frame_indices.extend(batch["frame_idx"].tolist())
            y_true.extend(batch["label"].cpu().numpy().ravel().tolist())
            y_prob.extend(probs.tolist())

    order = np.argsort(frame_indices)
    yt = np.asarray(y_true, dtype=np.int64)[order]
    yp = (np.asarray(y_prob, dtype=np.float64)[order] >= pred_threshold).astype(np.int64)
    return yt, yp


def format_metrics_row(clip_id: str, m: dict[str, float]) -> str:
    return (
        f"{clip_id:20s}  "
        f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  f1={m['f1']:.3f}  "
        f"acc={m['accuracy']:.3f}  cost={m['cost']:.0f}  "
        f"TP={int(m['tp']):4d} FP={int(m['fp']):4d} TN={int(m['tn']):4d} FN={int(m['fn']):4d}  "
        f"n={int(m['n_frames'])}"
    )


def evaluate_test_clips(
    model: TemporalPlayingClassifier,
    test_ids: list[str],
    *,
    backbone: str,
    labels_by_clip: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    pos_weight: float,
    pred_threshold: float = DEFAULT_PRED_THRESHOLD,
    save_report: Path | None = CHECKPOINT_DIR / "test_clip_metrics.json",
) -> dict:
    """Per-clip and pooled frame metrics on held-out clips."""
    per_clip: dict[str, dict] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    print(f"\n=== test set inference (threshold={pred_threshold}) ===")
    for clip_id in test_ids:
        yt, yp = predict_clip(
            model,
            clip_id,
            backbone=backbone,
            labels_by_clip=labels_by_clip,
            device=device,
            batch_size=batch_size,
            pred_threshold=pred_threshold,
        )
        m = binary_metrics(yt, yp, pos_weight=pos_weight)
        per_clip[clip_id] = m
        all_true.append(yt)
        all_pred.append(yp)
        print(format_metrics_row(clip_id, m))

    pooled = binary_metrics(
        np.concatenate(all_true), np.concatenate(all_pred), pos_weight=pos_weight
    )
    print("\npooled:")
    print(format_metrics_row("ALL", pooled))

    report = {"test_clip_ids": test_ids, "per_clip": per_clip, "pooled": pooled}
    if save_report is not None:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {save_report}")
    return report


def _is_better_checkpoint(
    metric: str,
    metrics: dict[str, float],
    *,
    best_loss: float,
    best_recall: float,
    best_cost: float,
) -> bool:
    if metric == "loss":
        return metrics["loss"] < best_loss
    if metric == "cost":
        return metrics["cost"] < best_cost or (
            metrics["cost"] == best_cost and metrics["recall"] > best_recall
        )
    # recall (default): highest recall, then lowest cost
    return metrics["recall"] > best_recall or (
        metrics["recall"] == best_recall and metrics["cost"] < best_cost
    )


def _checkpoint_save_msg(metric: str, metrics: dict[str, float]) -> str:
    if metric == "loss":
        return f"test_loss={metrics['loss']:.4f}"
    if metric == "cost":
        return f"cost={metrics['cost']:.0f} recall={metrics['recall']:.4f}"
    return f"recall={metrics['recall']:.4f} cost={metrics['cost']:.0f}"


def train(
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    split_mode: str = DEFAULT_SPLIT_MODE,
    test_size: float | None = None,
    hold_out_sources: str = "",
    lr: float | None = None,
    weight_decay: float | None = None,
    pred_threshold: float | None = None,
    head_dropout: float | None = None,
    checkpoint_metric: str = DEFAULT_CHECKPOINT_METRIC,
    checkpoint_dir: Path | None = None,
    boundary_margin: int | None = None,
    train_frame_stride: int | None = None,
    early_stop_patience: int | None = None,
    quiet: bool = False,
    skip_final_eval: bool = False,
) -> dict[str, Any]:
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lr_val = LR if lr is None else lr
    wd = WEIGHT_DECAY if weight_decay is None else weight_decay
    pred_thr = DEFAULT_PRED_THRESHOLD if pred_threshold is None else pred_threshold
    hd = HEAD_DROPOUT if head_dropout is None else head_dropout
    bmargin = DEFAULT_BOUNDARY_MARGIN if boundary_margin is None else boundary_margin
    frame_stride = DEFAULT_FRAME_STRIDE if train_frame_stride is None else train_frame_stride
    es_patience = early_stop_patience
    ckpt_dir = CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    if checkpoint_metric not in ("loss", "recall", "cost"):
        raise ValueError(f"checkpoint_metric must be loss, recall, or cost; got {checkpoint_metric!r}")
    dev = resolve_device(device) if device is not None else resolve_device()
    print(
        f"device={dev} backbone={BACKBONE} epochs={epochs} batch_size={batch_size} "
        f"lr={lr_val} weight_decay={wd} head_dropout={hd} pred_threshold={pred_thr} "
        f"boundary_margin={bmargin} train_frame_stride={frame_stride} "
        f"early_stop_patience={es_patience} checkpoint_metric={checkpoint_metric} "
        f"checkpoint_dir={ckpt_dir}"
    )

    meta = load_feature_meta(BACKBONE)
    if meta.get("backbone") != BACKBONE:
        raise RuntimeError(f"meta backbone {meta.get('backbone')!r} != config {BACKBONE!r}")
    feat_dim = int(meta["feat_dim"])
    cached_img_size = int(meta.get("img_size", 0))
    expected_img_size = get_encoder(BACKBONE, device="cpu").img_size
    if cached_img_size and cached_img_size != expected_img_size:
        raise RuntimeError(
            f"feature cache img_size={cached_img_size} but {BACKBONE} expects {expected_img_size}; "
            "re-run: python models/lstm/extract_features.py --force --device mps"
        )

    labels_by_clip = load_labels_by_clip(FRAME_LABELS_CSV)
    all_clip_ids = list_clip_ids(FRAME_LABELS_CSV)
    ts_frac = TEST_SIZE if test_size is None else test_size
    hold_out = tuple(s.strip() for s in hold_out_sources.split(",") if s.strip())
    smap = clip_to_source_map(FRAME_LABELS_CSV)
    train_ids, test_ids, split_info = split_train_test_clips(
        all_clip_ids,
        smap,
        split_mode=split_mode,
        test_size=ts_frac,
        random_state=RANDOM_SEED,
        hold_out_sources=hold_out,
    )
    print(
        f"split={split_info['mode']}  test_size={ts_frac}  "
        f"clips train={len(train_ids)} test={len(test_ids)}"
    )
    if split_info["mode"] == "stratified_by_source":
        per_src = split_info["per_source"]
        n_with_test = sum(1 for v in per_src.values() if v["n_test"] > 0)
        print(f"  sources={split_info['n_sources']}  sources_with_test_clips={n_with_test}")
    else:
        print(
            f"  sources train={split_info['n_sources_train']} "
            f"test={split_info['n_sources_test']}"
        )
        print(f"  test source_id(s): {split_info['test_sources']}")

    train_ds = FeatureWindowDataset(
        train_ids,
        backbone=BACKBONE,
        labels_by_clip=labels_by_clip,
        boundary_margin=bmargin,
        frame_stride=frame_stride,
    )
    test_ds = FeatureWindowDataset(
        test_ids,
        backbone=BACKBONE,
        labels_by_clip=labels_by_clip,
        boundary_margin=bmargin,
        frame_stride=1,
    )
    if train_ds.feat_dim != feat_dim:
        raise RuntimeError(f"feat_dim mismatch: cache={feat_dim} dataset={train_ds.feat_dim}")

    n_pos, n_neg = train_label_counts(
        train_ids, labels_by_clip, boundary_margin=bmargin, frame_stride=frame_stride
    )
    tversky_alpha, tversky_beta = tversky_coefficients_from_counts(n_pos, n_neg)
    pos_weight = class_weight_ratio_from_counts(n_pos, n_neg)
    print(
        f"  class weights: n_pos={n_pos} n_neg={n_neg} "
        f"tversky_alpha={tversky_alpha:.4f} tversky_beta={tversky_beta:.4f} "
        f"(sum={tversky_alpha + tversky_beta:.4f}) pos_weight={pos_weight:.4f} (cost FN)"
    )

    print(f"  train samples={len(train_ds)} test samples={len(test_ds)}")
    if frame_stride > 1:
        print(f"  train subsampling: every {frame_stride}th frame")
    if bmargin > 0:
        n_loss, n_train_frames = train_ds.loss_frame_counts()
        n_test_loss, n_test_frames = test_ds.loss_frame_counts()
        train_pct = 100.0 * (1.0 - n_loss / max(n_train_frames, 1))
        test_pct = 100.0 * (1.0 - n_test_loss / max(n_test_frames, 1))
        print(
            f"  boundary loss mask: train {n_loss}/{n_train_frames} "
            f"({train_pct:.1f}% ignored), test {n_test_loss}/{n_test_frames} "
            f"({test_pct:.1f}% ignored) within {bmargin} frames of transitions"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
    )

    model = TemporalPlayingClassifier(
        feat_dim,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        head_dropout=hd,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=wd)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "backbone": BACKBONE,
        "feat_dim": feat_dim,
        "features_root": str(FEATURES_ROOT.relative_to(REPO_ROOT)),
        "frame_labels_csv": str(FRAME_LABELS_CSV.relative_to(REPO_ROOT)),
        "train_clip_ids": train_ids,
        "test_clip_ids": test_ids,
        "split": split_info,
        "split_mode": split_mode,
        "test_size": ts_frac,
        "hold_out_sources": list(hold_out) if hold_out else None,
        "random_seed": RANDOM_SEED,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr_val,
        "weight_decay": wd,
        "loss": "tversky",
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "pos_weight": pos_weight,
        "train_n_pos": n_pos,
        "train_n_neg": n_neg,
        "pred_threshold": pred_thr,
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "lstm_num_layers": LSTM_NUM_LAYERS,
        "lstm_dropout": LSTM_DROPOUT,
        "head_dropout": hd,
        "checkpoint_metric": checkpoint_metric,
        "boundary_margin": bmargin,
        "train_frame_stride": frame_stride,
        "early_stop_patience": es_patience,
    }
    (ckpt_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_loss = float("inf")
    best_recall = -1.0
    best_cost = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    epoch_iter = tqdm(range(epochs), desc="epochs", disable=quiet)
    for epoch in epoch_iter:
        model.train()
        train_loss_sum = 0.0
        train_mask_sum = 0.0

        batch_iter = tqdm(train_loader, desc=f"train {epoch}", leave=False, disable=quiet)
        for batch in batch_iter:
            seq = batch["seq"].to(dev)
            labels = batch["label"].to(dev).unsqueeze(1)
            loss_mask = batch["loss_mask"].to(dev).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq)
            if bmargin > 0:
                wsum, msum = masked_tversky_loss_sum(
                    logits,
                    labels,
                    loss_mask,
                    alpha=tversky_alpha,
                    beta=tversky_beta,
                )
                loss = wsum / msum.clamp(min=1.0)
                train_loss_sum += float(wsum.item())
                train_mask_sum += float(msum.item())
            else:
                loss = tversky_loss(
                    logits, labels, alpha=tversky_alpha, beta=tversky_beta
                )
                train_loss_sum += float(loss.item()) * labels.numel()
                train_mask_sum += float(labels.numel())
            loss.backward()
            optimizer.step()

        train_loss = train_loss_sum / max(train_mask_sum, 1.0)
        metrics = evaluate_loader(
            model,
            test_loader,
            dev,
            pos_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            boundary_margin=bmargin,
            pred_threshold=pred_thr,
        )
        if not quiet:
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"test_loss={metrics['loss']:.4f} "
                f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
                f"f1={metrics['f1']:.4f} cost={metrics['cost']:.0f} "
                f"(TP={metrics['tp']:.0f} FP={metrics['fp']:.0f} "
                f"TN={metrics['tn']:.0f} FN={metrics['fn']:.0f})"
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        improved = _is_better_checkpoint(
            checkpoint_metric,
            metrics,
            best_loss=best_loss,
            best_recall=best_recall,
            best_cost=best_cost,
        )
        if improved:
            best_loss = min(best_loss, metrics["loss"])
            best_recall = max(best_recall, metrics["recall"])
            best_cost = min(best_cost, metrics["cost"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            if not quiet:
                print(f"  saved best.pt ({_checkpoint_save_msg(checkpoint_metric, metrics)})")
        else:
            epochs_without_improvement += 1

        if es_patience is not None and es_patience > 0 and epochs_without_improvement >= es_patience:
            if not quiet:
                print(
                    f"  early stop: no {checkpoint_metric} improvement for {es_patience} epoch(s) "
                    f"(best epoch {best_epoch})"
                )
            break

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    if not skip_final_eval:
        evaluate_test_clips(
            model,
            test_ids,
            backbone=BACKBONE,
            labels_by_clip=labels_by_clip,
            device=dev,
            batch_size=batch_size,
            pos_weight=pos_weight,
            pred_threshold=pred_thr,
            save_report=ckpt_dir / "test_clip_metrics.json",
        )
    if not quiet:
        print(f"\ndone. checkpoints in {ckpt_dir}")

    return {
        "best_epoch": int(best_ckpt["epoch"]),
        "metrics": dict(best_ckpt["metrics"]),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_dir": str(ckpt_dir),
        "hparams": {
            "lr": lr_val,
            "weight_decay": wd,
            "pos_weight": pos_weight,
            "pred_threshold": pred_thr,
            "head_dropout": hd,
            "boundary_margin": bmargin,
            "train_frame_stride": frame_stride,
            "early_stop_patience": es_patience,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }


def eval_checkpoint(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    device: str | None = None,
    batch_size: int = 64,
    extract_batch_size: int = 32,
    force_extract: bool = False,
    pred_threshold: float | None = None,
) -> None:
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")

    dev = resolve_device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    config = ckpt.get("config") or {}
    feat_dim = int(config.get("feat_dim", 0))
    if feat_dim <= 0:
        raise RuntimeError(f"checkpoint missing feat_dim: {checkpoint}")

    backbone = str(config.get("backbone", BACKBONE))
    tversky_alpha, tversky_beta = tversky_coefficients_from_config(config)
    pos_weight = pos_weight_from_config(config)
    pred_thr = (
        pred_threshold_from_config(config)
        if pred_threshold is None
        else pred_threshold
    )
    test_ids = resolved_test_clip_ids_from_config(config)
    print(
        f"checkpoint={checkpoint.name} device={dev} test clips={len(test_ids)} "
        f"tversky_alpha={tversky_alpha:.4f} tversky_beta={tversky_beta:.4f} "
        f"pred_threshold={pred_thr}"
    )

    ensure_features_for_clips(
        test_ids,
        backbone=backbone,
        clip_to_source=clip_to_source_map(),
        batch_size=extract_batch_size,
        force=force_extract,
        device=str(dev),
        show_frames=force_extract,
    )

    model = temporal_model_from_config(feat_dim, config).to(dev)
    model.load_state_dict(ckpt["model_state"])
    evaluate_test_clips(
        model,
        test_ids,
        backbone=backbone,
        labels_by_clip=load_labels_by_clip(FRAME_LABELS_CSV),
        device=dev,
        batch_size=batch_size,
        pos_weight=pos_weight,
        pred_threshold=pred_thr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BiLSTM; evaluate test clips when done.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR, help=f"AdamW learning rate (default: {LR})")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {WEIGHT_DECAY}; use 0 to disable)",
    )
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; run per-clip test metrics from a checkpoint",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--force-extract", action="store_true", help="Re-run CNN for test clips")
    p.add_argument(
        "--split-mode",
        choices=("stratified_by_source", "random_by_source", "hold_out_sources"),
        default=DEFAULT_SPLIT_MODE,
        help="stratified_by_source: ~test_size of each video's clips in test (default); "
        "random_by_source: whole videos in train or test; "
        "hold_out_sources: all clips from --hold-out-sources in test.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        metavar="FRAC",
        help=f"Fraction of clips in test per video (stratified, default {TEST_SIZE}) "
        "or fraction of videos (random_by_source).",
    )
    p.add_argument(
        "--hold-out-sources",
        default="",
        metavar="IDS",
        help="Comma-separated source_id values; required when --split-mode=hold_out_sources.",
    )
    p.add_argument(
        "--pred-threshold",
        type=float,
        default=DEFAULT_PRED_THRESHOLD,
        metavar="T",
        help=f"Sigmoid threshold for playing (default: {DEFAULT_PRED_THRESHOLD}; lower => higher recall).",
    )
    p.add_argument(
        "--head-dropout",
        type=float,
        default=None,
        metavar="P",
        help=f"Dropout on BiLSTM center frame before linear head (default: {HEAD_DROPOUT}).",
    )
    p.add_argument(
        "--checkpoint-metric",
        choices=("loss", "recall", "cost"),
        default=DEFAULT_CHECKPOINT_METRIC,
        help="Save best.pt when this test metric improves (default: loss).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=f"Directory for best.pt / last.pt (default: {CHECKPOINT_DIR.relative_to(REPO_ROOT)}).",
    )
    p.add_argument(
        "--boundary-margin",
        type=int,
        default=DEFAULT_BOUNDARY_MARGIN,
        metavar="N",
        help="Exclude frames within N indices of each 0/1 label transition from train and test "
        f"loss (default: {DEFAULT_BOUNDARY_MARGIN}; try {WINDOW_RADIUS} to match the 30-frame "
        "window). Thresholded recall/precision/F1 still use all frames.",
    )
    p.add_argument(
        "--train-frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        metavar="N",
        help="Use every Nth frame for training samples only (default: 1). Reduces overlapping "
        "windows from the same clip.",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N epochs without improvement on --checkpoint-metric (default: off). "
        "Use 2 with --checkpoint-metric loss when test loss rises after epoch 0.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.eval_only:
            eval_checkpoint(
                args.checkpoint,
                device=args.device,
                batch_size=args.batch_size,
                force_extract=args.force_extract,
                pred_threshold=args.pred_threshold,
            )
        else:
            train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                split_mode=args.split_mode,
                test_size=args.test_size,
                hold_out_sources=args.hold_out_sources,
                lr=args.lr,
                weight_decay=args.weight_decay,
                pred_threshold=args.pred_threshold,
                head_dropout=args.head_dropout,
                checkpoint_metric=args.checkpoint_metric,
                checkpoint_dir=args.checkpoint_dir,
                boundary_margin=args.boundary_margin,
                train_frame_stride=args.train_frame_stride,
                early_stop_patience=args.early_stop_patience,
            )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
