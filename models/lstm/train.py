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

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.dataset import (  # noqa: E402
    FeatureWindowDataset,
    clip_to_source_map,
    list_clip_ids,
    load_labels_by_clip,
)
from models.lstm.encoders import default_backbone, get_encoder, resolve_device  # noqa: E402
from models.lstm.extract_features import ensure_features_for_clips  # noqa: E402
from models.lstm.model import TemporalPlayingClassifier  # noqa: E402

FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"
CHECKPOINT_DIR = REPO_ROOT / "models" / "lstm" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

BACKBONE = default_backbone()
TEST_SIZE = 0.3
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
RECALL_BOOST = 4.0
FN_COST = 5.0
FP_COST = 1.0
NUM_WORKERS = 0


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict[str, float]:
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
        "cost": fn_cost * fn + fp_cost * fp,
        "n_frames": float(n),
    }


def load_feature_meta(backbone: str) -> dict:
    meta_path = FEATURES_ROOT / backbone / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(
            f"missing {meta_path}; run: python models/lstm/extract_features.py"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def split_clip_ids(clip_ids: list[str]) -> tuple[list[str], list[str]]:
    if len(clip_ids) < 2:
        raise RuntimeError(f"need at least 2 clips, found {len(clip_ids)}")
    train_ids, test_ids = train_test_split(
        clip_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    return sorted(train_ids), sorted(test_ids)


def collate_batch(batch: list[dict]) -> dict:
    return {
        "seq": torch.stack([b["seq"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "clip_id": [b["clip_id"] for b in batch],
        "frame_idx": torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long),
    }


def compute_pos_weight(labels_by_clip: dict, train_clip_ids: list[str]) -> torch.Tensor:
    y = np.concatenate([labels_by_clip[c] for c in train_clip_ids])
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - pos), 1)
    return torch.tensor([(neg / pos) * RECALL_BOOST], dtype=torch.float32)


def evaluate_loader(
    model: TemporalPlayingClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(seq)
            total_loss += float(criterion(logits, labels).item())
            n_batches += 1
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().ravel()
            y_true.append(labels.long().cpu().numpy().ravel())
            y_pred.append(preds)

    metrics = binary_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def predict_clip(
    model: TemporalPlayingClassifier,
    clip_id: str,
    *,
    backbone: str,
    labels_by_clip: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
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
    yp = (np.asarray(y_prob, dtype=np.float64)[order] >= 0.5).astype(np.int64)
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
    save_report: Path | None = CHECKPOINT_DIR / "test_clip_metrics.json",
) -> dict:
    """Per-clip and pooled frame metrics on held-out clips."""
    per_clip: dict[str, dict] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    print("\n=== test set inference (threshold=0.5) ===")
    for clip_id in test_ids:
        yt, yp = predict_clip(
            model,
            clip_id,
            backbone=backbone,
            labels_by_clip=labels_by_clip,
            device=device,
            batch_size=batch_size,
        )
        m = binary_metrics(yt, yp)
        per_clip[clip_id] = m
        all_true.append(yt)
        all_pred.append(yp)
        print(format_metrics_row(clip_id, m))

    pooled = binary_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    print("\npooled:")
    print(format_metrics_row("ALL", pooled))

    report = {"test_clip_ids": test_ids, "per_clip": per_clip, "pooled": pooled}
    if save_report is not None:
        save_report.parent.mkdir(parents=True, exist_ok=True)
        save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {save_report}")
    return report


def train(
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> None:
    epochs = EPOCHS if epochs is None else epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    dev = resolve_device(device) if device is not None else resolve_device()
    print(f"device={dev} backbone={BACKBONE} epochs={epochs} batch_size={batch_size}")

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
    train_ids, test_ids = split_clip_ids(all_clip_ids)
    print(f"clips: train={len(train_ids)} test={len(test_ids)}")

    train_ds = FeatureWindowDataset(train_ids, backbone=BACKBONE, labels_by_clip=labels_by_clip)
    test_ds = FeatureWindowDataset(test_ids, backbone=BACKBONE, labels_by_clip=labels_by_clip)
    if train_ds.feat_dim != feat_dim:
        raise RuntimeError(f"feat_dim mismatch: cache={feat_dim} dataset={train_ds.feat_dim}")

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

    model = TemporalPlayingClassifier(feat_dim).to(dev)
    pos_weight = compute_pos_weight(labels_by_clip, train_ids).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "backbone": BACKBONE,
        "feat_dim": feat_dim,
        "features_root": str(FEATURES_ROOT.relative_to(REPO_ROOT)),
        "frame_labels_csv": str(FRAME_LABELS_CSV.relative_to(REPO_ROOT)),
        "train_clip_ids": train_ids,
        "test_clip_ids": test_ids,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": LR,
        "recall_boost": RECALL_BOOST,
    }
    (CHECKPOINT_DIR / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_recall = -1.0
    best_cost = float("inf")

    for epoch in tqdm(range(epochs), desc="epochs"):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            seq = batch["seq"].to(dev)
            labels = batch["label"].to(dev).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        metrics = evaluate_loader(model, test_loader, dev)
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
        torch.save(ckpt, CHECKPOINT_DIR / "last.pt")

        if metrics["recall"] > best_recall or (
            metrics["recall"] == best_recall and metrics["cost"] < best_cost
        ):
            best_recall = metrics["recall"]
            best_cost = metrics["cost"]
            torch.save(ckpt, CHECKPOINT_DIR / "best.pt")
            print(f"  saved best.pt (recall={best_recall:.4f} cost={best_cost:.0f})")

    best_ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    evaluate_test_clips(
        model,
        test_ids,
        backbone=BACKBONE,
        labels_by_clip=labels_by_clip,
        device=dev,
        batch_size=batch_size,
    )
    print(f"\ndone. checkpoints in {CHECKPOINT_DIR}")


def eval_checkpoint(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    *,
    device: str | None = None,
    batch_size: int = 64,
    extract_batch_size: int = 32,
    force_extract: bool = False,
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
    test_ids = list(config.get("test_clip_ids") or split_clip_ids(list_clip_ids(FRAME_LABELS_CSV))[1])
    print(f"checkpoint={checkpoint.name} device={dev} test clips={len(test_ids)}")

    ensure_features_for_clips(
        test_ids,
        backbone=backbone,
        clip_to_source=clip_to_source_map(),
        batch_size=extract_batch_size,
        force=force_extract,
        device=str(dev),
        show_frames=force_extract,
    )

    model = TemporalPlayingClassifier(feat_dim).to(dev)
    model.load_state_dict(ckpt["model_state"])
    evaluate_test_clips(
        model,
        test_ids,
        backbone=backbone,
        labels_by_clip=load_labels_by_clip(FRAME_LABELS_CSV),
        device=dev,
        batch_size=batch_size,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BiLSTM; evaluate test clips when done.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; run per-clip test metrics from a checkpoint",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--force-extract", action="store_true", help="Re-run CNN for test clips")
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
            )
        else:
            train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
