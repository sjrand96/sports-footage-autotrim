#!/usr/bin/env python3
"""Train BiLSTM playing classifier on cached EfficientNetV2 features.

Prerequisites::

    python data/preprocess_labels.py
    python models/lstm/extract_features.py
    python models/lstm/train.py
"""

from __future__ import annotations

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
    list_clip_ids,
    load_labels_by_clip,
)
from models.lstm.encoders import default_backbone, resolve_device  # noqa: E402
from models.lstm.model import TemporalPlayingClassifier  # noqa: E402

FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"
CHECKPOINT_DIR = REPO_ROOT / "models" / "lstm" / "checkpoints"

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


def evaluate(
    model: TemporalPlayingClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(seq)
            total_loss += float(criterion(logits, labels).item())
            n_batches += 1
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().ravel()
            y = labels.long().cpu().numpy().ravel()
            tp += int(((preds == 1) & (y == 1)).sum())
            fp += int(((preds == 1) & (y == 0)).sum())
            tn += int(((preds == 0) & (y == 0)).sum())
            fn += int(((preds == 0) & (y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    cost = FN_COST * fn + FP_COST * fp

    return {
        "loss": total_loss / max(n_batches, 1),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost": cost,
    }


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
    from models.lstm.encoders import get_encoder

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
            logits = model(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        metrics = evaluate(model, test_loader, dev)
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

    print(f"done. checkpoints in {CHECKPOINT_DIR}")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Train BiLSTM on cached frame features.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", default=None, help="cuda, mps, or cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
