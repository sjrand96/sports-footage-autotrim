#!/usr/bin/env python3
"""Extract and cache per-frame EfficientNetV2 embeddings for all clips.

Writes ``data/preprocessed_features/<backbone>/<clip_id>.pt`` and ``meta.json``.

Run::

    python models/lstm/extract_features.py
    python models/lstm/extract_features.py --device mps --batch-size 32
    python models/lstm/extract_features.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lstm.encoders import default_backbone, get_encoder, resolve_device  # noqa: E402

CLIPS_ROOT = REPO_ROOT / "data" / "clips"
FRAME_LABELS_CSV = REPO_ROOT / "data" / "preprocessed_labels" / "frame_labels.csv"
FEATURES_ROOT = REPO_ROOT / "data" / "preprocessed_features"
DEFAULT_BATCH_SIZE = 32


def clip_mp4_path(source_id: str, clip_id: str) -> Path:
    return CLIPS_ROOT / source_id / f"{clip_id}.mp4"


def list_clips_from_csv() -> list[tuple[str, str]]:
    """Return sorted (clip_id, source_id) pairs from the label table."""
    seen: set[tuple[str, str]] = set()
    with FRAME_LABELS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add((row["clip_id"], row["source_id"]))
    return sorted(seen)


def cache_path(backbone: str, clip_id: str) -> Path:
    return FEATURES_ROOT / backbone / f"{clip_id}.pt"


def needs_extract(mp4: Path, out_pt: Path, *, force: bool) -> bool:
    if force or not out_pt.is_file():
        return True
    return mp4.stat().st_mtime > out_pt.stat().st_mtime


def video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else 0
    finally:
        cap.release()


def encode_video_streaming(
    encoder,
    video_path: Path,
    *,
    batch_size: int,
    show_frames: bool = True,
) -> torch.Tensor:
    """Decode and encode frame-by-frame with batched GPU forward passes."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    total = video_frame_count(video_path)
    use_total = total if total > 0 else None
    feats: list[torch.Tensor] = []
    batch_frames: list = []

    frame_bar = tqdm(
        total=use_total,
        desc=video_path.stem,
        leave=False,
        disable=not show_frames,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch_frames.append(frame)
            if len(batch_frames) >= batch_size:
                images = encoder.preprocess_batch_bgr(batch_frames)
                feats.append(encoder.encode_batch(images).cpu())
                frame_bar.update(len(batch_frames))
                batch_frames = []

        if batch_frames:
            images = encoder.preprocess_batch_bgr(batch_frames)
            feats.append(encoder.encode_batch(images).cpu())
            frame_bar.update(len(batch_frames))
    finally:
        cap.release()
        frame_bar.close()

    if not feats:
        raise RuntimeError(f"video has no frames: {video_path}")

    return torch.cat(feats, dim=0).to(torch.float32)


def extract_one(
    encoder,
    *,
    clip_id: str,
    source_id: str,
    batch_size: int,
    force: bool,
    show_frames: bool,
) -> dict | None:
    mp4 = clip_mp4_path(source_id, clip_id)
    if not mp4.is_file():
        print(f"WARN: missing video {mp4}")
        return None

    out_pt = cache_path(encoder.name, clip_id)
    if not needs_extract(mp4, out_pt, force=force):
        return None

    features = encode_video_streaming(
        encoder,
        mp4,
        batch_size=batch_size,
        show_frames=show_frames,
    )

    payload = {
        "clip_id": clip_id,
        "source_id": source_id,
        "backbone": encoder.name,
        "feat_dim": encoder.feat_dim,
        "img_size": encoder.img_size,
        "num_frames": int(features.shape[0]),
        "features": features,
    }
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)

    if encoder.device.type == "mps":
        torch.mps.empty_cache()

    return {
        "clip_id": clip_id,
        "source_id": source_id,
        "num_frames": int(features.shape[0]),
        "feat_dim": encoder.feat_dim,
        "img_size": encoder.img_size,
        "video_path": str(mp4.relative_to(REPO_ROOT)),
        "cache_path": str(out_pt.relative_to(REPO_ROOT)),
    }


def write_meta(backbone: str, encoder, clips_meta: list[dict], *, device: str) -> None:
    meta_path = FEATURES_ROOT / backbone / "meta.json"
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "feat_dim": encoder.feat_dim,
        "img_size": encoder.img_size,
        "device": device,
        "num_clips": len(clips_meta),
        "clips": clips_meta,
        "frame_labels_csv": str(FRAME_LABELS_CSV.relative_to(REPO_ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def extract_all(
    *,
    backbone: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    device: str | None = None,
    show_frames: bool = True,
) -> None:
    if not FRAME_LABELS_CSV.is_file():
        raise RuntimeError(f"missing labels CSV: {FRAME_LABELS_CSV}; run data/preprocess_labels.py")

    backbone = backbone or default_backbone()
    dev = resolve_device(device)
    encoder = get_encoder(backbone, device=dev)
    print(
        f"backbone={encoder.name} feat_dim={encoder.feat_dim} "
        f"device={encoder.device} batch_size={batch_size}"
    )
    if encoder.device.type == "cpu":
        print(
            "WARN: running on CPU — extraction will be slow. "
            "On Apple Silicon use --device mps; on NVIDIA use --device cuda."
        )

    clips = list_clips_from_csv()
    clips_meta: list[dict] = []
    extracted = 0
    skipped = 0

    for clip_id, source_id in tqdm(clips, desc="clips"):
        mp4 = clip_mp4_path(source_id, clip_id)
        out_pt = cache_path(encoder.name, clip_id)
        if not needs_extract(mp4, out_pt, force=force):
            skipped += 1
            if out_pt.is_file():
                data = torch.load(out_pt, map_location="cpu", weights_only=False)
                clips_meta.append(
                    {
                        "clip_id": clip_id,
                        "source_id": source_id,
                        "num_frames": int(data["num_frames"]),
                        "feat_dim": int(data["feat_dim"]),
                        "img_size": int(data.get("img_size", 0)),
                        "video_path": str(mp4.relative_to(REPO_ROOT)),
                        "cache_path": str(out_pt.relative_to(REPO_ROOT)),
                    }
                )
            continue

        meta = extract_one(
            encoder,
            clip_id=clip_id,
            source_id=source_id,
            batch_size=batch_size,
            force=force,
            show_frames=show_frames,
        )
        if meta is None:
            continue
        clips_meta.append(meta)
        extracted += 1
        print(f"  {clip_id}: {meta['num_frames']} frames -> {meta['cache_path']}")

    if not clips_meta:
        raise RuntimeError("no clips processed")

    stale = [m["clip_id"] for m in clips_meta if m.get("img_size") != encoder.img_size]
    if stale:
        print(
            f"WARN: {len(stale)} clip(s) cached at wrong resolution {stale[:3]}... "
            f"(expected {encoder.img_size}px). Re-run with --force."
        )

    write_meta(backbone, encoder, clips_meta, device=str(encoder.device))
    print(f"done: extracted={extracted} skipped={skipped} meta={FEATURES_ROOT / backbone / 'meta.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache per-frame CNN features for LSTM training.")
    p.add_argument("--backbone", default=None, help=f"default: {default_backbone()}")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--device",
        default=None,
        help="cuda, mps (Apple GPU), or cpu; default: best available",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--no-frame-progress",
        action="store_true",
        help="hide per-clip frame tqdm (only show clip-level bar)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        extract_all(
            backbone=args.backbone,
            batch_size=args.batch_size,
            force=args.force,
            device=args.device,
            show_frames=not args.no_frame_progress,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
