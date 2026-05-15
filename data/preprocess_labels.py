"""Build a single CSV of per-frame playing labels from ``data/labels/**/*.json``.

Reads Label Studio timeline exports (Playing ranges in frame indices @ 30 FPS),
aligns to each clip's frame count from the matching MP4 under ``data/clips/``,
and writes:

  data/preprocessed_labels/frame_labels.csv
  data/preprocessed_labels/meta.json

Run::

    python data/preprocess_labels.py

Use ``--force`` to rebuild even when the cache looks up to date.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_ROOT = REPO_ROOT / "data" / "clips"
LABELS_ROOT = REPO_ROOT / "data" / "labels"
CACHE_DIR = REPO_ROOT / "data" / "preprocessed_labels"
FRAME_LABELS_CSV = CACHE_DIR / "frame_labels.csv"
META_JSON = CACHE_DIR / "meta.json"

LABEL_FPS = 30.0
_CLIP_STEM_RE = re.compile(r"^(?P<source_id>.+)_(?P<clip_index>\d+)$")

CSV_COLUMNS = ("clip_id", "source_id", "clip_index", "frame_idx", "is_playing")


def playing_ranges_frame_indices(payload: dict[str, Any]) -> list[tuple[int, int]]:
    """Parse Playing timeline ranges as inclusive frame index pairs."""
    ranges: list[tuple[int, int]] = []
    ann = payload.get("label_studio_annotation")
    if not isinstance(ann, dict):
        return ranges

    result = ann.get("result")
    if not isinstance(result, list):
        return ranges

    for item in result:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        if value.get("timelinelabels") != ["Playing"]:
            continue
        item_ranges = value.get("ranges")
        if not isinstance(item_ranges, list):
            continue
        for r in item_ranges:
            if not isinstance(r, dict):
                continue
            start = r.get("start")
            end = r.get("end")
            if start is None or end is None:
                continue
            s = int(min(start, end))
            e = int(max(start, end))
            ranges.append((s, e))

    return ranges


def parse_clip_stem(clip_id: str) -> tuple[str, int]:
    """Split ``jZ18INu4LQc_006`` into source id and 1-based clip index."""
    m = _CLIP_STEM_RE.match(clip_id)
    if not m:
        raise ValueError(f"could not parse clip_id: {clip_id!r}")
    return m.group("source_id"), int(m.group("clip_index"))


def clip_mp4_path(source_id: str, clip_id: str) -> Path:
    """Path to the MP4 for a label JSON stem."""
    return CLIPS_ROOT / source_id / f"{clip_id}.mp4"


def read_num_frames(video_path: Path) -> int:
    """Return frame count from a video file via OpenCV."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python required (pip install opencv-python-headless)") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return count
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        if n == 0:
            raise RuntimeError(f"video has no frames: {video_path}")
        return n
    finally:
        cap.release()


def frame_labels_from_ranges(
    num_frames: int,
    ranges: list[tuple[int, int]],
) -> np.ndarray:
    """Build ``uint8`` vector of length ``num_frames``; 1 = playing (inclusive ranges)."""
    labels = np.zeros(num_frames, dtype=np.uint8)
    for start, end in ranges:
        s = max(0, start)
        e = min(num_frames - 1, end)
        if e < s:
            continue
        labels[s : e + 1] = 1
    return labels


def load_payload(label_json_path: Path) -> dict[str, Any]:
    """Load the Label Studio ``payload`` object from a fetch_data export JSON."""
    raw = json.loads(label_json_path.read_text(encoding="utf-8"))
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"missing payload in {label_json_path}")
    return payload


def iter_label_json_paths() -> list[Path]:
    """All exported label JSON files under ``data/labels/``."""
    return sorted(LABELS_ROOT.rglob("*.json"))


def newest_mtime(paths: list[Path]) -> float:
    if not paths:
        return 0.0
    return max(p.stat().st_mtime for p in paths)


def cache_is_fresh(label_paths: list[Path], *, force: bool) -> bool:
    if force or not FRAME_LABELS_CSV.is_file() or not META_JSON.is_file():
        return False
    cache_mtime = min(FRAME_LABELS_CSV.stat().st_mtime, META_JSON.stat().st_mtime)
    return newest_mtime(label_paths) <= cache_mtime


def build_frame_label_rows(label_json_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return CSV rows and per-clip metadata for one label file."""
    clip_id = label_json_path.stem
    source_id, clip_index = parse_clip_stem(clip_id)
    video_path = clip_mp4_path(source_id, clip_id)

    if not video_path.is_file():
        raise FileNotFoundError(f"no matching clip: {video_path}")

    payload = load_payload(label_json_path)
    ranges = playing_ranges_frame_indices(payload)
    num_frames = read_num_frames(video_path)
    labels = frame_labels_from_ranges(num_frames, ranges)

    rows: list[dict[str, Any]] = []
    for frame_idx in range(num_frames):
        rows.append(
            {
                "clip_id": clip_id,
                "source_id": source_id,
                "clip_index": clip_index,
                "frame_idx": frame_idx,
                "is_playing": int(labels[frame_idx]),
            }
        )

    meta = {
        "clip_id": clip_id,
        "source_id": source_id,
        "clip_index": clip_index,
        "num_frames": num_frames,
        "num_playing_frames": int(labels.sum()),
        "label_json": str(label_json_path.relative_to(REPO_ROOT)),
        "video_path": str(video_path.relative_to(REPO_ROOT)),
        "playing_ranges": [{"start": s, "end": e} for s, e in ranges],
    }
    return rows, meta


def write_cache(
    all_rows: list[dict[str, Any]],
    clips_meta: list[dict[str, Any]],
    *,
    label_paths: list[Path],
) -> None:
    """Write ``frame_labels.csv`` and ``meta.json``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with FRAME_LABELS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "label_fps": LABEL_FPS,
        "num_clips": len(clips_meta),
        "num_rows": len(all_rows),
        "csv_path": str(FRAME_LABELS_CSV.relative_to(REPO_ROOT)),
        "columns": list(CSV_COLUMNS),
        "clips": clips_meta,
        "source_label_files": [str(p.relative_to(REPO_ROOT)) for p in label_paths],
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def preprocess_all(*, force: bool = False) -> Path:
    """Build label cache from all JSON exports; return path to CSV."""
    label_paths = iter_label_json_paths()
    if not label_paths:
        raise RuntimeError(f"no label JSON files under {LABELS_ROOT}")

    if cache_is_fresh(label_paths, force=force):
        print(f"cache up to date: {FRAME_LABELS_CSV}")
        return FRAME_LABELS_CSV

    all_rows: list[dict[str, Any]] = []
    clips_meta: list[dict[str, Any]] = []
    warnings = 0

    for label_path in label_paths:
        try:
            rows, clip_meta = build_frame_label_rows(label_path)
        except Exception as e:
            warnings += 1
            print(f"WARN: skipping {label_path.name}: {e}")
            continue
        all_rows.extend(rows)
        clips_meta.append(clip_meta)
        playing = clip_meta["num_playing_frames"]
        total = clip_meta["num_frames"]
        print(f"  {clip_meta['clip_id']}: {playing}/{total} playing frames")

    if not all_rows:
        raise RuntimeError("no clips processed successfully")

    write_cache(all_rows, clips_meta, label_paths=label_paths)
    print(f"wrote {len(all_rows)} rows -> {FRAME_LABELS_CSV}")
    print(f"wrote meta -> {META_JSON}")
    if warnings:
        print(f"finished with {warnings} warning(s)")
    return FRAME_LABELS_CSV


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build data/preprocessed_labels/frame_labels.csv from label JSON.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if newer than source JSON files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        preprocess_all(force=args.force)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
