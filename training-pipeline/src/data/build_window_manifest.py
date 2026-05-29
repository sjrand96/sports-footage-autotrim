#!/usr/bin/env python3
"""Build labeled sliding windows from canonical clip annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _window_label(
    window: Tuple[float, float],
    segments: List[Dict[str, Any]],
    play_labels: set[str],
    min_overlap_ratio: float,
) -> int:
    w_start, w_end = window
    duration = w_end - w_start
    if duration <= 0:
        return 0
    overlap = 0.0
    for seg in segments:
        label = str(seg.get("label", "")).strip().lower().replace(" ", "_")
        if label not in play_labels:
            continue
        overlap += _overlap_sec(w_start, w_end, seg["start_sec"], seg["end_sec"])
    return 1 if (overlap / duration) >= min_overlap_ratio else 0


def _build_windows(
    clip_row: Dict[str, Any],
    window_sizes: List[float],
    stride_sec: float,
    play_labels: set[str],
    min_overlap_ratio: float,
    clip_duration_sec: float,
) -> Iterable[Dict[str, Any]]:
    annotations = clip_row.get("annotations", [])
    clip_id = clip_row.get("clip_id")
    clip_path = clip_row.get("clip_path")
    source_id = clip_row.get("source_id")
    match_id = clip_row.get("match_id")

    for window_size in window_sizes:
        if window_size <= 0:
            continue
        max_start = max(0.0, clip_duration_sec - window_size)
        steps = int(math.floor(max_start / stride_sec)) + 1
        for step in range(steps):
            start = step * stride_sec
            end = start + window_size
            label = _window_label((start, end), annotations, play_labels, min_overlap_ratio)
            yield {
                "clip_id": clip_id,
                "clip_path": clip_path,
                "source_id": source_id,
                "match_id": match_id,
                "window_start_sec": round(start, 3),
                "window_end_sec": round(end, 3),
                "label": int(label),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labeled windows from canonical clip manifest.")
    parser.add_argument("--input", required=True, help="Canonical clip manifest JSONL.")
    parser.add_argument("--output", required=True, help="Output window manifest JSONL.")
    parser.add_argument("--clip-duration-sec", type=float, default=60.0, help="Clip duration in seconds.")
    parser.add_argument("--window-sizes-sec", default="2,3,4", help="Comma-separated window sizes.")
    parser.add_argument("--stride-sec", type=float, default=1.0, help="Sliding stride in seconds.")
    parser.add_argument(
        "--play-labels",
        default="play,playing,ball_in_play",
        help="Comma-separated labels to treat as playtime.",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.5,
        help="Minimum overlap ratio to label playtime.",
    )
    args = parser.parse_args()

    window_sizes = [float(x) for x in args.window_sizes_sec.split(",") if x.strip()]
    play_labels = {x.strip().lower().replace(" ", "_") for x in args.play_labels.split(",") if x.strip()}

    rows = []
    for clip_row in _load_jsonl(args.input):
        rows.extend(
            list(
                _build_windows(
                    clip_row,
                    window_sizes,
                    args.stride_sec,
                    play_labels,
                    args.min_overlap_ratio,
                    args.clip_duration_sec,
                )
            )
        )

    count = _write_jsonl(rows, args.output)
    print(f"Wrote {count} windows to {args.output}")


if __name__ == "__main__":
    main()
