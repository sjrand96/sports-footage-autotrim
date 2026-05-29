#!/usr/bin/env python3
"""Build a window manifest from clip-level labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List


def _infer_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".jl"}:
        return "jsonl"
    if ext in {".json", ".js"}:
        return "json"
    if ext in {".csv", ".tsv"}:
        return "csv"
    return "jsonl"


def _load_rows(path: str, fmt: str) -> Iterable[Dict[str, Any]]:
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for row in payload:
                yield row
        elif isinstance(payload, dict):
            for row in payload.get("items", []):
                yield row
        return
    if fmt == "csv":
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
        return
    raise ValueError(f"Unsupported input format: {fmt}")


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _safe_clip_id(row: Dict[str, Any]) -> str | None:
    clip_id = row.get("clip_id")
    if clip_id:
        return str(clip_id)
    clip_path = row.get("clip_path") or row.get("path") or row.get("video") or row.get("video_path")
    if not clip_path:
        return None
    base = os.path.basename(str(clip_path))
    return os.path.splitext(base)[0]


def _label_to_int(value: Any, play_labels: set[str]) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) > 0 else 0
    text = str(value).strip().lower().replace(" ", "_")
    if text in play_labels:
        return 1
    if text in {"0", "false", "no", "downtime", "inactive"}:
        return 0
    try:
        return 1 if float(text) > 0 else 0
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create window manifest from clip-level labels.")
    parser.add_argument("--input", required=True, help="Clip-level label file (jsonl/json/csv).")
    parser.add_argument("--output", required=True, help="Output window manifest JSONL.")
    parser.add_argument("--input-format", default=None, help="Optional override: csv/json/jsonl.")
    parser.add_argument(
        "--field-map",
        default="{}",
        help="JSON map from canonical fields to input columns.",
    )
    parser.add_argument("--label-field", default="label", help="Field for clip-level labels.")
    parser.add_argument(
        "--play-labels",
        default="play,playing,ball_in_play,playtime,1,true,yes",
        help="Comma-separated label values that map to playtime.",
    )
    parser.add_argument("--clip-duration-sec", type=float, default=60.0)
    args = parser.parse_args()

    input_format = args.input_format or _infer_format(args.input)
    field_map = json.loads(args.field_map)
    play_labels = {x.strip().lower().replace(" ", "_") for x in args.play_labels.split(",") if x.strip()}

    def pick(row: Dict[str, Any], name: str, *fallback: str) -> Any:
        mapped = field_map.get(name)
        if mapped and row.get(mapped) not in (None, ""):
            return row.get(mapped)
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
        for fb in fallback:
            if fb in row and row.get(fb) not in (None, ""):
                return row.get(fb)
        return None

    rows: List[Dict[str, Any]] = []
    for raw in _load_rows(args.input, input_format):
        clip_path = pick(raw, "clip_path", "path", "video", "video_path")
        if not clip_path:
            continue
        clip_id = pick(raw, "clip_id") or _safe_clip_id(raw)
        label_value = pick(raw, "label", args.label_field)
        rows.append(
            {
                "clip_id": clip_id,
                "clip_path": clip_path,
                "source_id": pick(raw, "source_id", "source", "match_id", "game_id"),
                "match_id": pick(raw, "match_id", "game_id", "source_id"),
                "window_start_sec": 0.0,
                "window_end_sec": float(args.clip_duration_sec),
                "label": _label_to_int(label_value, play_labels),
            }
        )

    count = _write_jsonl(rows, args.output)
    print(f"Wrote {count} clip windows to {args.output}")


if __name__ == "__main__":
    main()
