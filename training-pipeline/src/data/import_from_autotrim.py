#!/usr/bin/env python3
"""Import clip metadata + annotations from sports-footage-autotrim exports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote, urlparse


_CLIP_KEY_RE = re.compile(
    r"clips/(?P<source_id>[^/]+)/(?P=source_id)_(?P<idx>\d+)\.mp4$",
    re.IGNORECASE,
)


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


def _parse_annotations(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _parse_video_labels(raw: Any, label_frame_rate: float) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []

    segments: List[Dict[str, Any]] = []
    for item in raw:
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
            segments.append(
                {
                    "start_sec": float(start) / label_frame_rate,
                    "end_sec": float(end) / label_frame_rate,
                    "label": label,
                }
            )
    return segments


def _pick_latest_annotation(annotations: Any) -> Dict[str, Any] | None:
    if not isinstance(annotations, list):
        return None
    candidates = [a for a in annotations if isinstance(a, dict) and not a.get("was_cancelled")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))


def _parse_label_studio_segments(row: Dict[str, Any], label_frame_rate: float) -> List[Dict[str, Any]]:
    latest = _pick_latest_annotation(row.get("annotations"))
    if latest is None:
        return []
    return _parse_video_labels(latest.get("result"), label_frame_rate)


def _pick_data_field(row: Dict[str, Any], name: str) -> Any:
    data = row.get("data")
    if isinstance(data, dict) and data.get(name) not in (None, ""):
        return data.get(name)
    return None


def _parse_clip_ref(ref: Any) -> tuple[str | None, int | None, str | None]:
    if not isinstance(ref, str) or not ref:
        return None, None, None
    text = unquote(ref.strip())
    if text.startswith("s3://"):
        path = text[5:].split("/", 1)[1] if "/" in text[5:] else ""
    else:
        path = urlparse(text).path.lstrip("/")
    match = _CLIP_KEY_RE.search(path)
    if not match:
        return None, None, None
    source_id = match.group("source_id")
    clip_index = int(match.group("idx"))
    return source_id, clip_index, f"{source_id}_{clip_index:03d}"


def _normalize_row(row: Dict[str, Any], field_map: Dict[str, str], label_frame_rate: float) -> Dict[str, Any]:
    def pick(*names: str) -> Any:
        for name in names:
            if name in row and row[name] not in (None, ""):
                return row[name]
        return None

    def pick_mapped(name: str, *fallback: str) -> Any:
        mapped = field_map.get(name)
        if mapped:
            return pick(mapped)
        return pick(*fallback)

    clip_path = pick_mapped("clip_path", "clip_path", "path", "video", "video_path") or _pick_data_field(row, "video")
    clip_id = pick_mapped("clip_id", "clip_id", "clip", "id")
    source_id = pick_mapped("source_id", "source_id", "source", "match_id", "game_id")
    match_id = pick_mapped("match_id", "match_id", "game_id", "source_id")
    parsed_source_id, parsed_clip_index, parsed_clip_id = _parse_clip_ref(clip_path)
    if parsed_clip_id and (not clip_id or str(clip_id).isdigit()):
        clip_id = parsed_clip_id
    if parsed_source_id and not source_id:
        source_id = parsed_source_id
    if parsed_source_id and not match_id:
        match_id = parsed_source_id
    annotations = pick_mapped("annotations", "annotations", "labels", "segments")
    annotations = _parse_annotations(annotations)
    if not annotations:
        annotations = _parse_video_labels(row.get("videoLabels"), label_frame_rate)
    if not annotations:
        annotations = _parse_label_studio_segments(row, label_frame_rate)

    normalized_segments: List[Dict[str, Any]] = []
    for seg in annotations:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start_sec", seg.get("start", seg.get("start_time")))
        end = seg.get("end_sec", seg.get("end", seg.get("end_time")))
        label = seg.get("label", seg.get("category", seg.get("name")))
        if start is None or end is None:
            continue
        normalized_segments.append(
            {
                "start_sec": float(start),
                "end_sec": float(end),
                "label": str(label) if label is not None else "",
            }
        )
    if not normalized_segments:
        fallback_annotations = _parse_video_labels(row.get("videoLabels"), label_frame_rate)
        if not fallback_annotations:
            fallback_annotations = _parse_label_studio_segments(row, label_frame_rate)
        for seg in fallback_annotations:
            normalized_segments.append(
                {
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "label": str(seg.get("label", "")),
                }
            )

    return {
        "clip_path": clip_path,
        "clip_id": clip_id,
        "source_id": source_id,
        "match_id": match_id,
        "clip_index": parsed_clip_index,
        "annotations": normalized_segments,
    }


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import autotrim exports into a canonical manifest.")
    parser.add_argument("--input", required=True, help="Path to autotrim export (csv/json/jsonl).")
    parser.add_argument("--output", required=True, help="Path to write canonical JSONL manifest.")
    parser.add_argument("--input-format", default=None, help="Optional override: csv/json/jsonl.")
    parser.add_argument(
        "--field-map",
        default="{}",
        help="JSON dict mapping canonical fields to input columns.",
    )
    parser.add_argument(
        "--label-frame-rate",
        type=float,
        default=30.0,
        help="Frame rate used by Label Studio timeline ranges; ranges are converted from frames to seconds.",
    )
    args = parser.parse_args()

    input_format = args.input_format or _infer_format(args.input)
    field_map = json.loads(args.field_map)

    rows = []
    for raw in _load_rows(args.input, input_format):
        normalized = _normalize_row(raw, field_map, args.label_frame_rate)
        if not normalized["clip_path"]:
            continue
        rows.append(normalized)

    count = _write_jsonl(rows, args.output)
    print(f"Wrote {count} clips to {args.output}")


if __name__ == "__main__":
    main()
