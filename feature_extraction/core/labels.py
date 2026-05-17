"""Timeline annotation → per-frame ``is_playing``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def extract_playing_ranges_seconds(payload: dict[str, Any], label_fps: float) -> list[tuple[float, float]]:
    ranges_sec: list[tuple[float, float]] = []
    ann = payload.get("label_studio_annotation")
    if isinstance(ann, dict):
        result = ann.get("result")
        if isinstance(result, list):
            for item in result:
                if not isinstance(item, dict):
                    continue
                value = item.get("value")
                if not isinstance(value, dict):
                    continue
                labels = value.get("timelinelabels")
                if labels != ["Playing"]:
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
                    s = float(min(start, end)) / label_fps
                    e = float(max(start, end)) / label_fps
                    ranges_sec.append((s, e))
    return ranges_sec


def add_ground_truth_labels(df_feat: pd.DataFrame, payload: dict[str, Any], label_fps: float) -> pd.DataFrame:
    ranges_sec = extract_playing_ranges_seconds(payload, label_fps=label_fps)
    ts = df_feat["timestamp_sec"].to_numpy(dtype=np.float64)
    labels = np.zeros(ts.shape[0], dtype=bool)
    for start_sec, end_sec in ranges_sec:
        labels |= (ts >= start_sec) & (ts <= end_sec)
    out = df_feat.copy()
    out["is_playing"] = labels
    return out


def fetch_latest_annotation_payload(db_helpers: Any, client: Any, source_id: str, clip_index: int) -> dict[str, Any]:
    clip_row = db_helpers.get_clip(client, source_id, clip_index)
    if clip_row is None:
        raise RuntimeError(f"clip not found in Supabase: source_id={source_id} clip_index={clip_index}")

    clip_id = int(clip_row["id"])
    res = (
        client.table("annotations")
        .select("id,payload,exported_at")
        .eq("clip_id", clip_id)
        .order("exported_at", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(f"no annotations found for clip_id={clip_id} ({source_id}_{clip_index:03d})")

    payload = res.data[0].get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("latest annotation row has invalid payload")
    return payload
