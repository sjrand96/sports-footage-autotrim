#!/usr/bin/env python3
"""Build labeled training windows from paired E2E feature/prediction parquets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.s3_cache import download_s3_uri, is_s3_uri, parse_s3_uri


def _list_s3_keys(prefix_uri: str, suffix: str) -> List[str]:
    import boto3

    parsed = parse_s3_uri(prefix_uri.rstrip("/") + "/")
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=parsed.bucket, Prefix=parsed.key):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                keys.append(f"s3://{parsed.bucket}/{key}")
    return sorted(keys)


def _list_local_files(prefix: str, suffix: str) -> List[str]:
    return sorted(str(path) for path in Path(prefix).glob(f"*{suffix}"))


def _stem(path_or_uri: str, suffix: str) -> str:
    name = path_or_uri.rstrip("/").split("/")[-1]
    if not name.endswith(suffix):
        raise ValueError(f"{path_or_uri} does not end with {suffix}")
    return name[: -len(suffix)]


def _resolve_files(prefix: str, cache_dir: str) -> List[Tuple[str, str, str]]:
    feature_suffix = "_features.parquet"
    pred_suffix = "_predictions.parquet"
    if is_s3_uri(prefix):
        feature_files = _list_s3_keys(prefix, feature_suffix)
        pred_files = _list_s3_keys(prefix, pred_suffix)
    else:
        feature_files = _list_local_files(prefix, feature_suffix)
        pred_files = _list_local_files(prefix, pred_suffix)

    features_by_stem = {_stem(path, feature_suffix): path for path in feature_files}
    preds_by_stem = {_stem(path, pred_suffix): path for path in pred_files}
    shared = sorted(set(features_by_stem) & set(preds_by_stem))
    if not shared:
        raise RuntimeError(
            f"No paired *_features.parquet / *_predictions.parquet files found under {prefix}."
        )

    pairs = []
    for stem in shared:
        feat_path = features_by_stem[stem]
        pred_path = preds_by_stem[stem]
        if is_s3_uri(feat_path):
            feat_path = download_s3_uri(feat_path, cache_dir)
        if is_s3_uri(pred_path):
            pred_path = download_s3_uri(pred_path, cache_dir)
        pairs.append((stem, feat_path, pred_path))
    return pairs


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _clip_relative_timestamps(df: Any, clip_duration_sec: float, clip_index: int | None) -> Any:
    timestamps = df["timestamp_sec"].astype(float)
    if len(timestamps) and float(timestamps.max()) > clip_duration_sec + 1.0 and clip_index is not None:
        return timestamps - (int(clip_index) - 1) * clip_duration_sec
    return timestamps


def _windows_for_clip(
    stem: str,
    df_feat: Any,
    df_pred: Any,
    window_sizes: List[float],
    stride_sec: float,
    min_positive_ratio: float,
    clip_duration_sec: float,
) -> Iterable[Dict[str, Any]]:
    if "is_playing" not in df_pred.columns:
        raise RuntimeError(f"{stem}_predictions.parquet is missing required column is_playing")
    if "timestamp_sec" not in df_pred.columns:
        raise RuntimeError(f"{stem}_predictions.parquet is missing required column timestamp_sec")

    source_id = str(df_feat["source_id"].dropna().iloc[0]) if "source_id" in df_feat.columns else stem.rsplit("_", 1)[0]
    clip_index = None
    if "clip_index" in df_feat.columns and len(df_feat["clip_index"].dropna()) > 0:
        clip_index = int(df_feat["clip_index"].dropna().iloc[0])
    elif stem.rsplit("_", 1)[-1].isdigit():
        clip_index = int(stem.rsplit("_", 1)[-1])

    clip_s3_uri = None
    for df in (df_feat, df_pred):
        if "clip_s3_uri" in df.columns and len(df["clip_s3_uri"].dropna()) > 0:
            clip_s3_uri = str(df["clip_s3_uri"].dropna().iloc[0])
            break

    timestamps = _clip_relative_timestamps(df_pred, clip_duration_sec, clip_index)
    labels = df_pred["is_playing"].astype(int)

    for window_size in window_sizes:
        max_start = max(0.0, clip_duration_sec - window_size)
        steps = int(math.floor(max_start / stride_sec)) + 1
        for step in range(steps):
            start = step * stride_sec
            end = start + window_size
            mask = (timestamps >= start) & (timestamps <= end)
            positive_ratio = float(labels[mask].mean()) if bool(mask.any()) else 0.0
            yield {
                "clip_id": stem,
                "clip_path": clip_s3_uri,
                "clip_s3_uri": clip_s3_uri,
                "source_id": source_id,
                "match_id": source_id,
                "clip_index": clip_index,
                "window_start_sec": round(start, 3),
                "window_end_sec": round(end, 3),
                "label": int(positive_ratio >= min_positive_ratio),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build window manifest from paired E2E parquets.")
    parser.add_argument("--parquet-prefix", required=True, help="Local dir or s3:// prefix containing paired parquets.")
    parser.add_argument("--output", required=True, help="Output JSONL window manifest.")
    parser.add_argument("--s3-cache-dir", default="data/s3_cache")
    parser.add_argument("--clip-duration-sec", type=float, default=60.0)
    parser.add_argument("--window-sizes-sec", default="2,3,4")
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--min-positive-ratio", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to build manifests from parquet files") from exc

    window_sizes = [float(x) for x in args.window_sizes_sec.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for stem, feat_path, pred_path in _resolve_files(args.parquet_prefix, args.s3_cache_dir):
        df_feat = pd.read_parquet(feat_path)
        df_pred = pd.read_parquet(pred_path)
        rows.extend(
            _windows_for_clip(
                stem,
                df_feat,
                df_pred,
                window_sizes,
                args.stride_sec,
                args.min_positive_ratio,
                args.clip_duration_sec,
            )
        )

    count = _write_jsonl(rows, args.output)
    print(f"Wrote {count} windows to {args.output}")


if __name__ == "__main__":
    main()
