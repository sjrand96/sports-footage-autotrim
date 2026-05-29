#!/usr/bin/env python3
"""Build a clip-level window manifest from parquet labels."""

from __future__ import annotations

import argparse
import json
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


def _resolve_files(prefix: str, cache_dir: str) -> List[Tuple[str, str, str | None]]:
    pred_suffix = "_predictions.parquet"
    feat_suffix = "_features.parquet"
    if is_s3_uri(prefix):
        pred_files = _list_s3_keys(prefix, pred_suffix)
        feat_files = _list_s3_keys(prefix, feat_suffix)
    else:
        pred_files = _list_local_files(prefix, pred_suffix)
        feat_files = _list_local_files(prefix, feat_suffix)

    preds_by_stem = {_stem(path, pred_suffix): path for path in pred_files}
    feats_by_stem = {_stem(path, feat_suffix): path for path in feat_files}

    if not preds_by_stem:
        raise RuntimeError(f"No *_predictions.parquet files found under {prefix}.")

    pairs: List[Tuple[str, str, str | None]] = []
    for stem, pred_path in preds_by_stem.items():
        feat_path = feats_by_stem.get(stem)
        if is_s3_uri(pred_path):
            pred_path = download_s3_uri(pred_path, cache_dir)
        if feat_path and is_s3_uri(feat_path):
            feat_path = download_s3_uri(feat_path, cache_dir)
        pairs.append((stem, pred_path, feat_path))
    return pairs


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _clip_label_from_is_playing(values: Any, *, threshold: float) -> int:
    series = values.astype(float)
    if series.empty:
        return 0
    return int(float(series.mean()) >= threshold)


def _clip_meta_from_df(df: Any, stem: str) -> tuple[str | None, str | None, int | None]:
    source_id = None
    if "source_id" in df.columns and len(df["source_id"].dropna()) > 0:
        source_id = str(df["source_id"].dropna().iloc[0])
    clip_index = None
    if "clip_index" in df.columns and len(df["clip_index"].dropna()) > 0:
        clip_index = int(df["clip_index"].dropna().iloc[0])
    elif stem.rsplit("_", 1)[-1].isdigit():
        clip_index = int(stem.rsplit("_", 1)[-1])
    clip_s3_uri = None
    if "clip_s3_uri" in df.columns and len(df["clip_s3_uri"].dropna()) > 0:
        clip_s3_uri = str(df["clip_s3_uri"].dropna().iloc[0])
    return source_id, clip_s3_uri, clip_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clip-level window manifest from parquet labels.")
    parser.add_argument("--parquet-prefix", required=True, help="Local dir or s3:// prefix containing *_predictions.parquet files.")
    parser.add_argument("--output", required=True, help="Output JSONL window manifest.")
    parser.add_argument("--s3-cache-dir", default="data/s3_cache")
    parser.add_argument("--clip-duration-sec", type=float, default=60.0)
    parser.add_argument("--label-threshold", type=float, default=0.5, help="Threshold on mean is_playing for label=1.")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to read parquet files") from exc

    rows: List[Dict[str, Any]] = []
    for stem, pred_path, feat_path in _resolve_files(args.parquet_prefix, args.s3_cache_dir):
        df_pred = pd.read_parquet(pred_path)
        if "is_playing" not in df_pred.columns:
            raise RuntimeError(f"{pred_path} is missing required column is_playing")
        label = _clip_label_from_is_playing(df_pred["is_playing"], threshold=args.label_threshold)

        clip_s3_uri = None
        source_id = None
        clip_index = None
        if feat_path:
            df_feat = pd.read_parquet(feat_path)
            source_id, clip_s3_uri, clip_index = _clip_meta_from_df(df_feat, stem)
        if not clip_s3_uri:
            source_id2, clip_s3_uri2, clip_index2 = _clip_meta_from_df(df_pred, stem)
            source_id = source_id or source_id2
            clip_s3_uri = clip_s3_uri2
            clip_index = clip_index or clip_index2

        rows.append(
            {
                "clip_id": stem,
                "clip_path": clip_s3_uri,
                "clip_s3_uri": clip_s3_uri,
                "source_id": source_id or stem.rsplit("_", 1)[0],
                "match_id": source_id or stem.rsplit("_", 1)[0],
                "clip_index": clip_index,
                "window_start_sec": 0.0,
                "window_end_sec": float(args.clip_duration_sec),
                "label": int(label),
            }
        )

    count = _write_jsonl(rows, args.output)
    print(f"Wrote {count} clip windows to {args.output}")


if __name__ == "__main__":
    main()
