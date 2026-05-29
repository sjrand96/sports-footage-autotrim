#!/usr/bin/env python3
"""Count parquet files and total rows under S3 train/test prefixes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

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


def _resolve_files(prefix: str) -> List[str]:
    suffix = ".parquet"
    if is_s3_uri(prefix):
        return _list_s3_keys(prefix, suffix)
    return _list_local_files(prefix, suffix)


def _count_rows(parquet_paths: Iterable[str], cache_dir: str) -> Tuple[int, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to count parquet rows") from exc

    total_rows = 0
    file_count = 0
    for path in parquet_paths:
        local_path = download_s3_uri(path, cache_dir) if is_s3_uri(path) else path
        meta = pq.ParquetFile(local_path).metadata
        total_rows += int(meta.num_rows) if meta is not None else 0
        file_count += 1
    return file_count, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Count parquet files and rows in S3 prefixes.")
    parser.add_argument("--train-prefix", required=True, help="S3 or local prefix for train parquets.")
    parser.add_argument("--test-prefix", required=True, help="S3 or local prefix for test parquets.")
    parser.add_argument("--s3-cache-dir", default="data/s3_cache")
    args = parser.parse_args()

    train_files = _resolve_files(args.train_prefix)
    test_files = _resolve_files(args.test_prefix)

    train_count, train_rows = _count_rows(train_files, args.s3_cache_dir)
    test_count, test_rows = _count_rows(test_files, args.s3_cache_dir)

    print("train:")
    print(f"  files: {train_count}")
    print(f"  rows:  {train_rows}")
    print("test:")
    print(f"  files: {test_count}")
    print(f"  rows:  {test_rows}")


if __name__ == "__main__":
    main()