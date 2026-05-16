"""Small helpers for lazily caching S3 objects used by training."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def is_s3_uri(value: str | None) -> bool:
    return bool(value and value.startswith("s3://"))


def parse_s3_uri(uri: str) -> S3Uri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return S3Uri(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


def cache_path_for_s3_uri(uri: str, cache_dir: str, suffix: str | None = None) -> str:
    parsed = parse_s3_uri(uri)
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]
    name = os.path.basename(parsed.key) or digest
    if suffix and not name.endswith(suffix):
        name = f"{name}{suffix}"
    return str(Path(cache_dir) / parsed.bucket / os.path.dirname(parsed.key) / f"{digest}_{name}")


def download_s3_uri(uri: str, cache_dir: str, region_name: str | None = None) -> str:
    """Download an S3 object once and return its local cached path."""
    local_path = cache_path_for_s3_uri(uri, cache_dir)
    if os.path.exists(local_path):
        return local_path

    import boto3

    parsed = parse_s3_uri(uri)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = boto3.client("s3", region_name=region_name or os.environ.get("AWS_REGION", "us-west-2"))
    client.download_file(parsed.bucket, parsed.key, local_path)
    return local_path
