"""Load calibration frames from local path or S3 (HTTPS public URL or boto3)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import requests

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV required: pip install -e '.[cv]'") from e


def fetch_image_bgr(bucket: str, key: str, *, region: str) -> np.ndarray | None:
    safe_key = requests.utils.quote(key, safe="/")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{safe_key}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        return None
    buf = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fetch_image_bgr_boto3(bucket: str, key: str) -> np.ndarray | None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    try:
        bc = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        obj = bc.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
    except ClientError:
        return None
    buf = np.frombuffer(body, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_image_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"could not read image: {path}")
    return im
