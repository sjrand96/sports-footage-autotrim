"""Optional per-frame JPEG export to ``clips_v2/`` (local + S3)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from feature_extraction.core.paths import REPO_ROOT

logger = logging.getLogger(__name__)

CLIPS_V2_ROOT = REPO_ROOT / "cv-pipeline" / "pose-detection" / "media" / "clips_v2"
JPEG_QUALITY = 85


def local_frames_dir(source_id: str, clip_index: int) -> Path:
    return CLIPS_V2_ROOT / source_id / str(clip_index) / "frames"


def clips_v2_frames_s3_prefix(source_id: str, clip_index: int) -> str:
    return f"clips_v2/{source_id}/{clip_index}/frames"


def upload_frames_directory(
    local_dir: Path,
    *,
    bucket: str,
    source_id: str,
    clip_index: int,
    region: str,
) -> tuple[int, str]:
    """Upload ``*.jpg`` under ``local_dir`` to S3. Returns (count, s3_prefix)."""
    from feature_extraction.s3_upload import get_s3_client, upload_file

    if not local_dir.is_dir():
        raise FileNotFoundError(f"frames directory not found: {local_dir}")

    jpgs = sorted(local_dir.glob("*.jpg"))
    if not jpgs:
        raise RuntimeError(f"no JPEG frames to upload in {local_dir}")

    prefix = clips_v2_frames_s3_prefix(source_id, clip_index)
    client = get_s3_client(region)
    for jpg in jpgs:
        key = f"{prefix}/{jpg.name}"
        upload_file(client=client, local_path=jpg, bucket=bucket, key=key)

    s3_uri = f"s3://{bucket}/{prefix}/"
    logger.info("uploaded %d frames -> %s", len(jpgs), s3_uri)
    return len(jpgs), s3_uri
