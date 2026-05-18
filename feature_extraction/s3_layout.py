"""S3 key helpers for ``feature_extraction/{run_id}/`` (legacy: ``feature_extractions/``)."""

from __future__ import annotations

import os
from pathlib import Path

from feature_extraction.core.paths import DEFAULT_BUCKET, DEFAULT_REGION

# Singular prefix per PLAN.md (not legacy ``feature_extractions/``).
FEATURE_EXTRACTION_ROOT = "feature_extraction"


def default_bucket() -> str:
    return os.environ.get("S3_BUCKET", DEFAULT_BUCKET).strip() or DEFAULT_BUCKET


def default_region() -> str:
    return os.environ.get("AWS_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION


def feature_extraction_prefix(run_id: str) -> str:
    return f"{FEATURE_EXTRACTION_ROOT}/{run_id}"


def parquet_key(run_id: str, split: str, stem: str) -> str:
    return f"{feature_extraction_prefix(run_id)}/{split}/{stem}.parquet"


def manifest_key(run_id: str) -> str:
    return f"{feature_extraction_prefix(run_id)}/manifest.json"


def run_report_key(run_id: str) -> str:
    return f"{feature_extraction_prefix(run_id)}/run_report.json"


def timings_key(run_id: str) -> str:
    return f"{feature_extraction_prefix(run_id)}/timings.json"


def clip_timing_key(run_id: str, clip_id: int) -> str:
    return f"{feature_extraction_prefix(run_id)}/_clips/{clip_id}/timing.json"


def clip_timings_prefix(run_id: str) -> str:
    return f"{feature_extraction_prefix(run_id)}/_clips/"


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def run_s3_uri(bucket: str, run_id: str) -> str:
    return s3_uri(bucket, feature_extraction_prefix(run_id) + "/")


def local_run_dir(out_dir: Path, run_id: str) -> Path:
    return out_dir / run_id


def local_parquet_path(out_dir: Path, run_id: str, split: str, stem: str) -> Path:
    return local_run_dir(out_dir, run_id) / split / f"{stem}.parquet"


def local_manifest_path(out_dir: Path, run_id: str) -> Path:
    return local_run_dir(out_dir, run_id) / "manifest.json"


def local_run_report_path(out_dir: Path, run_id: str) -> Path:
    return local_run_dir(out_dir, run_id) / "run_report.json"


def local_timings_path(out_dir: Path, run_id: str) -> Path:
    return local_run_dir(out_dir, run_id) / "timings.json"


def local_clip_timing_path(out_dir: Path, run_id: str, clip_id: int) -> Path:
    return local_run_dir(out_dir, run_id) / "_clips" / str(clip_id) / "timing.json"
