"""Upload a local feature-extraction run to S3 (``feature_extraction/{run_id}/``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from feature_extraction.s3_layout import (
    default_bucket,
    default_region,
    feature_extraction_prefix,
    manifest_key,
    parquet_key,
    run_report_key,
    s3_uri,
    timings_key,
)

logger = logging.getLogger(__name__)


@dataclass
class UploadedObject:
    local_path: str
    s3_key: str
    s3_uri: str

    def to_dict(self) -> dict[str, str]:
        return {
            "local_path": self.local_path,
            "s3_key": self.s3_key,
            "s3_uri": self.s3_uri,
        }


@dataclass
class UploadFailure:
    local_path: str
    s3_key: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "local_path": self.local_path,
            "s3_key": self.s3_key,
            "error": self.error,
        }


@dataclass
class RunUploadResult:
    bucket: str
    run_id: str
    prefix: str
    uploaded: list[UploadedObject] = field(default_factory=list)
    failures: list[UploadFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def to_manifest_fragment(self) -> dict[str, Any]:
        return {
            "s3_bucket": self.bucket,
            "s3_prefix": self.prefix,
            "s3_run_uri": s3_uri(self.bucket, self.prefix),
            "s3_upload": {
                "n_uploaded": len(self.uploaded),
                "n_failed": len(self.failures),
                "uploaded": [u.to_dict() for u in self.uploaded],
                "failures": [f.to_dict() for f in self.failures],
            },
        }


def get_s3_client(region: str):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 upload. Install with `pip install boto3`.") from exc
    return boto3.client("s3", region_name=region)


def upload_file(*, client: Any, local_path: Path, bucket: str, key: str) -> None:
    from botocore.exceptions import ClientError

    if not local_path.is_file():
        raise FileNotFoundError(str(local_path))
    try:
        client.upload_file(str(local_path), bucket, key)
    except ClientError as exc:
        raise RuntimeError(f"S3 upload failed: s3://{bucket}/{key}: {exc}") from exc


def collect_run_files(
    run_dir: Path,
    *,
    phase: str = "all",
) -> list[tuple[Path, str]]:
    """Return ``(local_path, s3_key)`` under ``feature_extraction/{run_id}/``.

    ``phase``: ``parquets`` | ``sidecars`` (manifest + run_report) | ``all``
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    run_id = run_dir.name
    files: list[tuple[Path, str]] = []

    if phase in ("parquets", "all"):
        for split in ("train", "test"):
            split_dir = run_dir / split
            if not split_dir.is_dir():
                continue
            for parquet in sorted(split_dir.glob("*.parquet")):
                files.append((parquet, parquet_key(run_id, split, parquet.stem)))

    if phase in ("sidecars", "all"):
        for name, key_fn in (
            ("manifest.json", manifest_key),
            ("run_report.json", run_report_key),
            ("timings.json", timings_key),
        ):
            path = run_dir / name
            if path.is_file():
                files.append((path, key_fn(run_id)))

    if not files:
        raise RuntimeError(f"no uploadable files under {run_dir} (phase={phase})")
    return files


def upload_run_directory(
    run_dir: Path,
    *,
    bucket: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    phase: str = "all",
) -> RunUploadResult:
    """Upload all parquets + manifest + run_report for one local run."""
    run_dir = run_dir.resolve()
    rid = run_id or run_dir.name
    bkt = bucket or default_bucket()
    reg = region or default_region()
    prefix = feature_extraction_prefix(rid) + "/"

    client = get_s3_client(reg)
    result = RunUploadResult(bucket=bkt, run_id=rid, prefix=prefix)

    for local_path, key in collect_run_files(run_dir, phase=phase):
        try:
            upload_file(client=client, local_path=local_path, bucket=bkt, key=key)
            uri = s3_uri(bkt, key)
            result.uploaded.append(
                UploadedObject(local_path=str(local_path), s3_key=key, s3_uri=uri)
            )
            logger.info("uploaded %s -> %s", local_path.name, uri)
        except (OSError, RuntimeError) as exc:
            msg = str(exc)
            logger.error("upload failed %s: %s", local_path, msg)
            result.failures.append(
                UploadFailure(local_path=str(local_path), s3_key=key, error=msg)
            )

    return result


def parquet_s3_uri_for_success(
    *,
    bucket: str,
    run_id: str,
    split: str,
    source_id: str,
    clip_index: int,
) -> str:
    stem = f"{source_id}_{clip_index:03d}"
    return s3_uri(bucket, parquet_key(run_id, split, stem))


def apply_upload_to_successes(
    run_report: Any,
    upload_result: RunUploadResult,
) -> None:
    """Set ``output_s3_uri`` on successes when parquet upload succeeded."""
    uri_by_local: dict[str, str] = {}
    for obj in upload_result.uploaded:
        if obj.s3_key.endswith(".parquet"):
            uri_by_local[obj.local_path] = obj.s3_uri

    for success in run_report.successes:
        local = success.output_path
        if local in uri_by_local:
            success.output_s3_uri = uri_by_local[local]


def upload_failures_for_run_report(
    upload_result: RunUploadResult,
    *,
    run_report: Any,
    specs_by_stem: dict[str, Any],
) -> None:
    """Map parquet upload failures back to clip failures (stage=upload)."""
    from feature_extraction.manifest import ClipFailure

    for fail in upload_result.failures:
        if not fail.s3_key.endswith(".parquet"):
            continue
        stem = Path(fail.local_path).stem
        spec = specs_by_stem.get(stem)
        if spec is None:
            continue
        run_report.failures.append(
            ClipFailure(
                clip_id=spec.clip_id,
                source_id=spec.source_id,
                clip_index=spec.clip_index,
                stage="upload",
                error=fail.error,
            )
        )
        run_report.successes = [
            s
            for s in run_report.successes
            if not (s.source_id == spec.source_id and s.clip_index == spec.clip_index)
        ]
