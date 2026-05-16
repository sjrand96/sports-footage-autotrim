"""Download training clips from S3 and timeline labels from Supabase.

Clips are stored under ``data/clips/<source_id>/``; labels under ``data/labels/<source_id>/``.
Loads ``.env`` from the repo root (``SUPABASE_*``, ``AWS_*``, optional ``S3_BUCKET``).

Run directly::

    python data/fetch_data.py

Edit ``SOURCE_IDS`` below to choose which sources to fetch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS_ROOT = REPO_ROOT / "data" / "clips"
DEFAULT_LABELS_ROOT = REPO_ROOT / "data" / "labels"
DEFAULT_BUCKET = "sports-footage-autotrim-bucket"
DEFAULT_REGION = "us-west-2"


def ensure_env_loaded() -> None:
    """Load repo-root ``.env`` (Supabase and AWS credentials)."""
    load_dotenv(REPO_ROOT / ".env")


def clip_local_path(source_id: str, filename: str, *, clips_root: Path = DEFAULT_CLIPS_ROOT) -> Path:
    """Local path for one clip MP4 under ``data/clips/<source_id>/``."""
    return clips_root / source_id / filename


def label_local_path(
    source_id: str,
    clip_index: int,
    *,
    labels_root: Path = DEFAULT_LABELS_ROOT,
) -> Path:
    """Local path for one clip's ground-truth JSON under ``data/labels/<source_id>/``."""
    return labels_root / source_id / f"{source_id}_{clip_index:03d}.json"


def download_s3_object(bucket: str, key: str, dest: Path, *, region: str) -> None:
    """Stream one S3 object to ``dest`` via a temporary ``.part`` file."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        raise RuntimeError("boto3 required (pip install boto3)") from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=region)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"S3 get_object failed: {bucket=} {key=} {e}") from e

    body = obj["Body"]
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with open(tmp, "wb") as f:
            for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def _get_db_helpers() -> Any:
    """Import and return ``src.db`` (adds repo root to ``sys.path`` if needed)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers

    return db_helpers

def list_clips_for_source(client: Any, source_id: str) -> list[dict[str, Any]]:
    """Return all ``clips`` rows for a YouTube ``source_id``, ordered by ``clip_index``."""
    res = (
        client.table("clips")
        .select("id,source_id,clip_index,filename,s3_bucket,s3_key")
        .eq("source_id", source_id)
        .order("clip_index")
        .execute()
    )
    return list(res.data or [])

def fetch_latest_annotation(
    client: Any,
    *,
    clip_id: int,
) -> dict[str, Any] | None:
    """Return the newest ``annotations`` row for ``clip_id``, or None."""
    res = (
        client.table("annotations")
        .select("id,payload,exported_at,annotator,label_studio_task_id")
        .eq("clip_id", clip_id)
        .order("exported_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None

def download_clips(
    source_id: str,
    *,
    clips_root: Path = DEFAULT_CLIPS_ROOT,
    bucket: str | None = None,
    region: str | None = None,
) -> list[Path]:
    """Download all clips for ``source_id`` from S3 into ``data/clips/<source_id>/``.

    Skips files that already exist locally. Warns and continues on per-clip failures.
    """
    ensure_env_loaded()
    bucket = bucket or os.environ.get("S3_BUCKET", DEFAULT_BUCKET)
    region = region or os.environ.get("AWS_REGION", DEFAULT_REGION)

    db_helpers = _get_db_helpers()
    client = db_helpers.get_supabase_client()
    clip_rows = list_clips_for_source(client, source_id)
    if not clip_rows:
        print(f"WARN: no clips in Supabase for source_id={source_id!r}")
        return []

    downloaded: list[Path] = []
    for row in clip_rows:
        filename = str(row["filename"])
        dest = clip_local_path(source_id, filename, clips_root=clips_root)
        if dest.is_file():
            print(f"already exists: {dest}")
            downloaded.append(dest)
            continue

        s3_bucket = str(row.get("s3_bucket") or bucket)
        s3_key = str(row["s3_key"])
        print(f"downloading clip: s3://{s3_bucket}/{s3_key}")
        try:
            download_s3_object(s3_bucket, s3_key, dest, region=region)
        except Exception as e:
            print(f"WARN: failed to download {dest.name} ({s3_key}): {e}")
            continue
        print(f"downloaded to: {dest}")
        downloaded.append(dest)

    return downloaded

def download_labels(
    source_id: str,
    *,
    labels_root: Path = DEFAULT_LABELS_ROOT,
) -> list[Path]:
    """Download latest timeline annotation payloads for each clip under ``source_id``.

    Writes one JSON file per annotated clip under ``data/labels/<source_id>/``.
    Skips files that already exist locally. Warns and continues on per-clip failures.
    """
    ensure_env_loaded()

    db_helpers = _get_db_helpers()
    client = db_helpers.get_supabase_client()
    clip_rows = list_clips_for_source(client, source_id)
    if not clip_rows:
        print(f"WARN: no clips in Supabase for source_id={source_id!r}")
        return []

    saved: list[Path] = []
    for row in clip_rows:
        clip_index = int(row["clip_index"])
        clip_id = int(row["id"])
        dest = label_local_path(source_id, clip_index, labels_root=labels_root)

        if dest.is_file():
            print(f"already exists: {dest}")
            saved.append(dest)
            continue

        ann = fetch_latest_annotation(client, clip_id=clip_id)
        if ann is None:
            print(
                f"WARN: no annotation in Supabase for {source_id}_{clip_index:03d} "
                f"(clip_id={clip_id})"
            )
            continue

        payload = ann.get("payload")
        if not isinstance(payload, dict):
            print(f"WARN: invalid annotation payload for clip_id={clip_id}, skipping")
            continue

        record = {
            "source_id": source_id,
            "clip_index": clip_index,
            "clip_id": clip_id,
            "annotation_id": ann.get("id"),
            "exported_at": ann.get("exported_at"),
            "annotator": ann.get("annotator"),
            "label_studio_task_id": ann.get("label_studio_task_id"),
            "payload": payload,
        }
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"WARN: failed to write labels for {source_id}_{clip_index:03d}: {e}")
            continue
        print(f"wrote labels: {dest}")
        saved.append(dest)

    if not saved:
        print(f"WARN: no label files saved for source_id={source_id!r}")
    return saved


if __name__ == "__main__":
    ensure_env_loaded()

    # Edit this list, then run: python data/fetch_data.py
    SOURCE_IDS = [
        "jZ18INu4LQc",
        "vq3CZAx3GnM",
        "GRuOSrz3kdY",
        "1rXZJyVXUHU",
        "6rRMEXuLAng",
        "Fr3ue3w5QRY",
        "ANwMhMfcwGM",
        "2crSZaHIBaY",
    ]

    if not SOURCE_IDS:
        print("SOURCE_IDS is empty; add source IDs to data/fetch_data.py", file=sys.stderr)
        sys.exit(1)

    for source_id in SOURCE_IDS:
        print(f"\n=== clips: {source_id} ===")
        download_clips(source_id)
        print(f"\n=== labels: {source_id} ===")
        download_labels(source_id)

    print("\ndone.")
