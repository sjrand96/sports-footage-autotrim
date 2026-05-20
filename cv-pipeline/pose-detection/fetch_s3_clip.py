#!/usr/bin/env python3
"""Download a clip from S3 into ``cv-pipeline/pose-detection/media/<key>`` (mirrors object key under a local root).

Uses AWS credentials from the environment (same as ``data_labeling/ingest_youtube_source``): load a ``.env``
at repo root via ``python-dotenv``.

Examples:
    # Default: the clip from your Label Studio task (id 78)
    python cv-pipeline/pose-detection/fetch_s3_clip.py

    python cv-pipeline/pose-detection/fetch_s3_clip.py "s3://bucket/clips/source/clip_006.mp4"

    python cv-pipeline/pose-detection/fetch_s3_clip.py --task-json path/to/task78.json

Default destination:
    cv-pipeline/pose-detection/media/clips/jZ18INu4LQc/jZ18INu4LQc_006.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_S3 = "s3://sports-footage-autotrim-bucket/clips/jZ18INu4LQc/jZ18INu4LQc_006.mp4"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    u = uri.strip()
    if not u.lower().startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got: {uri!r}")
    rest = u[5:]
    slash = rest.find("/")
    if slash == -1:
        raise ValueError(f"invalid S3 URI (no key): {uri!r}")
    return rest[:slash], rest[slash + 1 :]


def _video_from_task_json(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task JSON must be a single object")
    data = raw.get("data") or {}
    video = data.get("video")
    if not isinstance(video, str) or not video.startswith("s3://"):
        raise ValueError("task JSON missing data.video s3 URI")
    return video


def download_s3_object(bucket: str, key: str, dest: Path, *, region: str) -> None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as e:
        raise SystemExit("boto3 required (pip install boto3)") from e

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


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")

    p = argparse.ArgumentParser(description="Download one S3 clip into cv-pipeline/pose-detection/media/")
    p.add_argument(
        "s3_uri",
        nargs="?",
        default=None,
        help=f"S3 URI (default if no --task-json: {_DEFAULT_S3})",
    )
    p.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help="Single Label Studio task JSON file; uses data.video (overrides positional s3_uri)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=_REPO_ROOT / "cv-pipeline" / "pose-detection" / "media",
        help="Local directory; file is written to <out-root>/<s3-key>",
    )
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="S3 region for boto3",
    )
    args = p.parse_args()

    if args.task_json is not None:
        if not args.task_json.is_file():
            print(f"not found: {args.task_json}", file=sys.stderr)
            return 1
        uri = _video_from_task_json(args.task_json)
    elif args.s3_uri is not None:
        uri = args.s3_uri
    else:
        uri = _DEFAULT_S3

    bucket, key = _parse_s3_uri(uri)
    dest = args.out_root / key

    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(k)]
    if missing:
        print(f"Missing env: {', '.join(missing)} (and optionally AWS_REGION)", file=sys.stderr)
        return 1

    print(f"Downloading s3://{bucket}/{key}", flush=True)
    print(f"  -> {dest}", flush=True)
    try:
        download_s3_object(bucket, key, dest, region=args.region)
    except (RuntimeError, SystemExit) as e:
        print(str(e), file=sys.stderr)
        return 1

    print(dest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
