#!/usr/bin/env python3
"""Download one annotated S3 clip plus its Label Studio label payload.

Uses only the Python standard library so it can run before project dependencies
are installed. Secrets are read from .env and are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _require(env: dict[str, str], key: str) -> str:
    value = env.get(key) or os.environ.get(key)
    if not value:
        raise SystemExit(f"missing required env var: {key}")
    return value


def _request_json(url: str, service_key: str) -> Any:
    req = Request(
        url,
        headers={
            "apikey": service_key,
            "authorization": f"Bearer {service_key}",
            "accept": "application/json",
        },
    )
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"user-agent": "sports-footage-autotrim-sample/1.0"})
    with urlopen(req, timeout=300) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _s3_https(bucket: str, key: str, region: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{quote(key, safe='/')}"


def _normalize_task(row: dict[str, Any], local_video_path: Path) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    task = payload.get("label_studio_task") if isinstance(payload.get("label_studio_task"), dict) else {}
    annotation = payload.get("label_studio_annotation") if isinstance(payload.get("label_studio_annotation"), dict) else {}

    if task:
        out = dict(task)
    else:
        out = {"id": row.get("label_studio_task_id"), "annotations": []}

    data = out.get("data") if isinstance(out.get("data"), dict) else {}
    data = dict(data)
    data["video"] = str(local_video_path)
    out["data"] = data

    if annotation:
        out["annotations"] = [annotation]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one labelled S3 sample clip and export JSON.")
    parser.add_argument("--source-id", default=None, help="Optional YouTube/source id to sample from.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "s3_sample")
    args = parser.parse_args()

    env = {**_load_env(REPO_ROOT / ".env"), **os.environ}
    supabase_url = _require(env, "SUPABASE_URL").rstrip("/")
    service_key = _require(env, "SUPABASE_SERVICE_KEY")
    region = env.get("AWS_REGION") or "us-west-2"

    select = (
        "id,clip_id,label_studio_task_id,label_studio_project_id,annotator,"
        "lead_time_sec,exported_at,payload,"
        "clips(id,source_id,clip_index,filename,s3_bucket,s3_key,thumbnail_s3_key,duration_sec)"
    )
    params = {
        "select": select,
        "order": "exported_at.desc",
        "limit": "25" if args.source_id else "1",
    }
    url = f"{supabase_url}/rest/v1/annotations?{urlencode(params)}"

    try:
        rows = _request_json(url, service_key)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"could not query Supabase annotations: {exc}") from exc

    if not isinstance(rows, list) or not rows:
        raise SystemExit("no annotation rows found in Supabase")

    chosen: dict[str, Any] | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        clip = row.get("clips")
        if not isinstance(clip, dict):
            continue
        if args.source_id and clip.get("source_id") != args.source_id:
            continue
        chosen = row
        break

    if chosen is None:
        raise SystemExit(f"no annotated clip found for source_id={args.source_id!r}")

    clip = chosen["clips"]
    bucket = str(clip["s3_bucket"])
    key = str(clip["s3_key"])
    filename = str(clip.get("filename") or Path(key).name)
    local_video = args.output_dir / filename
    export_json = args.output_dir / "label_studio_export.json"
    manifest_json = args.output_dir / "sample_info.json"

    video_url = _s3_https(bucket, key, region)
    try:
        _download(video_url, local_video)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"could not download S3 clip {key!r}: {exc}") from exc

    task = _normalize_task(chosen, local_video.resolve())
    export_json.write_text(json.dumps([task], indent=2, ensure_ascii=True), encoding="utf-8")
    manifest_json.write_text(
        json.dumps(
            {
                "source_id": clip.get("source_id"),
                "clip_index": clip.get("clip_index"),
                "s3_bucket": bucket,
                "s3_key": key,
                "local_video": str(local_video.resolve()),
                "label_studio_export": str(export_json.resolve()),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    print(f"downloaded_video={local_video}")
    print(f"label_export={export_json}")
    print(f"sample_info={manifest_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
