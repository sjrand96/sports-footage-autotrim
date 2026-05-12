"""Push Label Studio **timeline** (Playing / timelinelabels) JSON exports into Supabase (`annotations`).

Court keypoint exports use `court_keypoints.py` and a separate calibration import (planned)—not this script.

Usage:
    python data_labeling/push_timeline_annotation_export.py path/to/project-export.json
    python data_labeling/push_timeline_annotation_export.py path/to/export.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_CLIP_KEY_RE = re.compile(
    r"clips/(?P<source_id>[^/]+)/(?P=source_id)_(?P<idx>\d+)\.mp4$",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_clip_from_video_url(video: str) -> tuple[str, int] | None:
    if not video or not isinstance(video, str):
        return None
    video = unquote(video.strip())
    if video.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return None

    path: str
    if video.startswith("s3://"):
        rest = video[5:]
        slash = rest.find("/")
        if slash == -1:
            return None
        path = rest[slash + 1 :]
    else:
        parsed = urlparse(video)
        path = parsed.path.lstrip("/")

    m = _CLIP_KEY_RE.search(path)
    if not m:
        return None
    return m.group("source_id"), int(m.group("idx"))


def pick_latest_annotation(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        if a.get("was_cancelled"):
            continue
        candidates.append(a)
    if not candidates:
        return None
    return max(candidates, key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""))


def main() -> int:
    from dotenv import load_dotenv
    from src import db

    parser = argparse.ArgumentParser(
        description="Import Label Studio timeline (Playing) JSON export into Supabase annotations."
    )
    parser.add_argument("export_json", type=Path, help="Path to Label Studio JSON export file")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Supabase")
    args = parser.parse_args()

    load_dotenv()
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "ANNOTATOR_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        log(f"ERROR: missing env vars: {', '.join(missing)}")
        return 1

    annotator = os.environ["ANNOTATOR_NAME"].strip()
    if not annotator:
        log("ERROR: ANNOTATOR_NAME is empty")
        return 1

    if not args.export_json.is_file():
        log(f"ERROR: not a file: {args.export_json}")
        return 1

    try:
        tasks = json.loads(args.export_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        log(f"ERROR: could not read JSON: {e}")
        return 1

    if not isinstance(tasks, list):
        log("ERROR: export JSON must be a list of tasks")
        return 1

    supabase = db.get_supabase_client()

    inserted = 0
    dry_run_would = 0
    skipped_dup = 0
    skipped_no_annotation = 0
    skipped_no_video = 0
    skipped_bad_url = 0
    skipped_no_clip = 0
    errors = 0

    for task in tasks:
        if not isinstance(task, dict):
            continue

        task_id = task.get("id")
        data = task.get("data")
        if not isinstance(data, dict):
            data = {}
        video = data.get("video")
        parsed = parse_clip_from_video_url(video) if isinstance(video, str) else None
        if video and parsed is None:
            if isinstance(video, str) and video.lower().split("?")[0].endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                skipped_no_video += 1
            else:
                skipped_bad_url += 1
                log(f"  WARN task {task_id}: could not parse clip URL: {video!r}")
            continue
        if not parsed:
            skipped_no_video += 1
            continue

        source_id, clip_index = parsed
        clip_row = db.get_clip(supabase, source_id, clip_index)
        if not clip_row:
            skipped_no_clip += 1
            log(f"  WARN task {task_id}: no clip row for {source_id=} {clip_index=}")
            continue

        clip_id = int(clip_row["id"])
        annotations = task.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            skipped_no_annotation += 1
            continue

        latest = pick_latest_annotation(annotations)
        if latest is None:
            skipped_no_annotation += 1
            continue

        if task_id is None:
            errors += 1
            log(f"  ERROR: task missing id, skipping (video={video!r})")
            continue

        tid = int(task_id)
        if db.annotation_exists_for_task(
            supabase,
            clip_id=clip_id,
            label_studio_task_id=tid,
            annotator=annotator,
        ):
            skipped_dup += 1
            continue

        project_id = task.get("project")
        if project_id is not None:
            project_id = int(project_id)

        lead = latest.get("lead_time")
        lead_sec = float(lead) if lead is not None else None

        payload: dict[str, Any] = {
            "label_studio_task": task,
            "label_studio_annotation": latest,
        }

        if args.dry_run:
            log(
                f"  DRY-RUN would insert: task_id={tid} clip_id={clip_id} "
                f"{source_id}_{clip_index:03d}.mp4 annotation_id={latest.get('id')}"
            )
            dry_run_would += 1
            continue

        try:
            db.insert_annotation(
                supabase,
                clip_id=clip_id,
                label_studio_task_id=tid,
                label_studio_project_id=project_id,
                annotator=annotator,
                lead_time_sec=lead_sec,
                payload=payload,
            )
            inserted += 1
            log(f"  inserted task_id={tid} clip_id={clip_id} ({source_id}_{clip_index:03d})")
        except Exception as e:  # noqa: BLE001
            errors += 1
            log(f"  ERROR task {tid}: {e}")

    log("")
    if args.dry_run:
        log(
            "Summary (dry-run): "
            f"would_insert={dry_run_would}, skipped_duplicate={skipped_dup}, "
            f"skipped_no_annotation={skipped_no_annotation}, skipped_no_video={skipped_no_video}, "
            f"skipped_bad_url={skipped_bad_url}, skipped_no_clip={skipped_no_clip}, errors={errors}"
        )
    else:
        log(
            "Summary: "
            f"inserted={inserted}, skipped_duplicate={skipped_dup}, "
            f"skipped_no_annotation={skipped_no_annotation}, skipped_no_video={skipped_no_video}, "
            f"skipped_bad_url={skipped_bad_url}, skipped_no_clip={skipped_no_clip}, errors={errors}"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

