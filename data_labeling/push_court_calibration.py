"""Fit court homography from Label Studio keypoint export (or normalized payloads) and upsert `court_calibrations`.

Usage:
    python data_labeling/push_court_calibration.py path/to/export.json
    python data_labeling/push_court_calibration.py path/to/export.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_CALIB_DIR = REPO_ROOT / "cv-pipeline" / "calibration"
for _p in (REPO_ROOT, _CALIB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def log(msg: str) -> None:
    print(msg, flush=True)


def load_calibration_pairs(export_path: Path) -> list[tuple[Any, dict[str, Any] | None]]:
    """Return ``(CalibrationRecord, raw_task_or_none)`` for each usable record."""
    from data_labeling.court_keypoints import (
        CalibrationRecord,
        calibration_payload_to_record,
        task_to_calibration_record,
    )

    raw = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_list: list[dict[str, Any]] = [raw]
    elif isinstance(raw, list):
        raw_list = [x for x in raw if isinstance(x, dict)]
    else:
        raise ValueError("export JSON must be an array of tasks/objects or a single object")

    if not raw_list:
        return []

    if raw_list[0].get("kind") == "court_keypoints_label_studio":
        out: list[tuple[CalibrationRecord, dict[str, Any] | None]] = []
        for item in raw_list:
            if item.get("kind") != "court_keypoints_label_studio":
                continue
            out.append((calibration_payload_to_record(item), None))
        return out

    out_ls: list[tuple[CalibrationRecord, dict[str, Any] | None]] = []
    for task in raw_list:
        rec = task_to_calibration_record(task)
        if rec is not None and rec.keypoints:
            out_ls.append((rec, task))
    return out_ls


def _annotation_sort_key(rec: Any) -> str:
    return str(rec.annotation_updated_at or rec.annotation_created_at or "")


def _build_raw_export(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if task is None:
        return None
    from data_labeling.court_keypoints import pick_latest_annotation

    ann = pick_latest_annotation(task.get("annotations") or [])
    return {
        "label_studio_task_id": task.get("id"),
        "label_studio_project": task.get("project"),
        "data": task.get("data"),
        "annotation": ann,
    }


def main() -> int:
    from dotenv import load_dotenv

    import court_homography
    from src import db

    default_geometry = _CALIB_DIR / "fivb_court_geometry.txt"

    parser = argparse.ArgumentParser(
        description="Push court calibration: Label Studio keypoint export → fit homography → upsert court_calibrations."
    )
    parser.add_argument("export_json", type=Path, help="Label Studio JSON export or normalized court_keypoints payloads")
    parser.add_argument("--geometry", type=Path, default=default_geometry, help="FIVB planar point table")
    parser.add_argument("--source-id", default=None, help="Only process this YouTube source_id")
    parser.add_argument("--pixels-per-metre", type=float, default=None, help="Override default (45) for stored column")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Supabase")
    parser.add_argument("--stop-on-error", action="store_true", help="Exit on first fit or DB error")
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
    if not args.geometry.is_file():
        log(f"ERROR: geometry file not found: {args.geometry}")
        return 1

    try:
        pairs = load_calibration_pairs(args.export_json)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as e:
        log(f"ERROR: could not load export: {e}")
        return 1

    if not pairs:
        log("ERROR: no calibration records parsed from export")
        return 1

    by_source: dict[str, list[tuple[Any, dict[str, Any] | None]]] = defaultdict(list)
    for rec, task in pairs:
        sid = (rec.source_id or "").strip()
        if not sid:
            continue
        if args.source_id and sid != args.source_id.strip():
            continue
        by_source[sid].append((rec, task))

    if not by_source:
        log("ERROR: no records after --source-id filter")
        return 1

    ppm = (
        float(args.pixels_per_metre)
        if args.pixels_per_metre is not None
        else float(court_homography._PIXELS_PER_METRE)
    )

    supabase = db.get_supabase_client()
    now_iso = datetime.now(timezone.utc).isoformat()

    upserted = 0
    skipped_no_source = 0
    errors = 0

    for source_id in sorted(by_source):
        items = by_source[source_id]
        rec, task = max(items, key=lambda it: _annotation_sort_key(it[0]))
        if len(items) > 1:
            log(f"  note {source_id}: {len(items)} tasks in export; using latest annotation timestamp")

        if not rec.image_s3_bucket or not rec.image_s3_key:
            log(f"  WARN {source_id}: missing ref image S3 bucket/key; skip")
            errors += 1
            if args.stop_on_error:
                return 1
            continue

        if db.get_source_video(supabase, source_id) is None:
            skipped_no_source += 1
            log(f"  WARN {source_id}: no source_videos row; run ingest_youtube_source first — skip")
            if args.stop_on_error:
                return 1
            continue

        try:
            row_fit, info = court_homography.fit_calibration_record_for_db(
                rec,
                args.geometry,
                pixels_per_metre=ppm,
            )
        except (ValueError, RuntimeError, OSError) as e:
            errors += 1
            log(f"  ERROR {source_id}: fit failed: {e}")
            if args.stop_on_error:
                return 1
            continue

        ref_clip = int(rec.clip_index) if rec.clip_index is not None and rec.clip_index >= 0 else None
        pid = rec.label_studio_project_id
        if pid is not None:
            pid = int(pid)

        row: dict[str, Any] = {
            "source_id": source_id,
            "ref_image_s3_bucket": rec.image_s3_bucket,
            "ref_image_s3_key": rec.image_s3_key,
            "ref_image_width_px": int(rec.image_width_px),
            "ref_image_height_px": int(rec.image_height_px),
            "ref_clip_index": ref_clip,
            "label_studio_task_id": int(rec.label_studio_task_id) if rec.label_studio_task_id is not None else None,
            "label_studio_annotation_id": int(rec.label_studio_annotation_id)
            if rec.label_studio_annotation_id is not None
            else None,
            "label_studio_project_id": pid,
            "annotator": annotator,
            "raw_label_studio_export": _build_raw_export(task),
            "schema_version": 1,
            "updated_at": now_iso,
            **row_fit,
        }

        rmse = info.get("rmse_all_px")
        inl = info.get("inliers")
        npl = info.get("n_planar_pairs")
        log(
            f"  {source_id}: rmse_px={rmse} inliers={inl}/{npl} "
            f"Wx=[{row['world_wx_min']:.2f},{row['world_wx_max']:.2f}] "
            f"Wy=[{row['world_wy_min']:.2f},{row['world_wy_max']:.2f}]"
        )

        if args.dry_run:
            log(f"  DRY-RUN would upsert court_calibrations for {source_id}")
            upserted += 1
            continue

        try:
            db.upsert_court_calibration(supabase, row)
            upserted += 1
            log(f"  upserted court_calibrations source_id={source_id}")
        except Exception as e:  # noqa: BLE001
            errors += 1
            log(f"  ERROR {source_id}: DB: {e}")
            if args.stop_on_error:
                return 1

    log("")
    log(
        f"Summary: upserted_or_dry_run={upserted}, skipped_no_source_video={skipped_no_source}, "
        f"errors={errors}, sources_in_export={len(by_source)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
