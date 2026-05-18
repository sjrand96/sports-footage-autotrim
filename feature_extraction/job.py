#!/usr/bin/env python3
"""Feature extraction job: clips → per-frame parquets + manifest (local and optional S3)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.clip_split import assign_train_test  # noqa: E402
from feature_extraction.core.clip_selection import (  # noqa: E402
    ClipSpec,
    clip_spec_by_id,
    list_eligible_clips,
)
from feature_extraction.core.extract import extract_features_for_clip  # noqa: E402
from feature_extraction.core.frames_export import local_frames_dir, upload_frames_directory  # noqa: E402
from feature_extraction.core.labels import add_ground_truth_labels, fetch_latest_annotation_payload  # noqa: E402
from feature_extraction.core.paths import (  # noqa: E402
    DEFAULT_LABEL_FPS,
    DEFAULT_REGION,
    clip_stem,
    fetch_module,
    homography_from_calibration_row,
    local_clip_path,
)
from feature_extraction.core.feature_columns import PARQUET_COLUMNS  # noqa: E402
from feature_extraction.manifest import (  # noqa: E402
    ClipFailure,
    ClipSuccess,
    RunReport,
    build_manifest,
    print_run_summary,
    write_manifest,
    write_run_report,
)
from feature_extraction.s3_layout import (  # noqa: E402
    default_bucket,
    local_clip_timing_path,
    local_manifest_path,
    local_parquet_path,
    local_run_dir,
    local_run_report_path,
    local_timings_path,
    run_s3_uri,
)
from feature_extraction.timing import (  # noqa: E402
    ClipTimer,
    build_timings_document,
    clip_entries_from_run_report,
    derived_timing_metrics,
    log_clip_timing,
    timing_summary_for_manifest,
    write_timings,
)
from feature_extraction.s3_upload import (  # noqa: E402
    RunUploadResult,
    apply_upload_to_successes,
    upload_failures_for_run_report,
    upload_run_directory,
)

logger = logging.getLogger(__name__)

SplitName = Literal["train", "test"]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _get_db():
    sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers

    return db_helpers, db_helpers.get_supabase_client()


def ensure_local_clip(*, s3_uri: str, local_path: Path, region: str) -> None:
    if local_path.is_file():
        logger.info("clip exists, skipping download: %s", local_path)
        return
    fetch = fetch_module()
    bucket, key = fetch._parse_s3_uri(s3_uri)
    logger.info("downloading s3://%s/%s", bucket, key)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fetch.download_s3_object(bucket, key, local_path, region=region)
    logger.info("downloaded to %s", local_path)


def process_one_clip(
    *,
    spec: ClipSpec,
    split: SplitName,
    run_id: str,
    out_dir: Path,
    db_helpers: Any,
    db_client: Any,
    region: str,
    label_fps: float,
    skip_download: bool,
    write_frames: bool,
    max_frames: int | None,
    bucket: str,
    delete_local_clip_after: bool = False,
) -> ClipSuccess:
    stem = clip_stem(spec.source_id, spec.clip_index)
    out_path = local_parquet_path(out_dir, run_id, split, stem)
    local_video = local_clip_path(spec.source_id, spec.clip_index)
    timer = ClipTimer()
    t_clip = perf_counter()

    frames_local: Path | None = local_frames_dir(spec.source_id, spec.clip_index) if write_frames else None

    with timer.stage("download"):
        if not skip_download:
            ensure_local_clip(s3_uri=spec.s3_uri, local_path=local_video, region=region)
        elif not local_video.is_file():
            raise FileNotFoundError(f"local clip missing: {local_video}")

    with timer.stage("calibration"):
        cal = db_helpers.get_court_calibration(db_client, spec.source_id)
        if cal is None:
            raise RuntimeError(f"no court_calibrations for source_id={spec.source_id!r}")
        H, wx_min, wx_max, wy_min, wy_max, _, _ = homography_from_calibration_row(cal)

    with timer.stage("extract"):
        df, video_meta = extract_features_for_clip(
            video_path=local_video,
            H=H,
            wx_min=wx_min,
            wx_max=wx_max,
            wy_min=wy_min,
            wy_max=wy_max,
            max_frames=max_frames,
            frames_dir=frames_local,
        )

    if write_frames and frames_local is not None:
        with timer.stage("frames_upload"):
            n_frames, frames_s3_uri = upload_frames_directory(
                frames_local,
                bucket=bucket,
                source_id=spec.source_id,
                clip_index=spec.clip_index,
                region=region,
            )
            logger.info("frames_upload count=%d uri=%s", n_frames, frames_s3_uri)

    with timer.stage("labels"):
        payload = fetch_latest_annotation_payload(db_helpers, db_client, spec.source_id, spec.clip_index)
        df = add_ground_truth_labels(df, payload, label_fps=label_fps)

    df["source_id"] = spec.source_id
    df["clip_index"] = spec.clip_index
    df["clip_id"] = spec.clip_id
    df["clip_s3_uri"] = spec.s3_uri
    df["clip_local_path"] = str(local_video.resolve())

    missing = [c for c in PARQUET_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"parquet missing columns: {missing}")

    with timer.stage("parquet_write"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df[PARQUET_COLUMNS].to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(df))

    timer.timings_sec["clip_total"] = perf_counter() - t_clip
    extract_sec = timer.timings_sec.get("extract", 0.0)
    derived = derived_timing_metrics(extract_sec=extract_sec, n_rows=len(df))

    success = ClipSuccess(
        clip_id=spec.clip_id,
        source_id=spec.source_id,
        clip_index=spec.clip_index,
        split=split,
        n_rows=len(df),
        output_path=str(out_path.resolve()),
        timings_sec=dict(timer.timings_sec),
        derived=derived or None,
        source_fps=float(video_meta["source_fps"]),
        n_source_frames=int(video_meta["n_source_frames"]),
    )
    log_clip_timing(success, log=logger)

    if delete_local_clip_after and local_video.is_file():
        local_video.unlink()
        logger.info("deleted local clip %s", local_video)

    return success


def _clip_split_map(
    train_clips: list[ClipSpec],
    test_clips: list[ClipSpec],
) -> dict[int, SplitName]:
    m: dict[int, SplitName] = {}
    for c in train_clips:
        m[c.clip_id] = "train"
    for c in test_clips:
        m[c.clip_id] = "test"
    return m


def _specs_stem_map(specs: list[ClipSpec]) -> dict[str, ClipSpec]:
    return {clip_stem(s.source_id, s.clip_index): s for s in specs}


def _split_meta_from_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    keys = (
        "split_method",
        "split_seed",
        "test_fraction",
        "train_clip_ids",
        "test_clip_ids",
    )
    return {k: manifest[k] for k in keys if k in manifest}


def _run_report_from_json(data: dict[str, Any] | None) -> RunReport:
    report = RunReport()
    if not data:
        return report
    for row in data.get("failures") or []:
        report.failures.append(
            ClipFailure(
                clip_id=int(row["clip_id"]),
                source_id=str(row["source_id"]),
                clip_index=int(row["clip_index"]),
                stage=str(row["stage"]),
                error=str(row["error"]),
            )
        )
    for row in data.get("successes") or []:
        report.successes.append(
            ClipSuccess(
                clip_id=int(row["clip_id"]),
                source_id=str(row["source_id"]),
                clip_index=int(row["clip_index"]),
                split=str(row["split"]),
                n_rows=int(row["n_rows"]),
                output_path=str(row["output_path"]),
                output_s3_uri=row.get("output_s3_uri"),
                timings_sec=row.get("timings_sec"),
                derived=row.get("derived"),
                source_fps=row.get("source_fps"),
                n_source_frames=row.get("n_source_frames"),
            )
        )
    return report


def _specs_by_stem_from_successes(run_report: RunReport) -> dict[str, ClipSpec]:
    out: dict[str, ClipSpec] = {}
    for s in run_report.successes:
        stem = clip_stem(s.source_id, s.clip_index)
        out[stem] = ClipSpec(
            clip_id=s.clip_id,
            source_id=s.source_id,
            clip_index=s.clip_index,
            s3_bucket="",
            s3_key="",
        )
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_upload_results(a: RunUploadResult, b: RunUploadResult) -> RunUploadResult:
    a.uploaded.extend(b.uploaded)
    a.failures.extend(b.failures)
    return a


def finalize_run_artifacts(
    *,
    run_id: str,
    out_dir: Path,
    split_meta: dict[str, Any],
    run_report: RunReport,
    label_fps: float,
    upload_s3: bool,
    bucket: str,
    region: str,
    specs_by_stem: dict[str, ClipSpec],
    run_started_at: datetime,
    max_frames: int | None,
    upload_parquet_only: bool = False,
) -> RunReport:
    """Write manifest/run_report/timings locally; optionally upload to S3 (parquets then sidecars)."""
    run_path = local_run_dir(out_dir, run_id)
    manifest_path = local_manifest_path(out_dir, run_id)
    report_path = local_run_report_path(out_dir, run_id)
    timings_path = local_timings_path(out_dir, run_id)

    s3_extra: dict[str, Any] | None = None
    parquet_upload: RunUploadResult | None = None
    upload_sec: float | None = None

    if upload_s3 and (run_report.n_success > 0 or any((run_path / s).glob("*.parquet") for s in ("train", "test"))):
        t_upload = perf_counter()
        parquet_upload = upload_run_directory(
            run_path,
            bucket=bucket,
            region=region,
            run_id=run_id,
            phase="parquets",
        )
        apply_upload_to_successes(run_report, parquet_upload)
        upload_failures_for_run_report(
            parquet_upload,
            run_report=run_report,
            specs_by_stem=specs_by_stem,
        )
        s3_extra = {
            "s3_bucket": bucket,
            "s3_prefix": parquet_upload.prefix,
            "s3_run_uri": run_s3_uri(bucket, run_id),
        }

    run_finished_at = datetime.now(timezone.utc)
    timings_doc = build_timings_document(
        run_id=run_id,
        run_report=run_report,
        started_at=run_started_at,
        finished_at=run_finished_at,
        max_frames=max_frames,
        upload_sec=upload_sec,
    )
    write_timings(timings_path, timings_doc)
    logger.info("wrote timings %s", timings_path)

    manifest_extra = dict(s3_extra or {})
    manifest_extra["timing_summary"] = timing_summary_for_manifest(timings_doc)

    manifest = build_manifest(
        run_id=run_id,
        split_meta=split_meta,
        run_report=run_report,
        out_dir=run_path,
        label_fps=label_fps,
        extra=manifest_extra or None,
    )
    if not upload_parquet_only:
        write_run_report(report_path, run_report)
        write_manifest(manifest_path, manifest)
        logger.info("wrote manifest %s", manifest_path)
        logger.info("wrote run_report %s", report_path)

    clip_entries = clip_entries_from_run_report(run_report)
    for entry in clip_entries:
        shard = {
            "run_id": run_id,
            "clips": [entry],
        }
        shard_path = local_clip_timing_path(out_dir, run_id, int(entry["clip_id"]))
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(json.dumps(shard, indent=2) + "\n", encoding="utf-8")

    if upload_s3 and upload_parquet_only and clip_entries:
        from feature_extraction.finalize_run import upload_clip_timing_shards
        from feature_extraction.s3_upload import get_s3_client

        upload_clip_timing_shards(
            get_s3_client(region),
            bucket=bucket,
            run_id=run_id,
            clip_entries=clip_entries,
        )
        logger.info("uploaded %d clip timing shard(s) to S3", len(clip_entries))

    if upload_s3 and parquet_upload is not None and not upload_parquet_only:
        from feature_extraction.s3_layout import manifest_key, timings_key
        from feature_extraction.s3_upload import get_s3_client, upload_file

        sidecar_upload = upload_run_directory(
            run_path,
            bucket=bucket,
            region=region,
            run_id=run_id,
            phase="sidecars",
        )
        merged = _merge_upload_results(parquet_upload, sidecar_upload)
        upload_sec = perf_counter() - t_upload
        timings_doc = build_timings_document(
            run_id=run_id,
            run_report=run_report,
            started_at=run_started_at,
            finished_at=datetime.now(timezone.utc),
            max_frames=max_frames,
            upload_sec=upload_sec,
        )
        write_timings(timings_path, timings_doc)
        manifest["s3_upload"] = merged.to_manifest_fragment()["s3_upload"]
        manifest["timing_summary"] = timing_summary_for_manifest(timings_doc)
        write_manifest(manifest_path, manifest)

        client = get_s3_client(region)
        upload_file(client=client, local_path=manifest_path, bucket=bucket, key=manifest_key(run_id))
        upload_file(client=client, local_path=timings_path, bucket=bucket, key=timings_key(run_id))
        logger.info("S3 run: %s", run_s3_uri(bucket, run_id))
        if merged.failures:
            for f in merged.failures:
                logger.error("S3 upload failed: %s -> %s: %s", f.local_path, f.s3_key, f.error)

    return run_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-frame features to local run directory.",
        allow_abbrev=False,
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Local root for feature_extraction/{run_id}/")
    p.add_argument("--run-id", type=str, default=None, help="Override run id (default: new timestamp+uuid)")
    p.add_argument("--clip-id", type=int, action="append", default=None, help="Supabase clips.id (repeatable)")
    p.add_argument("--max-clips", type=int, default=None, help="Cap clips when using full eligible list")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Random placeholder test fraction")
    p.add_argument("--split-seed", type=int, default=42, help="RNG seed for train/test placeholder split")
    p.add_argument("--label-fps", type=float, default=DEFAULT_LABEL_FPS)
    p.add_argument("--region", type=str, default=DEFAULT_REGION)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument(
        "--write-frames",
        action="store_true",
        help="Write every decoded frame as JPEG under clips_v2/…/frames/ and upload to S3",
    )
    p.add_argument("--fail-fast", action="store_true", help="Stop after first clip failure")
    p.add_argument("--dry-run", action="store_true", help="List clips and split only; no extract")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Smoke test only: process at most N frames per clip (default: all frames)",
    )
    p.add_argument(
        "--upload-s3",
        action="store_true",
        help="After extract, upload to s3://{bucket}/feature_extraction/{run_id}/",
    )
    p.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip extract; upload existing local run (--run-id required)",
    )
    p.add_argument(
        "--bucket",
        type=str,
        default=None,
        help=f"S3 bucket (default: $S3_BUCKET or {default_bucket()})",
    )
    p.add_argument(
        "--force-split",
        choices=("train", "test"),
        default=None,
        help="Force train/test partition (parallel workers; one clip per task).",
    )
    p.add_argument(
        "--upload-parquet-only",
        action="store_true",
        help="Upload parquets + per-clip timing shards only (no manifest/timings overwrite).",
    )
    p.add_argument(
        "--delete-local-clip-after",
        action="store_true",
        help="Remove downloaded MP4 after successful extract (saves ephemeral disk).",
    )
    return p.parse_args()


def main() -> int:
    _configure_logging()
    args = parse_args()

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[misc, assignment]
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    if args.upload_only and not args.run_id:
        logger.error("--upload-only requires --run-id")
        return 1
    if args.upload_only and args.upload_s3 is False:
        args.upload_s3 = True
    if args.test_fraction <= 0 or args.test_fraction >= 1:
        logger.error("--test-fraction must be in (0, 1)")
        return 1
    if args.force_split and args.clip_id and len(args.clip_id) > 1:
        logger.error("--force-split with multiple --clip-id values is not supported")
        return 1

    bucket = args.bucket or default_bucket()
    run_id = args.run_id or _new_run_id()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    local_run_dir(out_dir, run_id).mkdir(parents=True, exist_ok=True)

    db_helpers, db_client = _get_db()

    if args.upload_only:
        run_path = local_run_dir(out_dir, run_id)
        if not run_path.is_dir():
            logger.error("upload-only: run directory not found: %s", run_path)
            return 1
        existing_manifest = _load_json(local_manifest_path(out_dir, run_id))
        split_meta = _split_meta_from_manifest(existing_manifest)
        run_report = _run_report_from_json(_load_json(local_run_report_path(out_dir, run_id)))
        if not run_report.successes and not (run_path / "train").exists() and not (run_path / "test").exists():
            logger.error("upload-only: no parquets or run_report under %s", run_id)
            return 1
        specs_by_stem = _specs_by_stem_from_successes(run_report)
        run_started_at = datetime.now(timezone.utc)
        run_report = finalize_run_artifacts(
            run_id=run_id,
            out_dir=out_dir,
            split_meta=split_meta,
            run_report=run_report,
            label_fps=args.label_fps,
            upload_s3=args.upload_s3,
            bucket=bucket,
            region=args.region,
            specs_by_stem=specs_by_stem,
            run_started_at=run_started_at,
            max_frames=args.max_frames,
            upload_parquet_only=args.upload_parquet_only,
        )
        print_run_summary(run_report)
        return 0 if run_report.n_failed == 0 else 1

    if args.clip_id:
        specs: list[ClipSpec] = []
        for cid in args.clip_id:
            spec = clip_spec_by_id(db_helpers, db_client, cid)
            if spec is None:
                logger.error("clip_id=%s not found or not eligible (annotation + calibration required)", cid)
                return 1
            specs.append(spec)
    else:
        specs = list_eligible_clips(db_helpers, db_client)
        if args.max_clips is not None:
            specs = specs[: args.max_clips]

    if not specs:
        logger.error("no eligible clips to process")
        return 1

    if args.force_split:
        if len(specs) != 1:
            logger.error("--force-split requires exactly one --clip-id per task (use run_fanout.py for many clips)")
            return 1
        forced: SplitName = args.force_split  # type: ignore[assignment]
        split_by_id = {specs[0].clip_id: forced}
        split_meta = {
            "split_method": "worker_cli",
            "split_seed": int(args.split_seed),
            "test_fraction": float(args.test_fraction),
            "train_clip_ids": [specs[0].clip_id] if forced == "train" else [],
            "test_clip_ids": [specs[0].clip_id] if forced == "test" else [],
        }
    else:
        train_clips, test_clips, split_meta = assign_train_test(
            specs,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        split_by_id = _clip_split_map(train_clips, test_clips)

    n_train = sum(1 for v in split_by_id.values() if v == "train")
    n_test = sum(1 for v in split_by_id.values() if v == "test")
    logger.info("run_id=%s", run_id)
    logger.info("eligible clips: %d (train=%d test=%d)", len(specs), n_train, n_test)
    for s in specs:
        logger.info("  %s_%03d clip_id=%d -> %s", s.source_id, s.clip_index, s.clip_id, split_by_id[s.clip_id])

    if args.dry_run:
        return 0

    specs_by_stem = _specs_stem_map(specs)
    run_report = RunReport()
    run_started_at = datetime.now(timezone.utc)

    for spec in specs:
        split = split_by_id[spec.clip_id]
        logger.info("processing %s_%03d [%s]", spec.source_id, spec.clip_index, split)
        try:
            success = process_one_clip(
                spec=spec,
                split=split,
                run_id=run_id,
                out_dir=out_dir,
                db_helpers=db_helpers,
                db_client=db_client,
                region=args.region,
                label_fps=args.label_fps,
                skip_download=args.skip_download,
                write_frames=args.write_frames,
                max_frames=args.max_frames,
                bucket=bucket,
                delete_local_clip_after=args.delete_local_clip_after,
            )
            run_report.successes.append(success)
        except FileNotFoundError as exc:
            _record_failure(run_report, spec, "download", exc)
            if args.fail_fast:
                break
        except RuntimeError as exc:
            stage = _infer_stage(str(exc))
            _record_failure(run_report, spec, stage, exc)
            if args.fail_fast:
                break
        except Exception as exc:  # noqa: BLE001
            _record_failure(run_report, spec, "extract", exc)
            if args.fail_fast:
                break

    run_report = finalize_run_artifacts(
        run_id=run_id,
        out_dir=out_dir,
        split_meta=split_meta,
        run_report=run_report,
        label_fps=args.label_fps,
        upload_s3=args.upload_s3,
        bucket=bucket,
        region=args.region,
        specs_by_stem=specs_by_stem,
        run_started_at=run_started_at,
        max_frames=args.max_frames,
        upload_parquet_only=args.upload_parquet_only,
    )

    print_run_summary(run_report)
    return 0 if run_report.n_failed == 0 else 1


def _record_failure(run_report: RunReport, spec: ClipSpec, stage: str, exc: BaseException) -> None:
    msg = str(exc)
    logger.error(
        "%s_%03d clip_id=%s failed [%s]: %s",
        spec.source_id,
        spec.clip_index,
        spec.clip_id,
        stage,
        msg,
    )
    run_report.failures.append(
        ClipFailure(
            clip_id=spec.clip_id,
            source_id=spec.source_id,
            clip_index=spec.clip_index,
            stage=stage,
            error=msg,
        )
    )


def _infer_stage(msg: str) -> str:
    lower = msg.lower()
    if "court_calibration" in lower or "homography" in lower or "calibration" in lower:
        return "calibration"
    if "annotation" in lower:
        return "labels"
    if "video" in lower or "frame" in lower:
        return "extract"
    return "extract"


if __name__ == "__main__":
    raise SystemExit(main())
