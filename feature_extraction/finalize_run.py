"""Merge parallel worker outputs into run-level manifest, run_report, and timings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from feature_extraction.core.paths import clip_stem
from feature_extraction.manifest import (
    ClipFailure,
    ClipSuccess,
    RunReport,
    build_manifest,
    write_manifest,
    write_run_report,
)
from feature_extraction.s3_layout import (
    clip_timing_key,
    clip_timings_prefix,
    default_bucket,
    feature_extraction_prefix,
    local_manifest_path,
    local_run_dir,
    local_run_report_path,
    local_timings_path,
    manifest_key,
    parquet_key,
    run_report_key,
    run_s3_uri,
    timings_key,
)
from feature_extraction.s3_upload import get_s3_client, upload_file
from feature_extraction.timing import (
    build_timings_document,
    merge_clip_timing_shards,
    timing_summary_for_manifest,
    write_timings,
)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def run_report_from_clip_entries(
    clip_entries: list[dict[str, Any]],
    *,
    bucket: str,
    run_id: str,
) -> RunReport:
    report = RunReport()
    for entry in clip_entries:
        if entry.get("status") == "ok":
            stem = f"{entry['source_id']}_{int(entry['clip_index']):03d}"
            split = str(entry["split"])
            report.successes.append(
                ClipSuccess(
                    clip_id=int(entry["clip_id"]),
                    source_id=str(entry["source_id"]),
                    clip_index=int(entry["clip_index"]),
                    split=split,
                    n_rows=int(entry.get("n_rows", 0)),
                    output_path="",
                    output_s3_uri=f"s3://{bucket}/{parquet_key(run_id, split, stem)}",
                    timings_sec=entry.get("timings_sec"),
                    derived=entry.get("derived"),
                    source_fps=entry.get("source_fps"),
                    n_source_frames=entry.get("n_source_frames"),
                )
            )
        else:
            report.failures.append(
                ClipFailure(
                    clip_id=int(entry["clip_id"]),
                    source_id=str(entry["source_id"]),
                    clip_index=int(entry["clip_index"]),
                    stage=str(entry.get("failed_stage") or "extract"),
                    error=str(entry.get("error") or "unknown"),
                )
            )
    return report


def run_report_from_plan_and_tasks(
    plan: dict[str, Any],
    *,
    bucket: str,
    run_id: str,
) -> RunReport:
    """Build run report from plan clips + ECS task outcomes + optional timing shards on S3."""
    clip_entries = list(plan.get("clip_timing_entries") or [])
    if clip_entries:
        return run_report_from_clip_entries(clip_entries, bucket=bucket, run_id=run_id)

    by_id = {int(c["clip_id"]): c for c in plan.get("clips") or []}
    report = RunReport()
    for task in plan.get("tasks") or []:
        cid = int(task["clip_id"])
        clip = by_id.get(cid)
        if clip is None:
            continue
        if task.get("exit_code") == 0 and task.get("parquet_on_s3"):
            stem = clip_stem(str(clip["source_id"]), int(clip["clip_index"]))
            split = str(clip["split"])
            report.successes.append(
                ClipSuccess(
                    clip_id=cid,
                    source_id=str(clip["source_id"]),
                    clip_index=int(clip["clip_index"]),
                    split=split,
                    n_rows=int(task.get("n_rows") or 0),
                    output_path="",
                    output_s3_uri=f"s3://{bucket}/{parquet_key(run_id, split, stem)}",
                )
            )
        else:
            report.failures.append(
                ClipFailure(
                    clip_id=cid,
                    source_id=str(clip["source_id"]),
                    clip_index=int(clip["clip_index"]),
                    stage=str(task.get("failed_stage") or "extract"),
                    error=str(task.get("error") or task.get("stopped_reason") or "task failed"),
                )
            )
    return report


def list_clip_timing_shards_s3(
    client: Any,
    *,
    bucket: str,
    run_id: str,
) -> list[dict[str, Any]]:
    prefix = clip_timings_prefix(run_id)
    shards: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj.get("Key") or ""
            if not key.endswith("/timing.json"):
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            shards.append(json.loads(body.decode("utf-8")))
    return shards


def clip_entries_from_timing_shards(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    for shard in shards:
        clips.extend(shard.get("clips") or [])
    return clips


def finalize_run_from_plan(
    *,
    plan: dict[str, Any],
    out_dir: Path,
    bucket: str | None = None,
    region: str | None = None,
    label_fps: float = 30.0,
    upload_s3: bool = True,
    publish_wandb: bool | None = None,
) -> tuple[RunReport, dict[str, Any]]:
    """Write merged manifest/run_report/timings locally; upload sidecars to S3."""
    from feature_extraction.s3_layout import default_region

    run_id = str(plan["run_id"])
    bkt = bucket or default_bucket()
    reg = region or default_region()
    split_meta = dict(plan.get("split_meta") or {})
    max_frames = plan.get("max_frames")

    client = get_s3_client(reg)
    shards = list_clip_timing_shards_s3(client, bucket=bkt, run_id=run_id)
    clip_entries = clip_entries_from_timing_shards(shards)
    plan["clip_timing_entries"] = clip_entries

    run_report = run_report_from_clip_entries(clip_entries, bucket=bkt, run_id=run_id)
    if not run_report.successes and not run_report.failures:
        run_report = run_report_from_plan_and_tasks(plan, bucket=bkt, run_id=run_id)

    started = _parse_iso(plan.get("started_at")) or datetime.now(timezone.utc)
    finished = datetime.now(timezone.utc)
    upload_sec = plan.get("finalize_upload_sec")

    timings_doc = merge_clip_timing_shards(
        shards,
        run_id=run_id,
        started_at=started,
        finished_at=finished,
        max_frames=max_frames,
        upload_sec=upload_sec,
    )
    if not shards and run_report.successes:
        timings_doc = build_timings_document(
            run_id=run_id,
            run_report=run_report,
            started_at=started,
            finished_at=finished,
            max_frames=max_frames,
            upload_sec=upload_sec,
        )

    run_path = local_run_dir(out_dir, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    write_timings(local_timings_path(out_dir, run_id), timings_doc)

    manifest_extra: dict[str, Any] = {
        "s3_bucket": bkt,
        "s3_prefix": feature_extraction_prefix(run_id) + "/",
        "s3_run_uri": run_s3_uri(bkt, run_id),
        "timing_summary": timing_summary_for_manifest(timings_doc),
        "orchestration": "parallel_ecs_fanout",
    }
    manifest = build_manifest(
        run_id=run_id,
        split_meta=split_meta,
        run_report=run_report,
        out_dir=run_path,
        label_fps=label_fps,
        extra=manifest_extra,
    )
    write_run_report(local_run_report_path(out_dir, run_id), run_report)
    write_manifest(local_manifest_path(out_dir, run_id), manifest)

    if upload_s3:
        upload_file(
            client=client,
            local_path=local_manifest_path(out_dir, run_id),
            bucket=bkt,
            key=manifest_key(run_id),
        )
        upload_file(
            client=client,
            local_path=local_run_report_path(out_dir, run_id),
            bucket=bkt,
            key=run_report_key(run_id),
        )
        upload_file(
            client=client,
            local_path=local_timings_path(out_dir, run_id),
            bucket=bkt,
            key=timings_key(run_id),
        )

    from feature_extraction.wandb_publish import maybe_publish_feature_run_artifact

    wandb_artifact = maybe_publish_feature_run_artifact(
        run_id=run_id,
        manifest=manifest,
        out_dir=out_dir,
        bucket=bkt,
        publish=publish_wandb,
    )
    if wandb_artifact:
        manifest["wandb_feature_artifact"] = wandb_artifact
        write_manifest(local_manifest_path(out_dir, run_id), manifest)
        if upload_s3:
            upload_file(
                client=client,
                local_path=local_manifest_path(out_dir, run_id),
                bucket=bkt,
                key=manifest_key(run_id),
            )

    return run_report, manifest


def upload_clip_timing_shards(
    client: Any,
    *,
    bucket: str,
    run_id: str,
    clip_entries: list[dict[str, Any]],
    host: str | None = None,
) -> None:
    """Upload one timing.json per clip (worker shard)."""
    import socket

    host = host or socket.gethostname()
    for entry in clip_entries:
        cid = int(entry["clip_id"])
        shard = {
            "run_id": run_id,
            "host": host,
            "clips": [entry],
        }
        key = clip_timing_key(run_id, cid)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=(json.dumps(shard, indent=2) + "\n").encode("utf-8"),
            ContentType="application/json",
        )
