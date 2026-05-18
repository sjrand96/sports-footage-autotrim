"""Wall-clock timings for feature-extraction runs (``timings.json`` sidecar)."""

from __future__ import annotations

import json
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from feature_extraction.core.version import EXTRACTOR_VERSION
from feature_extraction.manifest import ClipFailure, ClipSuccess, RunReport


@dataclass
class ClipTimer:
    """Accumulate seconds per named stage for one clip."""

    timings_sec: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            self.timings_sec[name] = self.timings_sec.get(name, 0.0) + (perf_counter() - t0)


def derived_timing_metrics(*, extract_sec: float, n_rows: int) -> dict[str, float]:
    if n_rows <= 0 or extract_sec <= 0:
        return {}
    return {
        "sec_per_frame": round(extract_sec / n_rows, 6),
        "rows_per_sec": round(n_rows / extract_sec, 4),
    }


def clip_timing_failure_entry(failure: ClipFailure) -> dict[str, Any]:
    return {
        "clip_id": failure.clip_id,
        "source_id": failure.source_id,
        "clip_index": failure.clip_index,
        "status": "failed",
        "failed_stage": failure.stage,
        "error": failure.error,
    }


def clip_timing_entry(success: ClipSuccess) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "clip_id": success.clip_id,
        "source_id": success.source_id,
        "clip_index": success.clip_index,
        "split": success.split,
        "status": "ok",
        "n_rows": success.n_rows,
        "source_fps": success.source_fps,
        "n_source_frames": success.n_source_frames,
        "timings_sec": {k: round(v, 4) for k, v in (success.timings_sec or {}).items()},
    }
    if success.derived:
        entry["derived"] = dict(success.derived)
    return entry


def clip_entries_from_run_report(run_report: RunReport) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = [clip_timing_entry(s) for s in run_report.successes]
    clips.extend(clip_timing_failure_entry(f) for f in run_report.failures)
    return clips


def build_timings_document(
    *,
    run_id: str,
    run_report: RunReport,
    started_at: datetime,
    finished_at: datetime,
    max_frames: int | None,
    upload_sec: float | None = None,
    clip_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clips = clip_entries if clip_entries is not None else clip_entries_from_run_report(run_report)

    totals: dict[str, float] = {}
    for key in ("download", "calibration", "extract", "frames_upload", "labels", "parquet_write", "clip_total"):
        total = sum((c.get("timings_sec") or {}).get(key, 0.0) for c in clips if c.get("status") == "ok")
        if total > 0:
            totals[key] = round(total, 4)

    extract_times = [
        (c.get("derived") or {}).get("sec_per_frame")
        for c in clips
        if c.get("status") == "ok" and (c.get("derived") or {}).get("sec_per_frame")
    ]
    wall = (finished_at - started_at).total_seconds()

    if upload_sec is not None and upload_sec > 0:
        totals["upload"] = round(upload_sec, 4)
    totals["wall_clock"] = round(wall, 4)

    doc: dict[str, Any] = {
        "run_id": run_id,
        "extractor_version": EXTRACTOR_VERSION,
        "sample_policy": "full_source_fps" if max_frames is None else "capped_frames",
        "max_frames": max_frames,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_clock_sec": round(wall, 4),
        "host": socket.gethostname(),
        "clips": clips,
        "totals_sec": totals,
    }
    n_ok = sum(1 for c in clips if c.get("status") == "ok")
    n_failed = sum(1 for c in clips if c.get("status") == "failed")
    if extract_times:
        doc["summary"] = {
            "mean_sec_per_frame_extract": round(sum(extract_times) / len(extract_times), 6),
            "n_clips_ok": n_ok,
            "n_clips_failed": n_failed,
        }
    elif clips:
        doc["summary"] = {
            "n_clips_ok": n_ok,
            "n_clips_failed": n_failed,
        }
    return doc


def merge_clip_timing_shards(
    shards: list[dict[str, Any]],
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    max_frames: int | None,
    upload_sec: float | None = None,
) -> dict[str, Any]:
    """Build run-level ``timings.json`` from per-clip worker shards."""
    clips: list[dict[str, Any]] = []
    for shard in shards:
        clips.extend(shard.get("clips") or [])
    clips.sort(key=lambda c: (c.get("clip_id", 0), c.get("clip_index", 0)))
    empty = RunReport()
    return build_timings_document(
        run_id=run_id,
        run_report=empty,
        started_at=started_at,
        finished_at=finished_at,
        max_frames=max_frames,
        upload_sec=upload_sec,
        clip_entries=clips,
    )


def timing_summary_for_manifest(timings_doc: dict[str, Any]) -> dict[str, Any]:
    summary = timings_doc.get("summary") or {}
    return {
        "wall_clock_sec": timings_doc.get("wall_clock_sec"),
        "mean_sec_per_frame_extract": summary.get("mean_sec_per_frame_extract"),
        "n_clips_ok": summary.get("n_clips_ok", 0),
        "n_clips_failed": summary.get("n_clips_failed", 0),
        "timings_uri_suffix": "timings.json",
    }


def write_timings(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def log_clip_timing(success: ClipSuccess, *, log: Any) -> None:
    t = success.timings_sec or {}
    extract_s = t.get("extract", 0.0)
    spf = (success.derived or {}).get("sec_per_frame")
    spf_s = f"{spf:.3f} s/frame" if spf is not None else "n/a"
    log.info(
        "timing %s_%03d clip_id=%s extract=%.1fs (%s, %d rows)",
        success.source_id,
        success.clip_index,
        success.clip_id,
        extract_s,
        spf_s,
        success.n_rows,
    )
