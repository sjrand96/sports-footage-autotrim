"""Run manifest and run report (written even on partial failure)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from feature_extraction.core.feature_columns import FEATURE_COLUMNS, PARQUET_COLUMNS
from feature_extraction.core.version import EXTRACTOR_VERSION, FEATURE_SCHEMA_VERSION

RunStatus = Literal["ok", "partial", "failed"]


@dataclass
class ClipFailure:
    clip_id: int
    source_id: str
    clip_index: int
    stage: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source_id": self.source_id,
            "clip_index": self.clip_index,
            "stage": self.stage,
            "error": self.error,
        }


@dataclass
class ClipSuccess:
    clip_id: int
    source_id: str
    clip_index: int
    split: str
    n_rows: int
    output_path: str
    output_s3_uri: str | None = None
    timings_sec: dict[str, float] | None = None
    derived: dict[str, float] | None = None
    source_fps: float | None = None
    n_source_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "clip_id": self.clip_id,
            "source_id": self.source_id,
            "clip_index": self.clip_index,
            "split": self.split,
            "n_rows": self.n_rows,
            "output_path": self.output_path,
        }
        if self.output_s3_uri:
            d["output_s3_uri"] = self.output_s3_uri
        if self.timings_sec:
            d["timings_sec"] = dict(self.timings_sec)
        if self.derived:
            d["derived"] = dict(self.derived)
        if self.source_fps is not None:
            d["source_fps"] = self.source_fps
        if self.n_source_frames is not None:
            d["n_source_frames"] = self.n_source_frames
        return d


@dataclass
class RunReport:
    failures: list[ClipFailure] = field(default_factory=list)
    successes: list[ClipSuccess] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return len(self.successes)

    @property
    def n_failed(self) -> int:
        return len(self.failures)

    @property
    def status(self) -> RunStatus:
        if self.n_success == 0 and self.n_failed > 0:
            return "failed"
        if self.n_failed > 0:
            return "partial"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "failures": [f.to_dict() for f in self.failures],
            "successes": [s.to_dict() for s in self.successes],
        }


def try_git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    run_id: str,
    split_meta: dict[str, Any],
    run_report: RunReport,
    out_dir: Path,
    label_fps: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "parquet_columns": list(PARQUET_COLUMNS),
        "sample_policy": "full_source_fps",
        "label_fps": float(label_fps),
        "output_local_dir": str(out_dir.resolve()),
        "git_sha": try_git_sha(),
        **split_meta,
        "run_report": run_report.to_dict(),
    }
    if extra:
        body.update(extra)
    return body


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_run_report(path: Path, run_report: RunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_report.to_dict(), indent=2) + "\n", encoding="utf-8")


def print_run_summary(run_report: RunReport, *, max_failures: int = 10) -> None:
    print("\n=== feature extraction run summary ===")
    print(f"status: {run_report.status}")
    print(f"successes: {run_report.n_success}")
    print(f"failures:  {run_report.n_failed}")
    if run_report.failures:
        print("\nFailures:")
        for f in run_report.failures[:max_failures]:
            print(f"  - {f.source_id}_{f.clip_index:03d} clip_id={f.clip_id} [{f.stage}]: {f.error}")
        if len(run_report.failures) > max_failures:
            print(f"  ... and {len(run_report.failures) - max_failures} more (see run_report.json)")
    print("======================================\n")
