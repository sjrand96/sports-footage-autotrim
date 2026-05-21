"""Publish feature-extraction runs to Weights & Biases (S3 reference artifacts)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from feature_extraction.s3_layout import (
    default_bucket,
    feature_extraction_prefix,
    local_manifest_path,
    local_run_report_path,
    local_timings_path,
    run_s3_uri,
)

logger = logging.getLogger(__name__)

# https://wandb.ai/cs348k-sports-footage-autotrim/volleyball-playtime
DEFAULT_WANDB_ENTITY = "cs348k-sports-footage-autotrim"
DEFAULT_WANDB_PROJECT = "volleyball-playtime"
FEATURE_ARTIFACT_NAME = "playing-features"


def wandb_entity() -> str:
    return os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY).strip() or DEFAULT_WANDB_ENTITY


def wandb_project() -> str:
    return os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT).strip() or DEFAULT_WANDB_PROJECT


def wandb_publish_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return bool(os.environ.get("WANDB_API_KEY", "").strip())


def publish_feature_run_artifact(
    *,
    run_id: str,
    manifest: dict[str, Any],
    out_dir: Path,
    bucket: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    artifact_name: str = FEATURE_ARTIFACT_NAME,
) -> str | None:
    """Register ``playing-features`` dataset artifact (S3 reference + local sidecars).

    Returns the W&B artifact qualified name (``entity/project/name:alias``), or None if skipped.
    """
    if not wandb_publish_enabled(True):
        logger.info("wandb: WANDB_API_KEY not set; skip feature artifact publish")
        return None

    import wandb

    bkt = bucket or str(manifest.get("s3_bucket") or default_bucket())
    s3_prefix_uri = run_s3_uri(bkt, run_id)
    entity = entity or wandb_entity()
    project = project or wandb_project()

    run_path = out_dir / run_id
    manifest_path = local_manifest_path(out_dir, run_id)
    timings_path = local_timings_path(out_dir, run_id)
    run_report_path = local_run_report_path(out_dir, run_id)

    split_meta = manifest.get("split_meta") if isinstance(manifest.get("split_meta"), dict) else {}
    timing_summary = manifest.get("timing_summary") if isinstance(manifest.get("timing_summary"), dict) else {}

    metadata: dict[str, Any] = {
        "feature_run_id": run_id,
        "extractor_version": manifest.get("extractor_version"),
        "feature_schema_version": manifest.get("feature_schema_version"),
        "split_method": manifest.get("split_method"),
        "split_seed": split_meta.get("split_seed"),
        "test_fraction": split_meta.get("test_fraction"),
        "s3_bucket": bkt,
        "s3_prefix": manifest.get("s3_prefix") or feature_extraction_prefix(run_id) + "/",
        "s3_run_uri": manifest.get("s3_run_uri") or s3_prefix_uri,
        "n_clips_ok": timing_summary.get("n_clips_ok"),
        "n_clips_failed": timing_summary.get("n_clips_failed"),
        "feature_columns": manifest.get("feature_columns"),
    }

    wb_run = wandb.init(
        entity=entity,
        project=project,
        job_type="publish_features",
        name=f"features-{run_id}",
        tags=["feature-extraction", "dataset"],
        config=metadata,
        reinit="finish_previous",
    )
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=f"Per-frame feature parquets for run {run_id} (S3 reference)",
            metadata=metadata,
        )

        artifact.add_reference(uri=s3_prefix_uri, name="feature_extraction", checksum=False)

        if manifest_path.is_file():
            artifact.add_file(str(manifest_path), name="manifest.json")
        if timings_path.is_file():
            artifact.add_file(str(timings_path), name="timings.json")
        if run_report_path.is_file():
            artifact.add_file(str(run_report_path), name="run_report.json")

        wb_run.log_artifact(artifact, aliases=[run_id])
        qualified = f"{entity}/{project}/{artifact_name}:{run_id}"
        logger.info("wandb: logged artifact %s (S3 ref %s)", qualified, s3_prefix_uri)
        return qualified
    finally:
        wb_run.finish()


def maybe_publish_feature_run_artifact(
    *,
    run_id: str,
    manifest: dict[str, Any],
    out_dir: Path,
    bucket: str | None = None,
    publish: bool | None = None,
) -> str | None:
    if not wandb_publish_enabled(publish):
        return None
    run_report = manifest.get("run_report") if isinstance(manifest.get("run_report"), dict) else {}
    if run_report.get("status") == "failed" and int(run_report.get("n_success") or 0) == 0:
        logger.info("wandb: skip publish for failed feature run (no successes)")
        return None
    return publish_feature_run_artifact(
        run_id=run_id,
        manifest=manifest,
        out_dir=out_dir,
        bucket=bucket,
    )


def main() -> int:
    """One-off: publish an existing local/S3 feature run to W&B (e.g. after a killed fanout)."""
    import argparse

    from dotenv import load_dotenv

    from feature_extraction.s3_layout import local_manifest_path

    parser = argparse.ArgumentParser(description="Publish feature-extraction run as W&B dataset artifact.")
    parser.add_argument("--run-id", required=True, help="feature_extraction run_id")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[0] / "_runs",
        help="Local runs root containing manifest.json",
    )
    parser.add_argument("--bucket", default=None, help="S3 bucket override")
    parser.add_argument("--entity", default=None, help=f"WANDB entity (default {DEFAULT_WANDB_ENTITY})")
    parser.add_argument("--project", default=None, help=f"WANDB project (default {DEFAULT_WANDB_PROJECT})")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    runs_root = args.runs_root.expanduser().resolve()
    manifest_path = local_manifest_path(runs_root, args.run_id)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualified = publish_feature_run_artifact(
        run_id=args.run_id,
        manifest=manifest,
        out_dir=runs_root,
        bucket=args.bucket,
        entity=args.entity,
        project=args.project,
    )
    if qualified:
        print(f"Published: {qualified}")
        print(f"https://wandb.ai/{wandb_entity()}/{wandb_project()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
