#!/usr/bin/env python3
"""Plan, fan out, wait, and finalize a parallel Fargate feature-extraction run (one ECS task per clip)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_extraction.clip_split import assign_train_test  # noqa: E402
from feature_extraction.core.clip_selection import ClipSpec, list_eligible_clips  # noqa: E402
from feature_extraction.core.paths import clip_stem  # noqa: E402
from feature_extraction.finalize_run import finalize_run_from_plan  # noqa: E402
from feature_extraction.manifest import print_run_summary  # noqa: E402
from feature_extraction.s3_layout import (  # noqa: E402
    default_bucket,
    default_region,
    feature_extraction_prefix,
    parquet_key,
)

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER = "default"
DEFAULT_TASK_DEFINITION = "default-sports-footage-fe-worker-e01e"
DEFAULT_CONTAINER = "Main"
DEFAULT_SUBNET = "subnet-00f575df9d041d3ab"
DEFAULT_SECURITY_GROUP = "sg-0b23a01f27c7eaf2c"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _current_git_sha_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _image_tag_from_uri(image_uri: str) -> str | None:
    if ":" not in image_uri:
        return None
    return image_uri.rsplit(":", 1)[1].strip() or None


def validate_task_definition_image(
    *,
    task_definition: str,
    container_name: str,
    region: str,
) -> None:
    """Fail fast when task definition image tag is stale."""
    import boto3

    ecs = boto3.client("ecs", region_name=region)
    td = ecs.describe_task_definition(taskDefinition=task_definition)["taskDefinition"]
    containers = td.get("containerDefinitions") or []
    container = next((c for c in containers if c.get("name") == container_name), None)
    if container is None and containers:
        container = containers[0]
    if container is None:
        raise SystemExit(f"task definition has no containers: {task_definition}")

    image = str(container.get("image") or "")
    tag = _image_tag_from_uri(image)
    git_tag = _current_git_sha_short()
    logger.info("task definition image: %s", image)
    if tag == "latest":
        logger.info("task definition uses :latest (auto-newest mode)")
        return
    if git_tag and tag and tag != git_tag:
        logger.warning(
            "task definition image tag != local git SHA (continuing).\n"
            "  task definition tag: %s\n"
            "  local git SHA:       %s",
            tag,
            git_tag,
        )


def _get_db():
    from src import db as db_helpers

    return db_helpers, db_helpers.get_supabase_client()


def _plan_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / run_id / "run_plan.json"


def build_plan(
    *,
    run_id: str,
    specs: list[ClipSpec],
    test_fraction: float,
    split_seed: int,
    max_frames: int | None,
    label_fps: float,
) -> dict[str, Any]:
    train_clips, test_clips, split_meta = assign_train_test(
        specs,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    train_ids = {c.clip_id for c in train_clips}
    clips = []
    for spec in specs:
        split = "train" if spec.clip_id in train_ids else "test"
        clips.append(
            {
                "clip_id": spec.clip_id,
                "source_id": spec.source_id,
                "clip_index": spec.clip_index,
                "split": split,
                "s3_uri": spec.s3_uri,
            }
        )
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "split_meta": split_meta,
        "max_frames": max_frames,
        "label_fps": label_fps,
        "clips": clips,
        "tasks": [],
    }


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_vcpu_quota_failure(failures: list[dict[str, Any]]) -> bool:
    text = str(failures).lower()
    return "vcpu" in text and "concurrently" in text


def _parquet_on_s3(
    s3: Any,
    *,
    bucket: str,
    run_id: str,
    clip: dict[str, Any],
) -> bool:
    stem = clip_stem(str(clip["source_id"]), int(clip["clip_index"]))
    key = parquet_key(run_id, str(clip["split"]), stem)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def pending_clips_for_resume(
    plan: dict[str, Any],
    *,
    bucket: str,
    region: str,
) -> tuple[list[dict[str, Any]], int]:
    """Clips in ``plan`` with no parquet yet on S3 for this ``run_id``."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    run_id = str(plan["run_id"])
    all_clips = list(plan["clips"])
    pending = [c for c in all_clips if not _parquet_on_s3(s3, bucket=bucket, run_id=run_id, clip=c)]
    return pending, len(all_clips) - len(pending)


def merge_task_records(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {int(t["clip_id"]): t for t in existing if t.get("clip_id") is not None}
    for t in new:
        if t.get("clip_id") is None:
            continue
        by_id[int(t["clip_id"])] = t
    return [by_id[k] for k in sorted(by_id)]


def _worker_command(
    *,
    run_id: str,
    clip_id: int,
    split: str,
    max_frames: int | None,
) -> list[str]:
    cmd = [
        "feature_extraction/job.py",
        "--out-dir",
        "/tmp/fe",
        "--run-id",
        run_id,
        "--clip-id",
        str(clip_id),
        "--force-split",
        split,
        "--upload-s3",
        "--upload-parquet-only",
        "--delete-local-clip-after",
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    return cmd


def start_tasks(
    plan: dict[str, Any],
    *,
    cluster: str,
    task_definition: str,
    container_name: str,
    subnet: str,
    security_group: str,
    region: str,
    concurrency: int,
    clips_to_run: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    import boto3

    ecs = boto3.client("ecs", region_name=region)
    run_id = plan["run_id"]
    max_frames = plan.get("max_frames")
    pending = list(clips_to_run if clips_to_run is not None else plan["clips"])
    finished: list[dict[str, Any]] = []
    running: dict[str, dict[str, Any]] = {}

    while pending or running:
        while pending and len(running) < concurrency:
            clip = pending.pop(0)
            overrides = {
                "containerOverrides": [
                    {
                        "name": container_name,
                        "command": _worker_command(
                            run_id=run_id,
                            clip_id=int(clip["clip_id"]),
                            split=str(clip["split"]),
                            max_frames=max_frames,
                        ),
                    }
                ]
            }
            resp = ecs.run_task(
                cluster=cluster,
                launchType="FARGATE",
                taskDefinition=task_definition,
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": [subnet],
                        "securityGroups": [security_group],
                        "assignPublicIp": "ENABLED",
                    }
                },
                overrides=overrides,
            )
            failures = resp.get("failures") or []
            if failures:
                if _is_vcpu_quota_failure(failures):
                    logger.warning(
                        "run_task deferred clip_id=%s (vCPU quota); will retry when a slot frees",
                        clip["clip_id"],
                    )
                    pending.insert(0, clip)
                else:
                    logger.error("run_task failed clip_id=%s: %s", clip["clip_id"], failures)
                    finished.append(
                        {
                            "clip_id": clip["clip_id"],
                            "task_arn": None,
                            "exit_code": 1,
                            "error": str(failures),
                            "parquet_on_s3": False,
                            "status": "STOPPED",
                        }
                    )
                continue
            task_arn = resp["tasks"][0]["taskArn"]
            logger.info("started clip_id=%s task=%s", clip["clip_id"], task_arn)
            running[task_arn] = {
                "clip_id": clip["clip_id"],
                "task_arn": task_arn,
                "status": "RUNNING",
            }

        if not running:
            if pending:
                time.sleep(15)
            else:
                break

        time.sleep(15)
        desc = ecs.describe_tasks(cluster=cluster, tasks=list(running.keys()))
        for task_info in desc.get("tasks") or []:
            arn = task_info["taskArn"]
            entry = running[arn]
            last = task_info.get("lastStatus")
            entry["last_status"] = last
            if last in ("RUNNING", "PENDING", "PROVISIONING", "ACTIVATING"):
                continue
            container = (task_info.get("containers") or [{}])[0]
            entry["exit_code"] = container.get("exitCode")
            entry["stopped_reason"] = task_info.get("stoppedReason")
            entry["status"] = "STOPPED"
            logger.info(
                "finished clip_id=%s exit=%s reason=%s",
                entry["clip_id"],
                entry.get("exit_code"),
                entry.get("stopped_reason"),
            )
            finished.append(entry)
            del running[arn]

    return finished


def enrich_plan_from_s3(plan: dict[str, Any], *, bucket: str, region: str) -> dict[str, Any]:
    import boto3

    s3 = boto3.client("s3", region_name=region)
    run_id = str(plan["run_id"])
    prev = {int(t["clip_id"]): t for t in plan.get("tasks") or [] if t.get("clip_id") is not None}
    tasks: list[dict[str, Any]] = []
    for clip in plan["clips"]:
        cid = int(clip["clip_id"])
        on_s3 = _parquet_on_s3(s3, bucket=bucket, run_id=run_id, clip=clip)
        task = dict(prev.get(cid, {}))
        task["clip_id"] = cid
        task["parquet_on_s3"] = on_s3
        if on_s3:
            task["exit_code"] = 0
            task.setdefault("status", "STOPPED")
        elif task.get("exit_code") is None:
            task["exit_code"] = 1
        tasks.append(task)
    plan["tasks"] = tasks
    return plan


def _run_workers_and_merge_tasks(
    plan: dict[str, Any],
    *,
    cluster: str,
    task_definition: str,
    container_name: str,
    subnet: str,
    security_group: str,
    region: str,
    concurrency: int,
    clips_to_run: list[dict[str, Any]],
) -> dict[str, Any]:
    session_tasks = start_tasks(
        plan,
        cluster=cluster,
        task_definition=task_definition,
        container_name=container_name,
        subnet=subnet,
        security_group=security_group,
        region=region,
        concurrency=concurrency,
        clips_to_run=clips_to_run,
    )
    plan["tasks"] = merge_task_records(plan.get("tasks") or [], session_tasks)
    return plan


def _finalize_plan(
    plan: dict[str, Any],
    *,
    out_dir: Path,
    bucket: str,
    region: str,
    label_fps: float,
    plan_path: Path,
    wandb_publish: bool | None,
) -> int:
    plan = enrich_plan_from_s3(plan, bucket=bucket, region=region)
    write_plan(plan_path, plan)
    run_report, manifest = finalize_run_from_plan(
        plan=plan,
        out_dir=out_dir,
        bucket=bucket,
        region=region,
        label_fps=label_fps,
        upload_s3=True,
        publish_wandb=wandb_publish,
    )
    print_run_summary(run_report)
    if manifest.get("wandb_feature_artifact"):
        logger.info("wandb feature artifact: %s", manifest["wandb_feature_artifact"])
    return 0 if run_report.n_failed == 0 else 1


def _resume_pending_or_log(
    plan: dict[str, Any],
    *,
    bucket: str,
    region: str,
    resume: bool,
) -> list[dict[str, Any]] | None:
    """Return clips to run, or None if resume mode and every clip already has a parquet on S3."""
    if not resume:
        return list(plan["clips"])
    pending, n_done = pending_clips_for_resume(plan, bucket=bucket, region=region)
    total = len(plan["clips"])
    logger.info(
        "resume: %d/%d clips already on S3; %d to run",
        n_done,
        total,
        len(pending),
    )
    if not pending:
        logger.info("resume: nothing left to run")
        return None
    return pending


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel ECS fan-out for feature extraction.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "feature_extraction" / "_runs")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--label-fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--bucket", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--cluster", type=str, default=DEFAULT_CLUSTER)
    p.add_argument("--task-definition", type=str, default=DEFAULT_TASK_DEFINITION)
    p.add_argument("--container-name", type=str, default=DEFAULT_CONTAINER)
    p.add_argument("--subnet", type=str, default=DEFAULT_SUBNET)
    p.add_argument("--security-group", type=str, default=DEFAULT_SECURITY_GROUP)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--start-only", action="store_true", help="Requires existing run_plan.json")
    p.add_argument("--finalize-only", action="store_true", help="Merge shards + upload manifest")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip clips that already have a parquet on S3 for this run_id; merge task history",
    )
    p.add_argument(
        "--wandb-publish",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish playing-features W&B artifact after finalize (default: on if WANDB_API_KEY set)",
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

    bucket = args.bucket or default_bucket()
    region = args.region or default_region()
    out_dir = args.out_dir.expanduser().resolve()
    run_id = args.run_id or _new_run_id()
    plan_path = _plan_path(out_dir, run_id)

    if args.finalize_only:
        if not plan_path.is_file():
            logger.error("plan not found: %s", plan_path)
            return 1
        plan = load_plan(plan_path)
        return _finalize_plan(
            plan,
            out_dir=out_dir,
            bucket=bucket,
            region=region,
            label_fps=float(plan.get("label_fps") or args.label_fps),
            plan_path=plan_path,
            wandb_publish=args.wandb_publish,
        )

    if args.start_only:
        if not plan_path.is_file():
            logger.error("plan not found: %s", plan_path)
            return 1
        plan = load_plan(plan_path)
        clips_to_run = _resume_pending_or_log(plan, bucket=bucket, region=region, resume=args.resume)
        if clips_to_run is None:
            return _finalize_plan(
                plan,
                out_dir=out_dir,
                bucket=bucket,
                region=region,
                label_fps=float(plan.get("label_fps") or args.label_fps),
                plan_path=plan_path,
                wandb_publish=args.wandb_publish,
            )
        validate_task_definition_image(
            task_definition=args.task_definition,
            container_name=args.container_name,
            region=region,
        )
        plan = _run_workers_and_merge_tasks(
            plan,
            cluster=args.cluster,
            task_definition=args.task_definition,
            container_name=args.container_name,
            subnet=args.subnet,
            security_group=args.security_group,
            region=region,
            concurrency=args.concurrency,
            clips_to_run=clips_to_run,
        )
        write_plan(plan_path, plan)
        return _finalize_plan(
            plan,
            out_dir=out_dir,
            bucket=bucket,
            region=region,
            label_fps=float(plan.get("label_fps") or args.label_fps),
            plan_path=plan_path,
            wandb_publish=args.wandb_publish,
        )

    db_helpers, db_client = _get_db()
    specs = list_eligible_clips(db_helpers, db_client)
    if args.max_clips is not None:
        specs = specs[: args.max_clips]
    if not specs:
        logger.error("no eligible clips")
        return 1

    plan = build_plan(
        run_id=run_id,
        specs=specs,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        max_frames=args.max_frames,
        label_fps=args.label_fps,
    )
    write_plan(plan_path, plan)
    logger.info("wrote plan %s (%d clips)", plan_path, len(plan["clips"]))
    for c in plan["clips"]:
        logger.info("  clip_id=%s %s_%03d -> %s", c["clip_id"], c["source_id"], c["clip_index"], c["split"])

    if args.plan_only:
        if args.resume:
            pending, n_done = pending_clips_for_resume(plan, bucket=bucket, region=region)
            print(f"\nResume preview: {n_done}/{len(plan['clips'])} complete on S3, {len(pending)} would run")
        print(f"\nRun id: {run_id}")
        print(f"Plan:   {plan_path}")
        print(f"S3:     s3://{bucket}/{feature_extraction_prefix(run_id)}/")
        print("\nNext: re-push amd64 image if needed, then:")
        print(f"  python feature_extraction/aws/run_fanout.py --run-id {run_id} --start-only --resume")
        return 0

    clips_to_run = _resume_pending_or_log(plan, bucket=bucket, region=region, resume=args.resume)
    if clips_to_run is None:
        write_plan(plan_path, plan)
        rc = _finalize_plan(
            plan,
            out_dir=out_dir,
            bucket=bucket,
            region=region,
            label_fps=args.label_fps,
            plan_path=plan_path,
            wandb_publish=args.wandb_publish,
        )
        print(f"\nS3: s3://{bucket}/{feature_extraction_prefix(run_id)}/")
        return rc

    validate_task_definition_image(
        task_definition=args.task_definition,
        container_name=args.container_name,
        region=region,
    )
    plan = _run_workers_and_merge_tasks(
        plan,
        cluster=args.cluster,
        task_definition=args.task_definition,
        container_name=args.container_name,
        subnet=args.subnet,
        security_group=args.security_group,
        region=region,
        concurrency=args.concurrency,
        clips_to_run=clips_to_run,
    )
    write_plan(plan_path, plan)
    rc = _finalize_plan(
        plan,
        out_dir=out_dir,
        bucket=bucket,
        region=region,
        label_fps=args.label_fps,
        plan_path=plan_path,
        wandb_publish=args.wandb_publish,
    )
    print(f"\nS3: s3://{bucket}/{feature_extraction_prefix(run_id)}/")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
