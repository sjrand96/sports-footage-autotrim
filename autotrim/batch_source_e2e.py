#!/usr/bin/env python3
"""Batch end-to-end autotrim: YOLO extract @ fps → XGB → postproc → ffmpeg."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autotrim.trim_clip import (  # noqa: E402
    DEFAULT_CV_RUN,
    ensemble_predict_proba,
    export_cut_video,
    load_cv_threshold,
    load_features,
    load_fold_models,
    prepare_feature_matrix,
    probe_video_duration_sec,
    segments_to_time_intervals,
)
from eval.segment_metrics import extract_segments_from_probs  # noqa: E402
from feature_extraction.core.feature_columns import active_feature_columns  # noqa: E402

DEFAULT_PARQUET_ROOT = REPO_ROOT / "feature_extraction/_runs/all_clips_features_v2/parquet"
DEFAULT_CLIP_ROOT = REPO_ROOT / "cv-pipeline/pose-detection/media/clips"


def _clip_key_frame(source_id: str, clip_index: int) -> str:
    return f"{source_id}_{int(clip_index):03d}"


def segment_params_for_extract_fps(extract_fps: float, *, base_fps: float = 30.0) -> dict[str, int]:
    scale = extract_fps / base_fps
    return {
        "segment_window_frames": max(1, round(15 * scale)),
        "segment_max_gap_frames": max(1, round(45 * scale)),
        "segment_min_segment_frames": max(1, round(90 * scale)),
    }


def _resolve_mp4(clip_key: str, *, clip_root: Path, download: bool) -> Path:
    source_id, _ = clip_key.rsplit("_", 1)
    local = clip_root / source_id / f"{clip_key}.mp4"
    if local.is_file():
        return local
    if not download:
        raise FileNotFoundError(f"missing mp4: {local}")
    uri = f"s3://sports-footage-autotrim-bucket/clips/{source_id}/{clip_key}.mp4"
    fetch = REPO_ROOT / "cv-pipeline/pose-detection/fetch_s3_clip.py"
    subprocess.run([sys.executable, str(fetch), uri], check=True)
    if not local.is_file():
        raise FileNotFoundError(f"download failed: {local}")
    return local


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-id", required=True)
    p.add_argument("--calibration-json", type=Path, required=True)
    p.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    p.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    p.add_argument("--cv-run", type=Path, default=DEFAULT_CV_RUN)
    p.add_argument("--extract-fps", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap YOLO sampled rows per clip (smoke tests only).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel clip workers (processes; each loads YOLO + XGB). Default 1 = serial.",
    )
    return p.parse_args()


def _stride_for_target_fps(target_fps: float, *, base_fps: int = 30) -> int:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps >= base_fps:
        return 1
    return max(1, round(base_fps / target_fps))


def _subsample_parquet(path: Path, *, extract_fps: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "clip_key" not in df.columns:
        df = df.copy()
        df["clip_key"] = _clip_key_frame(str(df["source_id"].iloc[0]), int(df["clip_index"].iloc[0]))
    stride = _stride_for_target_fps(extract_fps)
    if stride <= 1:
        return df
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("clip_key", sort=False):
        ordered = group.sort_values("frame_idx", kind="stable")
        parts.append(ordered.iloc[::stride])
    return pd.concat(parts, ignore_index=True)


def _gt_labels_from_parquet(path: Path, *, extract_fps: float) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return _subsample_parquet(path, extract_fps=extract_fps)


def _metrics_from_gt(gt_df: pd.DataFrame | None, probs, threshold: float) -> dict[str, float]:
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    if gt_df is None:
        return {}
    yt = gt_df["is_playing"].astype(int).to_numpy()
    yp = (probs >= threshold).astype(int)
    return {
        "f2": float(fbeta_score(yt, yp, beta=2, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "playing_gt_frac": float(yt.mean()),
    }


def process_clip_e2e(
    *,
    clip_key: str,
    mp4: Path,
    calibration_json: Path,
    out_dir: Path,
    cv_run: Path,
    feature_columns: list[str],
    threshold: float,
    extract_fps: float,
    seg_params: dict[str, int],
    yolo_device: str,
    gt_df: pd.DataFrame | None,
    max_frames: int | None = None,
    parquet_path: Path | None = None,
) -> dict:
    t_total = perf_counter()
    timings: dict[str, float] = {}

    t0 = perf_counter()
    df, source_fps, frame_stride = load_features(
        input_path=mp4,
        features_parquet=None,
        calibration_json=calibration_json,
        calibration_npz=None,
        extract_fps=extract_fps,
        weights="yolov8s-pose.pt",
        yolo_device=yolo_device,
        max_frames=max_frames,
    )
    timings["extract"] = round(perf_counter() - t0, 3)

    if gt_df is None and parquet_path is not None:
        gt_df = _gt_labels_from_parquet(parquet_path, extract_fps=extract_fps)

    t0 = perf_counter()
    # Load XGB only after YOLO extract (macOS segfault if reversed).
    models = load_fold_models(cv_run)
    X = prepare_feature_matrix(df, feature_columns)
    probs = ensemble_predict_proba(models, X)
    segments, _ = extract_segments_from_probs(
        probs,
        threshold=threshold,
        window_frames=seg_params["segment_window_frames"],
        aggregation="mean",
        max_gap_frames=seg_params["segment_max_gap_frames"],
        min_segment_frames=seg_params["segment_min_segment_frames"],
    )
    video_duration = probe_video_duration_sec(mp4)
    intervals = segments_to_time_intervals(
        segments, df, source_fps=source_fps, video_duration_sec=video_duration
    )
    timings["infer"] = round(perf_counter() - t0, 3)

    out_mp4 = out_dir / "videos" / f"{clip_key}_playing.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    if intervals:
        export_cut_video(input_path=mp4, output_path=out_mp4, intervals=intervals)
        playing_sec = sum(iv["end"] - iv["start"] for iv in intervals)
    else:
        out_mp4 = None
        playing_sec = 0.0
    timings["ffmpeg"] = round(perf_counter() - t0, 3)
    timings["total"] = round(perf_counter() - t_total, 3)

    metrics: dict[str, float] = {}
    if gt_df is not None and len(gt_df) == len(df):
        metrics = _metrics_from_gt(gt_df, probs, threshold)

    sidecar = {
        "clip_key": clip_key,
        "extract_fps": extract_fps,
        "frame_stride": frame_stride,
        "source_fps": source_fps,
        "n_rows": len(df),
        "metrics": metrics,
        "n_segments": len(intervals),
        "playing_sec": playing_sec,
        "intervals": intervals,
        "output_mp4": str(out_mp4) if out_mp4 else None,
        "timings_sec": timings,
    }
    (out_dir / "sidecars").mkdir(parents=True, exist_ok=True)
    (out_dir / "sidecars" / f"{clip_key}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(
        f"  {clip_key}: extract={timings['extract']:.1f}s infer={timings['infer']:.1f}s "
        f"ffmpeg={timings['ffmpeg']:.1f}s total={timings['total']:.1f}s",
        flush=True,
    )
    return sidecar


def _worker(job: dict[str, Any]) -> dict:
    """Process-pool entry: each worker runs full clip pipeline."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    feature_columns = active_feature_columns("all")
    return process_clip_e2e(
        clip_key=job["clip_key"],
        mp4=Path(job["mp4"]),
        calibration_json=Path(job["calibration_json"]),
        out_dir=Path(job["out_dir"]),
        cv_run=Path(job["cv_run"]),
        feature_columns=feature_columns,
        threshold=job["threshold"],
        extract_fps=job["extract_fps"],
        seg_params=job["seg_params"],
        yolo_device=job["yolo_device"],
        gt_df=None,
        max_frames=job.get("max_frames"),
        parquet_path=Path(job["parquet_path"]),
    )


def _build_jobs(
    *,
    parquets: list[Path],
    args: argparse.Namespace,
    out_dir: Path,
    cal: Path,
    threshold: float,
    seg_params: dict[str, int],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for pq in parquets:
        df_peek = pd.read_parquet(pq, columns=["source_id", "clip_index"])
        clip_key = _clip_key_frame(str(df_peek["source_id"].iloc[0]), int(df_peek["clip_index"].iloc[0]))
        mp4 = _resolve_mp4(clip_key, clip_root=args.clip_root, download=not args.no_download)
        jobs.append(
            {
                "clip_key": clip_key,
                "mp4": str(mp4),
                "parquet_path": str(pq),
                "calibration_json": str(cal),
                "out_dir": str(out_dir),
                "cv_run": str(args.cv_run.resolve()),
                "threshold": threshold,
                "extract_fps": args.extract_fps,
                "seg_params": seg_params,
                "yolo_device": args.yolo_device,
                "max_frames": args.max_frames,
            }
        )
    return jobs


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    out_dir = (args.output_dir or (REPO_ROOT / "autotrim/_runs" / args.source_id / "2fps_e2e")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = args.calibration_json.expanduser().resolve()
    if not cal.is_file():
        raise SystemExit(f"calibration not found: {cal}")

    parquets = sorted(args.parquet_root.glob(f"{args.source_id}_*.parquet"))
    if args.max_clips is not None:
        parquets = parquets[: args.max_clips]
    if not parquets:
        raise SystemExit(f"no parquets for {args.source_id}")

    threshold = load_cv_threshold(args.cv_run.resolve(), None)
    seg_params = segment_params_for_extract_fps(args.extract_fps)

    print(
        f"source={args.source_id} clips={len(parquets)} extract_fps={args.extract_fps} "
        f"e2e=Y workers={args.workers}",
        flush=True,
    )
    print(f"calibration={cal}", flush=True)
    print(f"output={out_dir}", flush=True)

    t_wall = perf_counter()
    jobs = _build_jobs(
        parquets=parquets,
        args=args,
        out_dir=out_dir,
        cal=cal,
        threshold=threshold,
        seg_params=seg_params,
    )

    summaries: list[dict] = []
    feature_columns = active_feature_columns("all")
    if args.workers == 1:
        for job in jobs:
            print(f"  {job['clip_key']}...", flush=True)
            summaries.append(
                process_clip_e2e(
                    clip_key=job["clip_key"],
                    mp4=Path(job["mp4"]),
                    calibration_json=cal,
                    out_dir=out_dir,
                    cv_run=args.cv_run.resolve(),
                    feature_columns=feature_columns,
                    threshold=threshold,
                    extract_fps=args.extract_fps,
                    seg_params=seg_params,
                    yolo_device=args.yolo_device,
                    gt_df=None,
                    max_frames=args.max_frames,
                    parquet_path=Path(job["parquet_path"]),
                )
            )
    else:
        print(f"  dispatching {len(jobs)} clips to {args.workers} workers...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, job): job["clip_key"] for job in jobs}
            for fut in as_completed(futures):
                clip_key = futures[fut]
                try:
                    summaries.append(fut.result())
                except Exception as exc:
                    print(f"  {clip_key} FAILED: {exc}", flush=True)
                    raise

    summaries.sort(key=lambda s: s["clip_key"])
    wall_sec = round(perf_counter() - t_wall, 1)

    def _sum(key: str) -> float:
        return sum(s["timings_sec"][key] for s in summaries)

    input_sec = 0.0
    clip_dir = args.clip_root / args.source_id
    for s in summaries:
        mp4 = clip_dir / f"{s['clip_key']}.mp4"
        if mp4.is_file():
            d = probe_video_duration_sec(mp4)
            if d:
                input_sec += d

    cpu_sec = round(_sum("total"), 1)
    report = {
        "source_id": args.source_id,
        "mode": "e2e",
        "workers": args.workers,
        "extract_fps": args.extract_fps,
        "n_clips": len(summaries),
        "input_clip_sec": round(input_sec, 1),
        "wall_sec": wall_sec,
        "cpu_sec_sum": cpu_sec,
        "parallel_speedup": round(cpu_sec / wall_sec, 2) if wall_sec > 0 else None,
        "timings_sec": {
            "extract": round(_sum("extract"), 1),
            "infer": round(_sum("infer"), 1),
            "ffmpeg": round(_sum("ffmpeg"), 1),
            "total": cpu_sec,
        },
        "mean_per_clip_sec": round(cpu_sec / len(summaries), 2) if summaries else 0,
        "decision_threshold": threshold,
        "segment_params": seg_params,
        "clips": summaries,
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["timings_sec"]
    print(
        f"\nTOTAL e2e (cpu-sum): extract={t['extract']:.1f}s infer={t['infer']:.1f}s "
        f"ffmpeg={t['ffmpeg']:.1f}s clip-sum={t['total']:.1f}s",
        flush=True,
    )
    print(
        f"wall={wall_sec:.1f}s ({wall_sec/60:.1f} min) speedup={report['parallel_speedup']}x "
        f"input={report['input_clip_sec']:.0f}s",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
