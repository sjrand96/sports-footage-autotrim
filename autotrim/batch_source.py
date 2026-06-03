#!/usr/bin/env python3
"""Batch autotrim for one source: 2 fps features from parquets + 5-fold XGB + plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.segment_metrics import extract_segments_from_probs  # noqa: E402
from feature_extraction.core.feature_columns import active_feature_columns  # noqa: E402
from models.tabular_xgb.fps_sensitivity import stride_for_target_fps, subsample_frames_within_clips  # noqa: E402
from models.tabular_xgb.train import _clip_key_frame  # noqa: E402

# Reuse trim helpers
from autotrim.trim_clip import (  # noqa: E402
    DEFAULT_CV_RUN,
    ensemble_predict_proba,
    export_cut_video,
    load_cv_threshold,
    load_fold_models,
    prepare_feature_matrix,
    probe_source_fps,
    probe_video_duration_sec,
    segments_to_time_intervals,
)

DEFAULT_PARQUET_ROOT = REPO_ROOT / "feature_extraction/_runs/all_clips_features_v2/parquet"
DEFAULT_CLIP_ROOT = REPO_ROOT / "cv-pipeline/pose-detection/media/clips"


def segment_params_for_extract_fps(extract_fps: float, *, base_fps: float = 30.0) -> dict[str, int]:
    scale = extract_fps / base_fps
    return {
        "segment_window_frames": max(1, round(15 * scale)),
        "segment_max_gap_frames": max(1, round(45 * scale)),
        "segment_min_segment_frames": max(1, round(90 * scale)),
    }


def clip_key_from_parquet(df: pd.DataFrame) -> str:
    sid = str(df["source_id"].iloc[0])
    idx = int(df["clip_index"].iloc[0])
    return _clip_key_frame(sid, idx)


def load_subsampled_parquet(path: Path, *, extract_fps: float, base_fps: float = 30.0) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "clip_key" not in df.columns:
        df = df.copy()
        df["clip_key"] = clip_key_from_parquet(df)
    stride = stride_for_target_fps(extract_fps, base_fps=int(base_fps))
    return subsample_frames_within_clips(df, stride)


def resolve_mp4(clip_key: str, *, clip_root: Path, download: bool) -> Path:
    source_id, idx_s = clip_key.rsplit("_", 1)
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


def process_clip(
    *,
    parquet_path: Path,
    clip_root: Path,
    out_dir: Path,
    models: list,
    feature_columns: list[str],
    threshold: float,
    extract_fps: float,
    seg_params: dict[str, int],
    download: bool,
) -> dict:
    t0 = perf_counter()
    df = load_subsampled_parquet(parquet_path, extract_fps=extract_fps)
    clip_key = str(df["clip_key"].iloc[0])
    mp4 = resolve_mp4(clip_key, clip_root=clip_root, download=download)
    source_fps = probe_source_fps(mp4)

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
    out_mp4 = out_dir / "videos" / f"{clip_key}_playing.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if intervals:
        export_cut_video(input_path=mp4, output_path=out_mp4, intervals=intervals)
    else:
        out_mp4 = None

    yt = df["is_playing"].astype(int).to_numpy()
    yp = (probs >= threshold).astype(int)
    metrics = {
        "f2": float(fbeta_score(yt, yp, beta=2, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "playing_gt_frac": float(yt.mean()),
    }

    pred_row = df[["clip_key", "frame_idx", "timestamp_sec", "is_playing"]].copy()
    if "source_id" in df.columns:
        pred_row["source_id"] = df["source_id"].astype(str)
    if "clip_index" in df.columns:
        pred_row["clip_index"] = df["clip_index"].astype(int)
    pred_row["prob_playing"] = probs
    pred_row["pred_playing"] = yp
    pred_row["decision_threshold"] = threshold

    sidecar = {
        "clip_key": clip_key,
        "extract_fps": extract_fps,
        "n_rows": len(df),
        "metrics": metrics,
        "n_segments": len(intervals),
        "playing_sec": sum(iv["end"] - iv["start"] for iv in intervals) if intervals else 0.0,
        "intervals": intervals,
        "output_mp4": str(out_mp4) if out_mp4 else None,
        "elapsed_sec": round(perf_counter() - t0, 2),
    }
    (out_dir / "sidecars").mkdir(parents=True, exist_ok=True)
    (out_dir / "sidecars" / f"{clip_key}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return {"clip_key": clip_key, "preds": pred_row, "sidecar": sidecar}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-id", required=True)
    p.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    p.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    p.add_argument("--cv-run", type=Path, default=DEFAULT_CV_RUN)
    p.add_argument("--extract-fps", type=float, default=2.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--skip-videos", action="store_true", help="Predictions + plots only.")
    args = p.parse_args()

    out_dir = (args.output_dir or (REPO_ROOT / "autotrim/_runs" / args.source_id / "2fps_batch")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(args.parquet_root.glob(f"{args.source_id}_*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquets for {args.source_id} under {args.parquet_root}")

    feature_columns = active_feature_columns("all")
    models = load_fold_models(args.cv_run.resolve())
    threshold = load_cv_threshold(args.cv_run.resolve(), None)
    seg_params = segment_params_for_extract_fps(args.extract_fps)

    print(f"source={args.source_id} clips={len(parquets)} extract_fps={args.extract_fps} threshold={threshold:.3f}")
    print(f"segment params: {seg_params}")
    print(f"output: {out_dir}")

    all_preds: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for pq in parquets:
        print(f"  {pq.stem}...", flush=True)
        if args.skip_videos:
            # preds only path
            df = load_subsampled_parquet(pq, extract_fps=args.extract_fps)
            clip_key = str(df["clip_key"].iloc[0])
            X = prepare_feature_matrix(df, feature_columns)
            probs = ensemble_predict_proba(models, X)
            yt = df["is_playing"].astype(int).to_numpy()
            yp = (probs >= threshold).astype(int)
            pred_row = df[["clip_key", "frame_idx", "timestamp_sec", "is_playing"]].copy()
            pred_row["source_id"] = args.source_id
            pred_row["prob_playing"] = probs
            pred_row["pred_playing"] = yp
            pred_row["decision_threshold"] = threshold
            all_preds.append(pred_row)
            summaries.append({"clip_key": clip_key, "metrics": {"f2": float(fbeta_score(yt, yp, beta=2))}})
            continue

        result = process_clip(
            parquet_path=pq,
            clip_root=args.clip_root,
            out_dir=out_dir,
            models=models,
            feature_columns=feature_columns,
            threshold=threshold,
            extract_fps=args.extract_fps,
            seg_params=seg_params,
            download=not args.no_download,
        )
        all_preds.append(result["preds"])
        summaries.append(result["sidecar"])

    preds_csv = out_dir / "preds_all.csv"
    pd.concat(all_preds, ignore_index=True).to_csv(preds_csv, index=False)

    # Pooled metrics
    big = pd.concat(all_preds, ignore_index=True)
    yt = big["is_playing"].astype(int).to_numpy()
    yp = big["pred_playing"].astype(int).to_numpy()
    report = {
        "source_id": args.source_id,
        "extract_fps": args.extract_fps,
        "n_clips": len(parquets),
        "n_frames": len(big),
        "decision_threshold": threshold,
        "segment_params": seg_params,
        "pooled_f2": float(fbeta_score(yt, yp, beta=2, zero_division=0)),
        "pooled_recall": float(recall_score(yt, yp, zero_division=0)),
        "pooled_precision": float(precision_score(yt, yp, zero_division=0)),
        "clips": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\npooled F2={report['pooled_f2']:.4f} recall={report['pooled_recall']:.4f} precision={report['pooled_precision']:.4f}")
    print(f"wrote {preds_csv}")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
