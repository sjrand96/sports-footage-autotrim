#!/usr/bin/env python3
"""Local clip autotrim: extract features → 5-fold XGB ensemble → segment smoother → playing MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.segment_metrics import Segment, extract_segments_from_probs  # noqa: E402
from feature_extraction.core.extract import extract_features_for_clip  # noqa: E402
from feature_extraction.core.feature_columns import (  # noqa: E402
    active_feature_columns,
    float_fillna_cols_for_features,
)
from feature_extraction.core.paths import ensure_import_paths  # noqa: E402

DEFAULT_CV_RUN = REPO_ROOT / "models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold"
DEFAULT_SEGMENT_THRESHOLD = 0.27


def stride_for_target_fps(target_fps: float, *, base_fps: int) -> int:
    """Match ``models/tabular_xgb/fps_sensitivity.stride_for_target_fps`` (avoid importing train)."""
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps >= base_fps:
        return 1
    return max(1, round(base_fps / target_fps))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autotrim a local clip: feature extract → XGB ensemble → segment cut → MP4."
    )
    p.add_argument("--input", type=Path, required=True, help="Local input .mp4")
    p.add_argument(
        "--features-parquet",
        type=Path,
        default=None,
        help="Skip YOLO extract; use precomputed per-frame features (smoke / debug).",
    )
    cal = p.add_mutually_exclusive_group(required=False)
    cal.add_argument("--calibration-json", type=Path, default=None, help="court_calibrations-style JSON")
    cal.add_argument("--calibration-npz", type=Path, default=None, help="homography.npz from calibration fit")
    p.add_argument(
        "--cv-run",
        type=Path,
        default=DEFAULT_CV_RUN,
        help=f"5-fold CV directory with fold-*/xgb_model.json (default: {DEFAULT_CV_RUN})",
    )
    p.add_argument(
        "--extract-fps",
        type=float,
        default=30.0,
        help="Target feature-extraction FPS (stride computed from source video FPS).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path (default: {input_stem}_playing.mp4 beside input).",
    )
    p.add_argument("--feature-subset", choices=("all", "base"), default="all")
    p.add_argument(
        "--segment-threshold",
        type=float,
        default=None,
        help="Playing probability cutoff (default: mean fold threshold from cv_summary.json).",
    )
    p.add_argument("--segment-window-frames", type=int, default=15)
    p.add_argument("--segment-aggregation", choices=("mean", "max"), default="mean")
    p.add_argument("--segment-max-gap-frames", type=int, default=45)
    p.add_argument("--segment-min-segment-frames", type=int, default=90)
    p.add_argument("--weights", type=str, default="yolov8s-pose.pt", help="YOLO pose weights path")
    p.add_argument(
        "--yolo-device",
        type=str,
        default="cpu",
        help="Ultralytics device (default cpu; avoids MPS segfaults on some Macs).",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Cap extracted rows (smoke tests only)")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Cache dir for sidecar JSON/CSV (default: autotrim/_runs/{timestamp}).",
    )
    p.add_argument(
        "--keep-sidecar",
        action="store_true",
        help="Write preds CSV + segments JSON under --run-dir.",
    )
    return p.parse_args()


def load_calibration(*, json_path: Path | None, npz_path: Path | None) -> tuple[np.ndarray, float, float, float, float]:
    ensure_import_paths()
    from homography_io import homography_arrays_from_court_calibration_row, load_homography_npz

    if json_path is not None:
        row = json.loads(json_path.read_text(encoding="utf-8"))
        H, wx_min, wx_max, wy_min, wy_max, _, _ = homography_arrays_from_court_calibration_row(row)
        return H, wx_min, wx_max, wy_min, wy_max
    assert npz_path is not None
    H, wx_min, wx_max, wy_min, wy_max, _, _ = load_homography_npz(npz_path)
    return H, wx_min, wx_max, wy_min, wy_max


def load_cv_threshold(cv_run: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    summary_path = cv_run / "cv_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mean = summary.get("summary", {}).get("decision_threshold", {}).get("mean")
        if mean is not None:
            return float(mean)
    return DEFAULT_SEGMENT_THRESHOLD


def load_fold_models(cv_run: Path) -> list[Any]:
    # Import after feature extract: loading XGB before Ultralytics can segfault on macOS.
    from xgboost import XGBClassifier

    fold_dirs = sorted(cv_run.glob("fold-*/xgb_model.json"))
    if not fold_dirs:
        raise FileNotFoundError(f"no fold-*/xgb_model.json under {cv_run}")
    models: list[Any] = []
    for model_path in fold_dirs:
        model = XGBClassifier()
        model.load_model(str(model_path))
        models.append(model)
    return models


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    fill_cols = float_fillna_cols_for_features(feature_columns)
    X = df[feature_columns].copy()
    for col in fill_cols:
        X[col] = X[col].fillna(-1.0)
    return X.to_numpy(dtype=np.float32)


def ensemble_predict_proba(models: list[Any], X: np.ndarray) -> np.ndarray:
    if len(models) == 1:
        return models[0].predict_proba(X)[:, 1]
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs.mean(axis=0)


def segments_to_time_intervals(
    segments: list[Segment],
    df: pd.DataFrame,
    *,
    source_fps: float,
    video_duration_sec: float | None = None,
) -> list[dict[str, float]]:
    if not segments:
        return []
    if "timestamp_sec" not in df.columns:
        raise ValueError("feature table missing timestamp_sec")

    ts = df["timestamp_sec"].to_numpy(dtype=float)
    n = len(df)
    intervals: list[dict[str, float]] = []
    for seg in segments:
        start_idx = max(0, min(seg.start, n - 1))
        end_idx = max(start_idx + 1, min(seg.end, n))
        start_sec = float(ts[start_idx])
        if end_idx < n:
            end_sec = float(ts[end_idx])
        elif video_duration_sec is not None:
            end_sec = float(video_duration_sec)
        else:
            end_sec = float(ts[end_idx - 1] + (1.0 / max(source_fps, 1e-6)))
        if end_sec > start_sec:
            intervals.append({"start": start_sec, "end": end_sec})
    return intervals


def probe_video_duration_sec(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def probe_has_audio(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def build_filter_complex(intervals: list[dict[str, float]], *, include_audio: bool) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, iv in enumerate(intervals):
        s = f"{float(iv['start']):.3f}"
        e = f"{float(iv['end']):.3f}"
        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        if include_audio:
            parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[v{i}][a{i}]")
        else:
            concat_inputs.append(f"[v{i}]")
    n = len(intervals)
    if include_audio:
        parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]")
    else:
        parts.append(f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[outv]")
    return ";".join(parts)


def export_cut_video(
    *,
    input_path: Path,
    output_path: Path,
    intervals: list[dict[str, float]],
) -> None:
    if not intervals:
        raise ValueError("no intervals to export")
    include_audio = probe_has_audio(input_path)
    filt = build_filter_complex(intervals, include_audio=include_audio)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        filt,
        "-map",
        "[outv]",
    ]
    if include_audio:
        args.extend(["-map", "[outa]", "-c:a", "aac", "-b:a", "128k"])
    else:
        args.append("-an")
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found (install ffmpeg, e.g. brew install ffmpeg).") from exc
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"ffmpeg exited with code {proc.returncode}")


def default_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "autotrim" / "_runs" / stamp


def resolve_weights_path(weights: str) -> str:
    path = Path(weights)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def probe_source_fps(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return fps


def load_features(
    *,
    input_path: Path,
    features_parquet: Path | None,
    calibration_json: Path | None,
    calibration_npz: Path | None,
    extract_fps: float,
    weights: str,
    yolo_device: str | None,
    max_frames: int | None,
) -> tuple[pd.DataFrame, float, int | None]:
    """Return (feature_df, source_fps, frame_stride or None if parquet)."""
    source_fps = probe_source_fps(input_path)

    if features_parquet is not None:
        path = features_parquet.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"features parquet not found: {path}")
        df = pd.read_parquet(path)
        if max_frames is not None:
            df = df.sort_values("frame_idx", kind="stable").head(max_frames).reset_index(drop=True)
        print(f"loaded features: {path} ({len(df)} rows)", flush=True)
        return df, source_fps, None

    if calibration_json is None and calibration_npz is None:
        raise SystemExit(
            "provide --calibration-json or --calibration-npz (or pass --features-parquet to skip extract)"
        )

    H, wx_min, wx_max, wy_min, wy_max = load_calibration(
        json_path=calibration_json,
        npz_path=calibration_npz,
    )
    frame_stride = stride_for_target_fps(float(extract_fps), base_fps=int(round(source_fps)))
    print(
        f"extract: target_fps={extract_fps} source_fps≈{source_fps:.2f} stride={frame_stride}",
        flush=True,
    )
    df, video_meta = extract_features_for_clip(
        video_path=input_path,
        H=H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        weights=resolve_weights_path(weights),
        frame_stride=frame_stride,
        max_frames=max_frames,
        yolo_device=yolo_device,
    )
    source_fps = float(video_meta.get("source_fps") or source_fps)
    return df, source_fps, frame_stride


def _round_sec(elapsed: float) -> float:
    return round(elapsed, 3)


def _print_timings(timings: dict[str, float]) -> None:
    print(
        "timings (sec): "
        f"extract={timings['extract']:.3f} "
        f"infer={timings['infer']:.3f} "
        f"ffmpeg={timings['ffmpeg']:.3f} "
        f"total={timings['total']:.3f}",
        flush=True,
    )


def main() -> None:
    t_total = perf_counter()
    timings: dict[str, float] = {}
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_playing.mp4")
    else:
        output_path = output_path.resolve()

    cv_run = args.cv_run.resolve()
    if not cv_run.is_dir():
        raise SystemExit(f"--cv-run not found: {cv_run}")

    t0 = perf_counter()
    df, source_fps, frame_stride = load_features(
        input_path=input_path,
        features_parquet=args.features_parquet,
        calibration_json=args.calibration_json,
        calibration_npz=args.calibration_npz,
        extract_fps=float(args.extract_fps),
        weights=args.weights,
        yolo_device=args.yolo_device,
        max_frames=args.max_frames,
    )
    timings["extract"] = _round_sec(perf_counter() - t0)

    feature_columns = active_feature_columns(args.feature_subset)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise SystemExit(f"extracted features missing columns: {missing}")

    t0 = perf_counter()
    models = load_fold_models(cv_run)
    print(f"loaded {len(models)} fold model(s) from {cv_run}", flush=True)

    X = prepare_feature_matrix(df, feature_columns)
    probs = ensemble_predict_proba(models, X)

    threshold = load_cv_threshold(cv_run, args.segment_threshold)
    segments, _ = extract_segments_from_probs(
        probs,
        threshold=threshold,
        window_frames=args.segment_window_frames,
        aggregation=args.segment_aggregation,
        max_gap_frames=args.segment_max_gap_frames,
        min_segment_frames=args.segment_min_segment_frames,
    )
    if not segments:
        raise SystemExit(
            "no playing segments detected after smoothing; try lowering --segment-threshold "
            "or --segment-min-segment-frames."
        )

    video_duration = probe_video_duration_sec(input_path)
    intervals = segments_to_time_intervals(
        segments,
        df,
        source_fps=source_fps,
        video_duration_sec=video_duration,
    )
    if not intervals:
        raise SystemExit("segments did not map to any time intervals")
    timings["infer"] = _round_sec(perf_counter() - t0)

    playing_sec = sum(iv["end"] - iv["start"] for iv in intervals)
    print(
        f"segments: {len(intervals)} intervals, ≈{playing_sec:.1f}s playing "
        f"(threshold={threshold:.3f})",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    export_cut_video(input_path=input_path, output_path=output_path, intervals=intervals)
    timings["ffmpeg"] = _round_sec(perf_counter() - t0)
    timings["total"] = _round_sec(perf_counter() - t_total)
    _print_timings(timings)
    print(f"wrote {output_path}", flush=True)

    if args.keep_sidecar:
        run_dir = (args.run_dir or default_run_dir()).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        pred_cols = ["frame_idx", "timestamp_sec"]
        if "source_frame_idx" in df.columns:
            pred_cols.insert(1, "source_frame_idx")
        preds = df[pred_cols].copy()
        preds["prob_playing"] = probs
        preds.to_csv(run_dir / "preds.csv", index=False)
        sidecar = {
            "input": str(input_path),
            "output": str(output_path),
            "cv_run": str(cv_run),
            "extract_fps": args.extract_fps,
            "frame_stride": frame_stride,
            "source_fps": source_fps,
            "segment_threshold": threshold,
            "n_segments": len(intervals),
            "playing_sec": playing_sec,
            "intervals": intervals,
            "segments": [s.to_dict() for s in segments],
            "timings_sec": timings,
        }
        (run_dir / "trim_sidecar.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"sidecar: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
