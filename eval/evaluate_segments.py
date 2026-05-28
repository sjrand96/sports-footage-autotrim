#!/usr/bin/env python3
"""Evaluate full-video playing segments from per-frame prediction CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.segment_metrics import evaluate_arrays, precision_recall_f1  # noqa: E402

REQUIRED_BASE_COLS = ("clip_key", "frame_idx")


def parse_clip_key(clip_key: str) -> tuple[str, int]:
    base, idx = str(clip_key).rsplit("_", 1)
    return base, int(idx)


def load_prediction_csv(path: Path, *, split_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    if missing:
        raise SystemExit(f"prediction CSV missing columns {missing}: {path}")
    if "prob_playing" not in df.columns and "pred_playing" not in df.columns:
        raise SystemExit("prediction CSV needs either prob_playing or pred_playing")
    if split_name is not None:
        if "split_name" not in df.columns:
            raise SystemExit("--split-name requires a split_name column")
        df = df[df["split_name"].astype(str) == split_name].copy()
        if df.empty:
            raise SystemExit(f"no rows with split_name={split_name!r}")

    if "source_id" not in df.columns:
        df["source_id"] = df["clip_key"].map(lambda key: parse_clip_key(str(key))[0])
    if "clip_index" not in df.columns:
        df["clip_index"] = df["clip_key"].map(lambda key: parse_clip_key(str(key))[1])
    if "prob_playing" not in df.columns:
        df["prob_playing"] = df["pred_playing"].astype(float)
    if "is_playing" in df.columns:
        df["is_playing"] = df["is_playing"].astype(int)
    df["clip_index"] = df["clip_index"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    return df


def concatenate_group(df: pd.DataFrame) -> pd.DataFrame:
    """Sort clips and create a contiguous frame index for one eval group."""
    parts: list[pd.DataFrame] = []
    offset = 0
    for _, clip_df in df.sort_values(["clip_index", "frame_idx"], kind="stable").groupby(
        "clip_key", sort=False
    ):
        part = clip_df.sort_values("frame_idx", kind="stable").copy()
        part["video_frame_idx"] = part["frame_idx"].astype(int) + offset
        if len(part):
            offset = int(part["video_frame_idx"].max()) + 1
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=list(df.columns) + ["video_frame_idx"])
    return pd.concat(parts, ignore_index=True)


def group_prediction_frames(df: pd.DataFrame, *, group_by: str) -> list[tuple[str, pd.DataFrame]]:
    if group_by == "source":
        groups = []
        for source_id, group in df.groupby("source_id", sort=True):
            groups.append((str(source_id), concatenate_group(group)))
        return groups
    if group_by == "clip":
        return [
            (str(clip_key), group.sort_values("frame_idx", kind="stable").copy())
            for clip_key, group in df.groupby("clip_key", sort=True)
        ]
    raise ValueError(f"group_by must be 'source' or 'clip', got {group_by!r}")


def evaluate_group(name: str, group: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    probs = group["prob_playing"].to_numpy(dtype=np.float64)
    y_true = group["is_playing"].to_numpy(dtype=np.int8) if "is_playing" in group.columns else None
    metrics = evaluate_arrays(
        probs=probs,
        y_true=y_true,
        threshold=args.threshold,
        window_frames=args.window_frames,
        aggregation=args.aggregation,
        max_gap_frames=args.max_gap_frames,
        min_segment_frames=args.min_segment_frames,
        boundary_tolerance_frames=args.boundary_tolerance_frames,
        max_overlength_frames=args.max_overlength_frames,
        max_overlength_ratio=args.max_overlength_ratio,
        pre_buffer_frames=args.pre_buffer_frames,
        post_buffer_frames=args.post_buffer_frames,
    )
    return {
        "name": name,
        "n_clips": int(group["clip_key"].nunique()),
        "source_ids": sorted(group["source_id"].astype(str).unique().tolist()),
        "metrics": metrics,
    }


def summarize_groups(group_results: list[dict[str, Any]], *, supervised: bool) -> dict[str, Any]:
    n_frames = sum(float(g["metrics"].get("n_frames", 0.0)) for g in group_results)
    kept_frames = sum(
        float(g["metrics"].get("kept_frame_ratio", 0.0)) * float(g["metrics"].get("n_frames", 0.0))
        for g in group_results
    )
    summary: dict[str, Any] = {
        "n_groups": len(group_results),
        "n_frames": n_frames,
        "kept_frame_ratio": kept_frames / max(n_frames, 1.0),
    }
    if not supervised:
        summary["predicted_segments"] = sum(
            float(g["metrics"].get("predicted_segment_duration_frames", {}).get("count", 0.0))
            for g in group_results
        )
        return summary

    tp = sum(float(g["metrics"].get("matched_segments", 0.0)) for g in group_results)
    fp = sum(float(g["metrics"].get("false_positive_segments", 0.0)) for g in group_results)
    fn = sum(float(g["metrics"].get("missed_segments", 0.0)) for g in group_results)
    pr = precision_recall_f1(int(tp), int(fp), int(fn))
    summary.update(
        {
            "matched_segments": tp,
            "false_positive_segments": fp,
            "missed_segments": fn,
            "segment_precision": pr["precision"],
            "segment_recall": pr["recall"],
            "segment_f1": pr["f1"],
            "macro_segment_f1": float(
                np.mean([g["metrics"].get("segment_f1", 0.0) for g in group_results])
            )
            if group_results
            else 0.0,
        }
    )
    return summary


def build_report(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    groups = group_prediction_frames(df, group_by=args.group_by)
    group_results = [evaluate_group(name, group, args) for name, group in groups]
    supervised = "is_playing" in df.columns
    return {
        "input_csv": str(args.csv.expanduser().resolve()),
        "group_by": args.group_by,
        "supervised": supervised,
        "parameters": {
            "threshold": args.threshold,
            "window_frames": args.window_frames,
            "aggregation": args.aggregation,
            "max_gap_frames": args.max_gap_frames,
            "min_segment_frames": args.min_segment_frames,
            "boundary_tolerance_frames": args.boundary_tolerance_frames,
            "max_overlength_frames": args.max_overlength_frames,
            "max_overlength_ratio": args.max_overlength_ratio,
            "pre_buffer_frames": args.pre_buffer_frames,
            "post_buffer_frames": args.post_buffer_frames,
        },
        "summary": summarize_groups(group_results, supervised=supervised),
        "groups": group_results,
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=== Segment Evaluation ===")
    print(f"csv: {report['input_csv']}")
    print(f"group_by: {report['group_by']}  supervised: {report['supervised']}")
    print(f"groups: {summary['n_groups']}  frames: {int(summary['n_frames'])}")
    print(f"kept_frame_ratio: {summary['kept_frame_ratio']:.3f}")
    if report["supervised"]:
        print(
            "segment: "
            f"precision={summary['segment_precision']:.3f} "
            f"recall={summary['segment_recall']:.3f} "
            f"f1={summary['segment_f1']:.3f} "
            f"matched={int(summary['matched_segments'])} "
            f"fp={int(summary['false_positive_segments'])} "
            f"missed={int(summary['missed_segments'])}"
        )
    else:
        print(f"predicted_segments: {int(summary['predicted_segments'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--group-by", choices=("source", "clip"), default="source")
    p.add_argument("--split-name", default=None)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--window-frames", type=int, default=15)
    p.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    p.add_argument("--max-gap-frames", type=int, default=45)
    p.add_argument("--min-segment-frames", type=int, default=90)
    p.add_argument("--boundary-tolerance-frames", type=int, default=30)
    p.add_argument("--max-overlength-frames", type=int, default=150)
    p.add_argument("--max-overlength-ratio", type=float, default=2.0)
    p.add_argument("--pre-buffer-frames", type=int, default=0)
    p.add_argument("--post-buffer-frames", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = load_prediction_csv(args.csv.expanduser().resolve(), split_name=args.split_name)
    report = build_report(df, args)
    print_summary(report)
    if args.out_json is not None:
        out = args.out_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
