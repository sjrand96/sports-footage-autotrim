"""Segment extraction and matching metrics for playing/downtime predictions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segment:
    """Half-open frame interval: ``[start, end)``."""

    start: int
    end: int
    score: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"segment start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"segment end must be >= start, got {self}")

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, int | float | None]:
        return {"start": self.start, "end": self.end, "duration": self.duration, "score": self.score}


@dataclass(frozen=True)
class Match:
    pred: Segment
    truth: Segment
    overlap: int
    start_error: int
    end_error: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pred": self.pred.to_dict(),
            "truth": self.truth.to_dict(),
            "overlap": self.overlap,
            "start_error": self.start_error,
            "end_error": self.end_error,
        }


def binary_series_to_segments(labels: np.ndarray, *, scores: np.ndarray | None = None) -> list[Segment]:
    """Convert a 0/1 frame series into contiguous positive segments."""
    y = np.asarray(labels, dtype=np.int8).ravel()
    if scores is not None and len(scores) != len(y):
        raise ValueError("scores length must match labels length")
    if len(y) == 0:
        return []

    padded = np.concatenate([[0], y, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    segments: list[Segment] = []
    for start, end in zip(starts, ends, strict=True):
        score = None
        if scores is not None and end > start:
            score = float(np.nanmean(np.asarray(scores, dtype=np.float64)[start:end]))
        segments.append(Segment(int(start), int(end), score))
    return segments


def rolling_aggregate(
    probs: np.ndarray,
    *,
    window_frames: int,
    mode: str = "mean",
) -> np.ndarray:
    """Return same-length rolling aggregation over probabilities."""
    values = np.asarray(probs, dtype=np.float64).ravel()
    if len(values) == 0 or window_frames <= 1:
        return values.copy()
    if mode not in {"mean", "max"}:
        raise ValueError(f"mode must be 'mean' or 'max', got {mode!r}")

    window = int(window_frames)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    if mode == "mean":
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(padded, kernel, mode="valid")

    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        out[idx] = float(np.max(padded[idx : idx + window]))
    return out


def close_short_gaps(binary: np.ndarray, *, max_gap_frames: int) -> np.ndarray:
    """Fill negative gaps between positive runs when gap length is small enough."""
    y = np.asarray(binary, dtype=np.int8).ravel().copy()
    if max_gap_frames <= 0 or len(y) == 0:
        return y

    segments = binary_series_to_segments(y)
    for left, right in zip(segments, segments[1:], strict=False):
        gap_start = left.end
        gap_end = right.start
        if 0 < gap_end - gap_start <= max_gap_frames:
            y[gap_start:gap_end] = 1
    return y


def remove_short_segments(binary: np.ndarray, *, min_segment_frames: int) -> np.ndarray:
    """Suppress positive runs shorter than ``min_segment_frames``."""
    y = np.asarray(binary, dtype=np.int8).ravel().copy()
    if min_segment_frames <= 1 or len(y) == 0:
        return y
    for segment in binary_series_to_segments(y):
        if segment.duration < min_segment_frames:
            y[segment.start : segment.end] = 0
    return y


def apply_segment_buffers(
    segments: list[Segment],
    *,
    n_frames: int,
    pre_buffer_frames: int = 0,
    post_buffer_frames: int = 0,
) -> list[Segment]:
    """Pad segments and merge any overlaps introduced by padding."""
    if not segments:
        return []
    padded = [
        Segment(
            max(0, s.start - max(0, pre_buffer_frames)),
            min(n_frames, s.end + max(0, post_buffer_frames)),
            s.score,
        )
        for s in segments
    ]
    padded.sort(key=lambda s: (s.start, s.end))
    merged: list[Segment] = [padded[0]]
    for segment in padded[1:]:
        prev = merged[-1]
        if segment.start <= prev.end:
            score = None
            if prev.score is not None or segment.score is not None:
                score = float(np.nanmean([v for v in (prev.score, segment.score) if v is not None]))
            merged[-1] = Segment(prev.start, max(prev.end, segment.end), score)
        else:
            merged.append(segment)
    return merged


def extract_segments_from_probs(
    probs: np.ndarray,
    *,
    threshold: float = 0.35,
    window_frames: int = 15,
    aggregation: str = "mean",
    max_gap_frames: int = 45,
    min_segment_frames: int = 90,
    pre_buffer_frames: int = 0,
    post_buffer_frames: int = 0,
) -> tuple[list[Segment], np.ndarray]:
    """Convert per-frame probabilities to predicted playing segments."""
    raw_probs = np.asarray(probs, dtype=np.float64).ravel()
    if len(raw_probs) == 0:
        return [], np.asarray([], dtype=np.int8)

    smoothed = rolling_aggregate(raw_probs, window_frames=window_frames, mode=aggregation)
    binary = (smoothed >= float(threshold)).astype(np.int8)
    binary = close_short_gaps(binary, max_gap_frames=max_gap_frames)
    binary = remove_short_segments(binary, min_segment_frames=min_segment_frames)
    segments = binary_series_to_segments(binary, scores=smoothed)
    if pre_buffer_frames or post_buffer_frames:
        segments = apply_segment_buffers(
            segments,
            n_frames=len(raw_probs),
            pre_buffer_frames=pre_buffer_frames,
            post_buffer_frames=post_buffer_frames,
        )
        binary = segments_to_binary(segments, n_frames=len(raw_probs))
    return segments, binary


def segments_to_binary(segments: list[Segment], *, n_frames: int) -> np.ndarray:
    y = np.zeros(n_frames, dtype=np.int8)
    for segment in segments:
        y[max(0, segment.start) : min(n_frames, segment.end)] = 1
    return y


def segment_overlap(left: Segment, right: Segment) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _passes_length_guard(
    pred: Segment,
    truth: Segment,
    *,
    max_overlength_frames: int,
    max_overlength_ratio: float,
) -> bool:
    if pred.duration <= 0 or truth.duration <= 0:
        return False
    allowed = max(
        truth.duration + max(0, int(max_overlength_frames)),
        int(round(truth.duration * float(max_overlength_ratio))),
    )
    return pred.duration <= allowed


def match_segments(
    predicted: list[Segment],
    truth: list[Segment],
    *,
    boundary_tolerance_frames: int = 30,
    max_overlength_frames: int = 150,
    max_overlength_ratio: float = 2.0,
) -> tuple[list[Match], list[Segment], list[Segment]]:
    """Greedy one-to-one segment matching with simple length guards."""
    candidates: list[tuple[int, int, int, int, int]] = []
    tolerance = max(0, int(boundary_tolerance_frames))
    for pred_idx, pred in enumerate(predicted):
        for truth_idx, gt in enumerate(truth):
            overlap = segment_overlap(pred, gt)
            start_err = abs(pred.start - gt.start)
            end_err = abs(pred.end - gt.end)
            close_boundaries = start_err <= tolerance and end_err <= tolerance
            if overlap <= 0 and not close_boundaries:
                continue
            if not _passes_length_guard(
                pred,
                gt,
                max_overlength_frames=max_overlength_frames,
                max_overlength_ratio=max_overlength_ratio,
            ):
                continue
            boundary_err = start_err + end_err
            candidates.append((-overlap, boundary_err, pred_idx, truth_idx, overlap))

    matched_pred: set[int] = set()
    matched_truth: set[int] = set()
    matches: list[Match] = []
    for _, _, pred_idx, truth_idx, overlap in sorted(candidates):
        if pred_idx in matched_pred or truth_idx in matched_truth:
            continue
        pred = predicted[pred_idx]
        gt = truth[truth_idx]
        matches.append(
            Match(
                pred=pred,
                truth=gt,
                overlap=int(overlap),
                start_error=int(pred.start - gt.start),
                end_error=int(pred.end - gt.end),
            )
        )
        matched_pred.add(pred_idx)
        matched_truth.add(truth_idx)

    false_positives = [s for idx, s in enumerate(predicted) if idx not in matched_pred]
    missed = [s for idx, s in enumerate(truth) if idx not in matched_truth]
    return matches, false_positives, missed


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.int8).ravel()
    yp = np.asarray(y_pred, dtype=np.int8).ravel()
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred length mismatch")
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    pr = precision_recall_f1(tp, fp, fn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": pr["precision"],
        "recall": pr["recall"],
        "f1": pr["f1"],
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def _duration_stats(segments: list[Segment]) -> dict[str, float]:
    durations = [s.duration for s in segments]
    if not durations:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": float(len(durations)),
        "min": float(min(durations)),
        "median": float(median(durations)),
        "mean": float(mean(durations)),
        "max": float(max(durations)),
    }


def segment_metrics(
    predicted: list[Segment],
    truth: list[Segment],
    *,
    n_frames: int,
    boundary_tolerance_frames: int = 30,
    max_overlength_frames: int = 150,
    max_overlength_ratio: float = 2.0,
) -> dict[str, Any]:
    matches, false_positives, missed = match_segments(
        predicted,
        truth,
        boundary_tolerance_frames=boundary_tolerance_frames,
        max_overlength_frames=max_overlength_frames,
        max_overlength_ratio=max_overlength_ratio,
    )
    tp = len(matches)
    fp = len(false_positives)
    fn = len(missed)
    pr = precision_recall_f1(tp, fp, fn)
    start_errors = [m.start_error for m in matches]
    end_errors = [m.end_error for m in matches]
    pred_frames = int(sum(s.duration for s in predicted))
    true_frames = int(sum(s.duration for s in truth))
    return {
        "n_frames": float(n_frames),
        "matched_segments": float(tp),
        "false_positive_segments": float(fp),
        "missed_segments": float(fn),
        "segment_precision": pr["precision"],
        "segment_recall": pr["recall"],
        "segment_f1": pr["f1"],
        "kept_frame_ratio": pred_frames / max(n_frames, 1),
        "true_playing_frame_ratio": true_frames / max(n_frames, 1),
        "predicted_segment_duration_frames": _duration_stats(predicted),
        "true_segment_duration_frames": _duration_stats(truth),
        "mean_start_error_frames": float(mean(start_errors)) if start_errors else 0.0,
        "median_start_error_frames": float(median(start_errors)) if start_errors else 0.0,
        "mean_end_error_frames": float(mean(end_errors)) if end_errors else 0.0,
        "median_end_error_frames": float(median(end_errors)) if end_errors else 0.0,
        "matches": [m.to_dict() for m in matches],
        "false_positives": [s.to_dict() for s in false_positives],
        "missed": [s.to_dict() for s in missed],
    }


def evaluate_arrays(
    *,
    probs: np.ndarray,
    y_true: np.ndarray | None = None,
    threshold: float = 0.35,
    window_frames: int = 15,
    aggregation: str = "mean",
    max_gap_frames: int = 45,
    min_segment_frames: int = 90,
    boundary_tolerance_frames: int = 30,
    max_overlength_frames: int = 150,
    max_overlength_ratio: float = 2.0,
    pre_buffer_frames: int = 0,
    post_buffer_frames: int = 0,
) -> dict[str, Any]:
    """Evaluate one contiguous video/clip probability series."""
    pred_segments, pred_binary = extract_segments_from_probs(
        probs,
        threshold=threshold,
        window_frames=window_frames,
        aggregation=aggregation,
        max_gap_frames=max_gap_frames,
        min_segment_frames=min_segment_frames,
        pre_buffer_frames=pre_buffer_frames,
        post_buffer_frames=post_buffer_frames,
    )
    n_frames = len(np.asarray(probs).ravel())
    result: dict[str, Any] = {
        "n_frames": float(n_frames),
        "predicted_segments": [s.to_dict() for s in pred_segments],
        "predicted_segment_duration_frames": _duration_stats(pred_segments),
        "kept_frame_ratio": float(pred_binary.sum()) / max(n_frames, 1),
    }
    if y_true is None:
        return result

    true_arr = np.asarray(y_true, dtype=np.int8).ravel()
    if len(true_arr) != n_frames:
        raise ValueError("y_true and probs length mismatch")
    true_segments = binary_series_to_segments(true_arr)
    result.update(
        segment_metrics(
            pred_segments,
            true_segments,
            n_frames=n_frames,
            boundary_tolerance_frames=boundary_tolerance_frames,
            max_overlength_frames=max_overlength_frames,
            max_overlength_ratio=max_overlength_ratio,
        )
    )
    result["frame_metrics"] = frame_metrics(true_arr, pred_binary)
    return result
