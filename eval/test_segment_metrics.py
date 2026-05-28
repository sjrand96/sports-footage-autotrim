from __future__ import annotations

import numpy as np

from eval.segment_metrics import (
    Segment,
    binary_series_to_segments,
    evaluate_arrays,
    match_segments,
)


def test_binary_series_to_segments_half_open() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0])

    assert binary_series_to_segments(labels) == [Segment(1, 3), Segment(4, 5)]


def test_match_segments_rejects_one_giant_prediction() -> None:
    predicted = [Segment(0, 100)]
    truth = [Segment(10, 20), Segment(40, 50)]

    matches, false_positives, missed = match_segments(
        predicted,
        truth,
        max_overlength_frames=5,
        max_overlength_ratio=2.0,
    )

    assert matches == []
    assert false_positives == predicted
    assert missed == truth


def test_evaluate_arrays_reports_segment_f1() -> None:
    y_true = np.zeros(120, dtype=np.int8)
    y_true[30:60] = 1
    probs = np.zeros(120, dtype=np.float64)
    probs[31:59] = 0.9

    result = evaluate_arrays(
        probs=probs,
        y_true=y_true,
        threshold=0.5,
        window_frames=1,
        max_gap_frames=0,
        min_segment_frames=1,
        boundary_tolerance_frames=2,
    )

    assert result["matched_segments"] == 1.0
    assert result["segment_f1"] == 1.0
    assert result["frame_metrics"]["recall"] < 1.0
