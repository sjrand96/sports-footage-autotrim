"""Load homography + world canvas parameters from ``homography.npz`` or a ``court_calibrations`` row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_homography_npz(path: Path) -> tuple[np.ndarray, float, float, float, float, int, int]:
    """Match ``pose_side_by_side_video`` (and similar) expectations."""
    z = np.load(path, allow_pickle=True)
    H = np.asarray(z["H_world_to_pixel"], dtype=np.float64)
    meta = json.loads(bytes(np.asarray(z["meta_json"]).tobytes()).decode("utf-8"))
    wx_min, wx_max, wy_min, wy_max = meta["world_bounds_xy"]
    ppm = float(meta.get("pixels_per_metre_requested", 45.0))
    out_w = max(2, int(round((wx_max - wx_min) * ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * ppm)))
    return H, wx_min, wx_max, wy_min, wy_max, out_w, out_h


def homography_arrays_from_court_calibration_row(row: Mapping[str, Any]) -> tuple[np.ndarray, float, float, float, float, int, int]:
    """Same tuple as :func:`load_homography_npz` using a Supabase ``court_calibrations`` dict."""
    H = np.asarray(row["homography_matrix"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography_matrix must be 3×3, got {H.shape}")
    wx_min = float(row["world_wx_min"])
    wx_max = float(row["world_wx_max"])
    wy_min = float(row["world_wy_min"])
    wy_max = float(row["world_wy_max"])
    ppm = float(row.get("pixels_per_metre") or 45.0)
    out_w = max(2, int(round((wx_max - wx_min) * ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * ppm)))
    return H, wx_min, wx_max, wy_min, wy_max, out_w, out_h
