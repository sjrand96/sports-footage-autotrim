"""Fit ground-plane homography from Label Studio keypoints + FIVB world points; warp to top-down."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_CALIB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from data_labeling.court_keypoints import CalibrationRecord, load_calibration_records

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV required: pip install -e '.[cv]'") from e

from image_io import fetch_image_bgr, fetch_image_bgr_boto3, load_image_bgr


# Opinionated defaults (no CLI flags — change here if needed)
_MARGIN_M = 1.0
_PIXELS_PER_METRE = 45.0
_CANVAS_MODE = "from-labels"  # label hull padded, expanded to full 18×9 m playing rectangle
AWS_REGION_FALLBACK = os.environ.get("AWS_REGION", "us-west-2")


_GEOMETRY_FILE = _CALIB_DIR / "fivb_court_geometry.txt"
_SKIP_LABELS_FOR_H = frozenset({"net_post_top_left", "net_post_top_right"})


# ----- Optional: keypoint overlay on camera frame (replaces standalone court_overlay.py) -----

_COLOR_BASELINE = (40, 40, 230)
_COLOR_ATTACK = (60, 200, 80)
_COLOR_CENTER = (230, 160, 60)
_COLOR_NET = (0, 220, 255)
_COLOR_NET_TOP = (80, 140, 255)
_COLOR_SIDE = (200, 200, 200)

_LINE_PAIRS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("far_baseline_left", "far_baseline_right", _COLOR_BASELINE),
    ("near_baseline_left", "near_baseline_right", _COLOR_BASELINE),
    ("far_attack_left", "far_attack_right", _COLOR_ATTACK),
    ("near_attack_left", "near_attack_right", _COLOR_ATTACK),
    ("centerline_left", "centerline_right", _COLOR_CENTER),
    ("net_post_base_left", "net_post_base_right", _COLOR_NET),
    ("net_post_top_left", "net_post_top_right", _COLOR_NET_TOP),
]

_POST_PAIRS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("net_post_base_left", "net_post_top_left", _COLOR_NET_TOP),
    ("net_post_base_right", "net_post_top_right", _COLOR_NET_TOP),
]

_SIDELINE_ORDER_LEFT = [
    "far_baseline_left",
    "far_attack_left",
    "centerline_left",
    "near_attack_left",
    "near_baseline_left",
]
_SIDELINE_ORDER_RIGHT = [
    "far_baseline_right",
    "far_attack_right",
    "centerline_right",
    "near_attack_right",
    "near_baseline_right",
]


def _pixel_map(rec: CalibrationRecord) -> dict[str, tuple[int, int]]:
    return {kp.label: (int(round(kp.x_px)), int(round(kp.y_px))) for kp in rec.keypoints}


def draw_camera_keypoint_overlay(img_bgr: np.ndarray, rec: CalibrationRecord) -> np.ndarray:
    """Return a copy with court lines over keypoints; skips edges when a label is missing."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    pts = _pixel_map(rec)

    def line(a: str, b: str, color: tuple[int, int, int], thickness: int) -> None:
        pa, pb = pts.get(a), pts.get(b)
        if pa is None or pb is None:
            return
        cv2.line(out, pa, pb, color, thickness, lineType=cv2.LINE_AA)

    for chain, color in ((_SIDELINE_ORDER_LEFT, _COLOR_SIDE), (_SIDELINE_ORDER_RIGHT, _COLOR_SIDE)):
        ring = [pts[l] for l in chain if l in pts]
        if len(ring) >= 2:
            arr = np.array([ring], dtype=np.int32)
            cv2.polylines(out, arr, isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    for a, b, c in _LINE_PAIRS:
        line(a, b, c, 3)

    for a, b, c in _POST_PAIRS:
        line(a, b, c, 3)

    for lbl, (px, py) in pts.items():
        cv2.circle(out, (px, py), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(out, (px, py), 5, (40, 220, 255), -1, lineType=cv2.LINE_AA)
        tx = min(w - 160, px + 8)
        ty = max(16, py - 6)
        cv2.putText(out, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    return out


def load_calibration_image(
    rec: CalibrationRecord,
    *,
    region: str,
    local_path: Path | None,
) -> np.ndarray:
    if local_path is not None:
        return load_image_bgr(local_path)
    if not rec.image_s3_bucket or not rec.image_s3_key:
        raise ValueError("missing S3 ref in export — pass --image with a local frame")
    img = fetch_image_bgr(rec.image_s3_bucket, rec.image_s3_key, region=region)
    if img is None:
        img = fetch_image_bgr_boto3(rec.image_s3_bucket, rec.image_s3_key)
    if img is None:
        raise RuntimeError("could not load image from S3; use --image with a local path")
    return img


def load_planar_world_points(geometry_txt: Path) -> dict[str, tuple[float, float]]:
    text = geometry_txt.read_text(encoding="utf-8")
    m = re.search(r"#BEGIN_POINT_TABLE\n(.*?)#END_POINT_TABLE", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no #BEGIN_POINT_TABLE block in {geometry_txt}")
    world: dict[str, tuple[float, float]] = {}
    for raw in m.group(1).strip().splitlines():
        if not raw.strip() or raw.startswith("label"):
            continue
        parts = raw.split("\t")
        if len(parts) < 4:
            parts = raw.split()
        label, sx, sy, planar = parts[0], parts[1], parts[2], parts[3]
        if planar.strip().lower() != "yes":
            continue
        world[label] = (float(sx), float(sy))
    return world


def image_world_correspondences(
    rec: CalibrationRecord,
    world_pts: dict[str, tuple[float, float]],
    *,
    skip_labels: frozenset[str] = _SKIP_LABELS_FOR_H,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pairs_i: list[tuple[float, float]] = []
    pairs_w: list[tuple[float, float]] = []
    used_labels: list[str] = []
    kp_map = {k.label: k for k in rec.keypoints}

    for label in sorted(kp_map):
        if label in skip_labels:
            continue
        if label not in world_pts:
            continue
        k = kp_map[label]
        wx, wy = world_pts[label]
        pairs_i.append((float(k.x_px), float(k.y_px)))
        pairs_w.append((wx, wy))
        used_labels.append(label)

    if len(pairs_i) < 4:
        raise ValueError(
            f"need >=4 planar point pairs for homography; got {len(pairs_i)}. "
            f"Labels used: {used_labels}; skipped (non-planar): {sorted(skip_labels & set(kp_map))}"
        )

    return np.array(pairs_i, dtype=np.float32), np.array(pairs_w, dtype=np.float32), used_labels


def compute_homography(
    image_xy: np.ndarray,
    world_xy: np.ndarray,
    *,
    ransac_thresh_px: float = 4.0,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Return H_world_to_homog_image (λ[u,v,1]^T = H @ [Wx,Wy,1]^T)."""
    H, mask = cv2.findHomography(world_xy, image_xy, cv2.RANSAC, ransac_thresh_px)
    info: dict[str, Any] = {"inliers": None, "rmse_all_px": None, "rmse_inlier_px": None}
    if H is None or mask is None:
        return None, mask, info

    n = len(image_xy)
    wh = np.c_[world_xy, np.ones(n, dtype=np.float32)].T
    pred = H @ wh
    pred = pred[:2] / pred[2]
    errors = np.linalg.norm(pred.T - image_xy, axis=1)
    info["rmse_all_px"] = float(np.sqrt(np.mean(errors**2)))

    inlier_idx = np.where(mask.ravel() == 1)[0]
    if len(inlier_idx) > 0:
        info["rmse_inlier_px"] = float(np.sqrt(np.mean(errors[inlier_idx] ** 2)))
    info["inliers"] = int(mask.sum())

    return H, mask.astype(bool).ravel(), info


def expand_world_bounds_to_playing_court(
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    *,
    margin_m: float,
) -> tuple[float, float, float, float]:
    return (
        min(wx_min, -4.5 - margin_m),
        max(wx_max, 4.5 + margin_m),
        min(wy_min, -9.0 - margin_m),
        max(wy_max, 9.0 + margin_m),
    )


def world_canvas_bounds(
    world_xy_used: np.ndarray,
    *,
    margin_m: float,
) -> tuple[float, float, float, float]:
    """Label hull padded, then expanded so full regulation playing rectangle fits."""
    tight = (
        float(np.min(world_xy_used[:, 0]) - margin_m),
        float(np.max(world_xy_used[:, 0]) + margin_m),
        float(np.min(world_xy_used[:, 1]) - margin_m),
        float(np.max(world_xy_used[:, 1]) + margin_m),
    )
    return expand_world_bounds_to_playing_court(*tight, margin_m=margin_m)


def warp_topdown(
    image_bgr: np.ndarray,
    H_world_to_image: np.ndarray,
    *,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    sx = (wx_max - wx_min) / max(out_w - 1, 1)
    sy = (wy_max - wy_min) / max(out_h - 1, 1)
    A = np.array([[sx, 0.0, wx_min], [0.0, -sy, wy_max], [0.0, 0.0, 1.0]], dtype=np.float64)
    canvas_to_image = H_world_to_image @ A
    warp_m = np.linalg.inv(canvas_to_image)
    return cv2.warpPerspective(
        image_bgr,
        warp_m.astype(np.float64),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30),
    )


def draw_world_rect_overlay(
    canvas_bgr: np.ndarray,
    *,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
) -> None:
    h, w = canvas_bgr.shape[:2]

    def wc(wx: float, wy: float) -> tuple[int, int]:
        ox = (wx - wx_min) / (wx_max - wx_min) * (w - 1)
        oy = (wy_max - wy) / (wy_max - wy_min) * (h - 1)
        return int(round(ox)), int(round(oy))

    outer = np.array(
        [wc(-4.5, 9.0), wc(4.5, 9.0), wc(4.5, -9.0), wc(-4.5, -9.0)],
        dtype=np.int32,
    )
    cv2.polylines(canvas_bgr, [outer], isClosed=True, color=(0, 255, 80), thickness=2, lineType=cv2.LINE_AA)

    def line_w(a: tuple[float, float], b: tuple[float, float], col: tuple[int, int, int]) -> None:
        pa, pb = wc(*a), wc(*b)
        cv2.line(canvas_bgr, pa, pb, col, 2, cv2.LINE_AA)

    line_w((-4.5, 0.0), (4.5, 0.0), (180, 120, 255))
    line_w((-4.5, 3.0), (4.5, 3.0), (80, 200, 220))
    line_w((-4.5, -3.0), (4.5, -3.0), (80, 200, 220))


def save_calibration_npz(
    path: Path,
    *,
    H: np.ndarray,
    used_labels: list[str],
    image_xy: np.ndarray,
    world_xy: np.ndarray,
    mask: np.ndarray | None,
    meta: dict[str, Any],
) -> None:
    meta_bytes = json.dumps(meta, separators=(",", ":"), default=str).encode("utf-8")
    payload: dict[str, Any] = {
        "H_world_to_pixel": H,
        "labels": np.array(used_labels, dtype=object),
        "image_xy": image_xy,
        "world_xy_m": world_xy,
        "meta_json": np.frombuffer(meta_bytes, dtype=np.uint8),
    }
    if mask is not None:
        payload["ransac_inliers"] = mask.astype(np.uint8)
    np.savez_compressed(path, **payload)


def fit_calibration_record_for_db(
    rec: CalibrationRecord,
    geometry_txt: Path,
    *,
    margin_m: float = _MARGIN_M,
    pixels_per_metre: float = _PIXELS_PER_METRE,
    ransac_thresh_px: float = 4.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit homography from keypoints + geometry; return DB column fragment and fit diagnostics.

    ``row_fragment`` keys: ``homography_matrix`` (3×3 nested lists, world→image),
    ``world_wx_min``, ``world_wx_max``, ``world_wy_min``, ``world_wy_max``,
    ``pixels_per_metre``, ``keypoints`` (``[{label, x_px, y_px}, ...]`` sorted by label).

    ``info`` includes RMSE / inlier counts and ``used_labels`` (planar homography subset).
    """
    world_pts = load_planar_world_points(geometry_txt)
    img_xy, w_xy, used_labels = image_world_correspondences(rec, world_pts)
    H, mask, hinfo = compute_homography(img_xy, w_xy, ransac_thresh_px=ransac_thresh_px)
    if H is None:
        raise RuntimeError("findHomography failed")
    wx_min, wx_max, wy_min, wy_max = world_canvas_bounds(w_xy, margin_m=margin_m)
    keypoints_db = [
        {"label": kp.label, "x_px": round(float(kp.x_px), 4), "y_px": round(float(kp.y_px), 4)}
        for kp in sorted(rec.keypoints, key=lambda k: k.label)
    ]
    H_list = np.asarray(H, dtype=np.float64).tolist()
    row_fragment: dict[str, Any] = {
        "homography_matrix": H_list,
        "world_wx_min": float(wx_min),
        "world_wx_max": float(wx_max),
        "world_wy_min": float(wy_min),
        "world_wy_max": float(wy_max),
        "pixels_per_metre": float(pixels_per_metre),
        "keypoints": keypoints_db,
    }
    info: dict[str, Any] = {
        **hinfo,
        "used_labels": list(used_labels),
        "n_planar_pairs": len(used_labels),
        "skipped_non_planar": sorted(_SKIP_LABELS_FOR_H & {k.label for k in rec.keypoints}),
    }
    if mask is not None:
        info["inlier_count_mask"] = int(mask.sum())
    return row_fragment, info


def fit_homography_from_export_task(
    calibration_json: Path,
    *,
    geometry: Path,
    task_index: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], np.ndarray, np.ndarray, list[str], CalibrationRecord]:
    """Load task, correspondences, RANSAC mask/hinfo, and H."""
    world_pts = load_planar_world_points(geometry)
    records = load_calibration_records(calibration_json)
    if not records or task_index < 0 or task_index >= len(records):
        raise ValueError(
            "No calibration records, or bad --task index. "
            "Use a Label Studio JSON export, or normalized payloads "
            "(e.g. output of `python data_labeling/court_keypoints.py export.json`)."
        )
    rec = records[task_index]
    img_xy, w_xy, labels = image_world_correspondences(rec, world_pts)
    H, mask, hinfo = compute_homography(img_xy, w_xy)
    if H is None:
        raise RuntimeError("findHomography failed")
    return H, mask, hinfo, img_xy, w_xy, labels, rec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit court homography + write top-down preview (defaults: from-label canvas, 45 px/m, 1 m margin)."
    )
    parser.add_argument(
        "calibration_json",
        type=Path,
        help="Label Studio export **or** normalized court_keypoints payloads (court_keypoints.py output)",
    )
    parser.add_argument("--geometry", type=Path, default=_GEOMETRY_FILE, help=f"default: {_GEOMETRY_FILE}")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--image", type=Path, default=None, help="Local calibration frame (overrides S3 in export)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_CALIB_DIR / "out",
        help=f"Writes homography.npz + topdown.png (default: {_CALIB_DIR / 'out'})",
    )
    parser.add_argument(
        "--region",
        default=AWS_REGION_FALLBACK,
        help="S3 region for HTTPS fallback (also respects AWS_REGION)",
    )
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION", args.region)

    if not args.calibration_json.is_file():
        print(f"not found: {args.calibration_json}", file=sys.stderr)
        return 1
    if not args.geometry.is_file():
        print(f"not found: {args.geometry}", file=sys.stderr)
        return 1

    try:
        H, mask, hinfo, img_xy, w_xy, labels, rec = fit_homography_from_export_task(
            args.calibration_json,
            geometry=args.geometry,
            task_index=args.task,
        )
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        img = load_calibration_image(rec, region=region, local_path=args.image)
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    wx_min, wx_max, wy_min, wy_max = world_canvas_bounds(w_xy, margin_m=_MARGIN_M)
    out_w = max(2, int(round((wx_max - wx_min) * _PIXELS_PER_METRE)))
    out_h = max(2, int(round((wy_max - wy_min) * _PIXELS_PER_METRE)))

    topdown = warp_topdown(
        img,
        H.astype(np.float64),
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).astype(np.uint8)

    meta = {
        **hinfo,
        "labels_used_planar_H": labels,
        "skipped_non_planar": sorted(_SKIP_LABELS_FOR_H & {k.label for k in rec.keypoints}),
        "world_bounds_xy": (wx_min, wx_max, wy_min, wy_max),
        "pixels_per_metre_requested": _PIXELS_PER_METRE,
        "geometry_file": str(args.geometry.resolve()),
        "ransac_threshold_px_default": 4.0,
        "canvas_mode": _CANVAS_MODE,
        "margin_m": _MARGIN_M,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = (args.out_dir / "homography.npz").resolve()
    top_path = (args.out_dir / "topdown.png").resolve()

    save_calibration_npz(
        npz_path,
        H=H,
        used_labels=labels,
        image_xy=img_xy,
        world_xy=w_xy,
        mask=mask,
        meta=meta,
    )

    draw_world_rect_overlay(
        topdown,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
    )

    if not cv2.imwrite(str(top_path), topdown):
        print(f"could not save {top_path}", file=sys.stderr)
        return 1

    print(f"homography_saved {npz_path}")
    print(f"topdown_preview {top_path}")
    rs = meta.get("rmse_all_px")
    print(
        f"world_canvas_m Wx[{wx_min:.2f},{wx_max:.2f}] Wy[{wy_min:.2f},{wy_max:.2f}] "
        f"rmse_px={rs:.4f} inliers={meta.get('inliers')}/{len(labels)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
