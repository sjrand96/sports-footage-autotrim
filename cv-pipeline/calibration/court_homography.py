"""Fit ground-plane homography from Label Studio keypoints + FIVB world points; inverse-warp to a top-down image.

Each output pixel is one (Wx, Wy) on the floor in metres; the homography decides which camera pixel
samples that turf. Regions of the rectangle that the camera never sees fall outside the frame and
warp to gray—especially if you use ``--canvas full-regulation`` without covering the whole court.
``--canvas from-labels`` starts from your labelled hull plus margin, then expands to include the full
18×9 m playing rectangle so drawn baselines are not clipped when you skipped near-baseline labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_CALIB_DIR = Path(__file__).resolve().parent

import numpy as np

from label_studio_keypoints import CalibrationRecord, parse_keypoint_export_file

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "OpenCV required. From repo root: .venv/bin/pip install -e '.[cv]'"
    ) from e

# Imported after fetch helpers
from court_overlay import (
    fetch_image_bgr,
    fetch_image_bgr_boto3,
    load_image_bgr,
)


_GEOMETRY_FILE = _CALIB_DIR / "fivb_court_geometry.txt"

# Non-coplanar with floor — including them poisons a ground-plane solve.
_SKIP_LABELS_FOR_H = frozenset({"net_post_top_left", "net_post_top_right"})


def load_planar_world_points(geometry_txt: Path) -> dict[str, tuple[float, float]]:
    text = geometry_txt.read_text(encoding="utf-8")
    m = re.search(
        r"#BEGIN_POINT_TABLE\n(.*?)#END_POINT_TABLE",
        text,
        flags=re.DOTALL,
    )
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

    # Stable order helps debugging reproducibility
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

    src = np.array(pairs_i, dtype=np.float32)
    dst = np.array(pairs_w, dtype=np.float32)
    return src, dst, used_labels


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
    wh = np.c_[world_xy, np.ones(n, dtype=np.float32)].T  # 3 x N
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
    """Grow a world Axis-Aligned Bounding Box so the 18×9 m playing outline fits (sidelines ±4.5 m, baselines ±9 m).

    Use when labels omit the near baseline but you still want regulation lines fully visible in the warp.
    ``margin_m`` is applied the same way as in :func:`world_canvas_bounds` (pad outside that outline).
    """
    return (
        min(wx_min, -4.5 - margin_m),
        max(wx_max, 4.5 + margin_m),
        min(wy_min, -9.0 - margin_m),
        max(wy_max, 9.0 + margin_m),
    )


def world_canvas_bounds(
    world_xy_used: np.ndarray,
    *,
    mode: str,
    margin_m: float,
) -> tuple[float, float, float, float]:
    """Return (wx_min, wx_max, wy_min, wy_max) in metres for the top-down raster."""
    if mode == "from-labels":
        tight = (
            float(np.min(world_xy_used[:, 0]) - margin_m),
            float(np.max(world_xy_used[:, 0]) + margin_m),
            float(np.min(world_xy_used[:, 1]) - margin_m),
            float(np.max(world_xy_used[:, 1]) + margin_m),
        )
        return expand_world_bounds_to_playing_court(*tight, margin_m=margin_m)
    if mode == "full-regulation":
        # Sidelines ±4.5 m; posts for FIVB World/Official at ±5.5 m — include posts in canvas width.
        return (
            -5.5 - margin_m,
            5.5 + margin_m,
            -9.0 - margin_m,
            9.0 + margin_m,
        )
    raise ValueError(f"unknown canvas mode {mode!r}")


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
    """Inverse-warp so each output pixel samples the camera image at the corresponding world ground point."""
    sx = (wx_max - wx_min) / max(out_w - 1, 1)
    sy = (wy_max - wy_min) / max(out_h - 1, 1)

    A = np.array(
        [
            [sx, 0.0, wx_min],
            [0.0, -sy, wy_max],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Composition: homogeneous image_xy ∝ H @ (A @ homogeneous canvas_xy).
    # warpPerspective(backward remap) interprets its matrix as inverse of dst→src; use inv().
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
    """Draw playing-court rectangle (sidelines × baselines) in canvas pixel space — in-place."""
    h, w = canvas_bgr.shape[:2]

    def wc(wx: float, wy: float) -> tuple[int, int]:
        ox = (wx - wx_min) / (wx_max - wx_min) * (w - 1)
        oy = (wy_max - wy) / (wy_max - wy_min) * (h - 1)
        return int(round(ox)), int(round(oy))

    outer = np.array(
        [
            wc(-4.5, 9.0),
            wc(4.5, 9.0),
            wc(4.5, -9.0),
            wc(-4.5, -9.0),
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas_bgr, [outer], isClosed=True, color=(0, 255, 80), thickness=2, lineType=cv2.LINE_AA)

    def line_w(a: tuple[float, float], b: tuple[float, float], col: tuple[int, int, int]) -> None:
        pa, pb = wc(*a), wc(*b)
        cv2.line(canvas_bgr, pa, pb, col, 2, cv2.LINE_AA)

    # Centre line across width
    line_w((-4.5, 0.0), (4.5, 0.0), (180, 120, 255))
    # Attack lines / baselines helpers
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


def main() -> int:
    parser = argparse.ArgumentParser(description="FIVB homography + top-down warp from LS keypoint export")
    parser.add_argument(
        "export_json",
        nargs="?",
        type=Path,
        default=_CALIB_DIR / "project-7-at-2026-05-05-19-45-119e8837.json",
    )
    parser.add_argument("--geometry", type=Path, default=_GEOMETRY_FILE)
    parser.add_argument("--image", type=Path, help="Local frame instead of downloading from export S3 ref")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--ppm", type=float, default=45.0, help="pixels per metre for top-down canvas")
    parser.add_argument(
        "--canvas",
        choices=("from-labels", "full-regulation"),
        default="from-labels",
        help="from-labels: label hull + margin, expanded to include full 18×9 m playing rectangle (so baselines draw in-frame). "
        "full-regulation: widen to ±5.5 m for posts plus margin — more grey outside FOV.",
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=1.0,
        help="extra metres padded around the chosen canvas bounds",
    )
    parser.add_argument("-o", "--output-npz", type=Path, default=_CALIB_DIR / "court_homography.npz")
    parser.add_argument("--topdown", type=Path, default=_CALIB_DIR / "court_topdown_preview.png")
    args = parser.parse_args()

    if not args.export_json.is_file():
        print(f"not found: {args.export_json}", file=sys.stderr)
        return 1
    if not args.geometry.is_file():
        print(f"not found: {args.geometry}", file=sys.stderr)
        return 1

    world_pts = load_planar_world_points(args.geometry)
    records = parse_keypoint_export_file(args.export_json)
    if not records or args.task < 0 or args.task >= len(records):
        print("missing task or bad --task index", file=sys.stderr)
        return 1
    rec = records[args.task]

    img_xy, w_xy, labels = image_world_correspondences(rec, world_pts)
    H, mask, hinfo = compute_homography(img_xy, w_xy)
    if H is None:
        print("findHomography failed", file=sys.stderr)
        return 1

    if args.image:
        img = load_image_bgr(args.image)
    else:
        if not rec.image_s3_bucket or not rec.image_s3_key:
            print("missing S3 ref in export — use --image", file=sys.stderr)
            return 1
        img = fetch_image_bgr(rec.image_s3_bucket, rec.image_s3_key, region=args.region)
        if img is None:
            img = fetch_image_bgr_boto3(rec.image_s3_bucket, rec.image_s3_key)
        if img is None:
            print("could not load image", file=sys.stderr)
            return 1

    wx_min, wx_max, wy_min, wy_max = world_canvas_bounds(
        w_xy, mode=args.canvas, margin_m=args.margin_m
    )
    out_w = max(2, int(round((wx_max - wx_min) * args.ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * args.ppm)))

    topdown = warp_topdown(
        img,
        H.astype(np.float64),
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    )

    td_u8 = topdown.astype(np.uint8)

    meta = {
        **hinfo,
        "labels_used_planar_H": labels,
        "skipped_non_planar": sorted(_SKIP_LABELS_FOR_H & {k.label for k in rec.keypoints}),
        "world_bounds_xy": (wx_min, wx_max, wy_min, wy_max),
        "pixels_per_metre_requested": args.ppm,
        "geometry_file": str(args.geometry.resolve()),
        "ransac_threshold_px_default": 4.0,
        "canvas_mode": args.canvas,
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_npz(
        args.output_npz.expanduser().resolve(),
        H=H,
        used_labels=labels,
        image_xy=img_xy,
        world_xy=w_xy,
        mask=mask,
        meta=meta,
    )

    draw_world_rect_overlay(
        td_u8,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
    )
    td_path = args.topdown.expanduser().resolve()
    td_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(td_path), td_u8):
        print(f"could not save {td_path}", file=sys.stderr)
        return 1

    nz = args.output_npz.expanduser().resolve()
    print(f"homography_saved {nz}")
    print(f"topdown_preview {td_path.resolve()}")
    print(
        f"world_canvas_m Wx[{wx_min:.2f},{wx_max:.2f}] Wy[{wy_min:.2f},{wy_max:.2f}] mode={args.canvas}",
        file=sys.stderr,
    )
    rs = meta.get("rmse_all_px")
    rs_s = f"{rs:.4f}" if rs is not None else "n/a"
    print(
        f"rmse_px={rs_s} inliers={meta.get('inliers')}/{len(labels)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())