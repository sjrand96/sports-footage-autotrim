#!/usr/bin/env python3
"""Side-by-side camera frame + ground-plane warp. Drag on the LEFT view to probe (Wx,Wy) and map to the warp.

Uses the saved homography (--npz), or recomputes from a Label Studio export like court_homography.py.

Quit: ``q``. From repo root: ``.venv/bin/python cv-pipeline/calibration/court_homography_interactive.py``

Requires OpenCV GUI (needs a display server). Uses ``opencv-python`` or ``opencv-python-headless``
may lack HighGUI depending on platform; if ``imshow`` fails, install ``opencv-python`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV required: .venv/bin/pip install -e '.[cv]'") from e

from court_homography import (
    compute_homography,
    draw_world_rect_overlay,
    expand_world_bounds_to_playing_court,
    image_world_correspondences,
    load_planar_world_points,
    warp_topdown,
    world_canvas_bounds,
)
from court_overlay import fetch_image_bgr, fetch_image_bgr_boto3, load_image_bgr
from label_studio_keypoints import parse_keypoint_export_file

_GEOMETRY_DEFAULT = _CALIB_DIR / "fivb_court_geometry.txt"
_NPZ_DEFAULT = _CALIB_DIR / "court_homography.npz"
_EXPORT_DEFAULT = _CALIB_DIR / "project-7-at-2026-05-05-19-45-119e8837.json"


def _uv_to_world_m(H_world_to_px: np.ndarray, u: float, v: float) -> tuple[float, float]:
    Hi = np.linalg.inv(H_world_to_px.astype(np.float64))
    p = Hi @ np.array([u, v, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def _world_to_canvas_px(
    wx: float,
    wy: float,
    *,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    out_w: int,
    out_h: int,
) -> tuple[float, float]:
    sx = (wx_max - wx_min) / max(out_w - 1, 1)
    sy = (wy_max - wy_min) / max(out_h - 1, 1)
    ox = (wx - wx_min) / sx
    oy = (wy_max - wy) / sy
    return ox, oy


def load_fit_from_npz(path: Path) -> tuple[np.ndarray, dict]:
    z = np.load(path, allow_pickle=True)
    H = np.asarray(z["H_world_to_pixel"], dtype=np.float64)
    meta_raw = bytes(np.asarray(z["meta_json"]).tobytes()).decode("utf-8")
    meta = json.loads(meta_raw)
    return H, meta


def load_fit_from_export(args: argparse.Namespace) -> tuple[np.ndarray, tuple[float, float, float, float], float, dict]:
    geo = Path(args.geometry)
    world_pts = load_planar_world_points(geo)
    records = parse_keypoint_export_file(Path(args.export_json))
    rec = records[args.task]
    img_xy, w_xy, labels = image_world_correspondences(rec, world_pts)
    H, mask, hinfo = compute_homography(img_xy, w_xy)
    if H is None:
        raise RuntimeError("findHomography failed")
    bounds = world_canvas_bounds(w_xy, mode=args.canvas, margin_m=args.margin_m)
    ppm = args.ppm
    meta_out = {"labels": labels, "inliers_info": hinfo, "canvas_mode": args.canvas}
    return H.astype(np.float64), bounds, float(ppm), meta_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Click/drag on camera frame → see mapped point on warp")
    parser.add_argument("--npz", type=Path, default=None, help="Saved court_homography.npz (skipped if absent)")
    parser.add_argument("--export-json", type=Path, default=_EXPORT_DEFAULT)
    parser.add_argument("--geometry", type=Path, default=_GEOMETRY_DEFAULT)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--canvas", choices=("from-labels", "full-regulation"), default="from-labels")
    parser.add_argument("--margin-m", type=float, default=1.0)
    parser.add_argument("--ppm", type=float, default=45.0)
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="scaled display height before stitching (preserve aspect ratios)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=12,
        help="divider strip width between panels (neutral gray)",
    )
    args = parser.parse_args()

    H: np.ndarray
    wx_min: float
    wx_max: float
    wy_min: float
    wy_max: float
    ppm: float

    npz_path = args.npz or _NPZ_DEFAULT
    if npz_path.is_file():
        H, meta = load_fit_from_npz(npz_path)
        wx_min, wx_max, wy_min, wy_max = meta["world_bounds_xy"]
        # Older .npz saved a label-only slab; always include full playing court for overlays.
        wx_min, wx_max, wy_min, wy_max = expand_world_bounds_to_playing_court(
            wx_min, wx_max, wy_min, wy_max, margin_m=args.margin_m
        )
        ppm = float(meta.get("pixels_per_metre_requested", args.ppm))
    else:
        H, bounds, ppm, meta = load_fit_from_export(args)
        wx_min, wx_max, wy_min, wy_max = bounds

    out_w = max(2, int(round((wx_max - wx_min) * ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * ppm)))

    img: np.ndarray
    if args.image:
        img = load_image_bgr(Path(args.image))
    else:
        records = parse_keypoint_export_file(Path(args.export_json))
        if not records:
            sys.exit("no tasks in export; pass --image or --export-json")
        rec = records[args.task]
        if not rec.image_s3_bucket or not rec.image_s3_key:
            sys.exit("export missing S3 ref; pass --image")
        img = fetch_image_bgr(rec.image_s3_bucket, rec.image_s3_key, region=args.region)
        if img is None:
            img = fetch_image_bgr_boto3(rec.image_s3_bucket, rec.image_s3_key)
        if img is None:
            sys.exit("could not download frame from S3; pass local --image")

    warp_canvas = warp_topdown(
        img,
        H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).copy()
    draw_world_rect_overlay(warp_canvas, wx_min=wx_min, wx_max=wx_max, wy_min=wy_min, wy_max=wy_max)

    src_h, src_w = img.shape[:2]
    mh = args.max_height
    if src_h > mh:
        scale_cam = mh / src_h
    else:
        scale_cam = 1.0
    cam_vis_w = max(2, int(round(src_w * scale_cam)))
    cam_vis_h = max(2, int(round(src_h * scale_cam)))

    warp_vis_w = max(2, round(warp_canvas.shape[1] * (cam_vis_h / warp_canvas.shape[0])))
    warp_vis_h = cam_vis_h

    cam_scale_x = cam_vis_w / float(src_w)
    cam_scale_y = cam_vis_h / float(src_h)

    warp_scale_x = warp_vis_w / float(out_w)
    warp_scale_y = warp_vis_h / float(out_h)

    divider = cam_vis_w
    gap_w = args.gap
    win_w = cam_vis_w + gap_w + warp_vis_w
    hud_h = 36

    dragging: dict[str, object] = {"active": False, "have_point": False}

    def redraw() -> None:
        ux = dragging.get("ux") if dragging.get("have_point") else None
        uy = dragging.get("uy") if dragging.get("have_point") else None

        left = cv2.resize(img, (cam_vis_w, cam_vis_h), interpolation=cv2.INTER_AREA)
        if ux is not None:
            xd = round(float(ux) * cam_scale_x)
            yd = round(float(uy) * cam_scale_y)
            cv2.drawMarker(left, (xd, yd), color=(60, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
            cv2.circle(left, (xd, yd), 7, color=(40, 200, 255), thickness=2, lineType=cv2.LINE_AA)

        right = cv2.resize(warp_canvas, (warp_vis_w, warp_vis_h), interpolation=cv2.INTER_AREA)
        if ux is not None:
            wx, wy = _uv_to_world_m(H, float(ux), float(uy))
            ox_f, oy_f = _world_to_canvas_px(
                wx,
                wy,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                out_w=out_w,
                out_h=out_h,
            )
            xd = round(ox_f * warp_scale_x)
            yd = round(oy_f * warp_scale_y)
            if 0 <= xd < warp_vis_w and 0 <= yd < warp_vis_h:
                cv2.drawMarker(right, (xd, yd), color=(60, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
                cv2.circle(right, (xd, yd), 7, color=(40, 200, 255), thickness=2, lineType=cv2.LINE_AA)

            hud = (
                f"u,v={float(ux):.1f},{float(uy):.1f}px  Wx,Wy={wx:.2f},{wy:.2f}m  topdown=({ox_f:.1f},{oy_f:.1f})"
            )
        else:
            hud = "Left: camera — click / drag. Right: ground plane (metres).  q quit"

        gap = np.full((cam_vis_h, gap_w, 3), 55, dtype=np.uint8)
        split = np.hstack([left, gap, right])
        bar = np.full((hud_h, win_w, 3), 28, dtype=np.uint8)
        cv2.putText(bar, hud, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([split, bar])
        cv2.imshow("homography explorer", canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, _p: object) -> None:
        if y < 0 or y >= cam_vis_h:
            return
        if x < 0 or x >= divider:
            return

        if event == cv2.EVENT_LBUTTONUP:
            dragging["active"] = False
            return

        do_update = False
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging["active"] = True
            do_update = True
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            do_update = bool(dragging["active"])

        if not do_update:
            return

        ux = x / cam_scale_x
        uy = y / cam_scale_y
        if not (0 <= ux < src_w and 0 <= uy < src_h):
            return
        dragging["have_point"] = True
        dragging["ux"], dragging["uy"] = ux, uy
        redraw()

    cv2.namedWindow("homography explorer", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("homography explorer", on_mouse)

    redraw()
    print("Window open: drag with left mouse on CAMERA (left) panel; q closes.", flush=True)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
