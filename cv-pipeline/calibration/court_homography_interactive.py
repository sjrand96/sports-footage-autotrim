#!/usr/bin/env python3
"""Interactive homography: camera + top-down warp. Optional --camera-overlay export.

First positional arg accepts either a Label Studio JSON export **or** normalized payloads
from ``python data_labeling/court_keypoints.py`` (same as ``court_homography.py``).

From repo root (with OpenCV GUI / display server):

    .venv/bin/python cv-pipeline/calibration/court_homography_interactive.py path/to/export.json
    .venv/bin/python cv-pipeline/calibration/court_homography_interactive.py path/to/export.json \\
        --npz cv-pipeline/calibration/out/homography.npz
    .venv/bin/python cv-pipeline/calibration/court_homography_interactive.py court_payloads.json --camera-overlay

Quit: q or Esc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV required: pip install -e '.[cv]'") from e

from data_labeling.court_keypoints import load_calibration_records

from court_homography import (
    _MARGIN_M,
    _PIXELS_PER_METRE,
    draw_camera_keypoint_overlay,
    draw_world_rect_overlay,
    expand_world_bounds_to_playing_court,
    fit_homography_from_export_task,
    load_calibration_image,
    warp_topdown,
    world_canvas_bounds,
)

# Display (fixed — adjust in code if needed)
_MAX_DISPLAY_HEIGHT = 900
_GAP_PX = 12

_DEFAULT_OVERLAY_PATH = _CALIB_DIR / "out" / "camera_overlay.png"


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


def main() -> int:
    p = argparse.ArgumentParser(description="Explore homography: click/drag on camera (left) → see world map (right).")
    p.add_argument(
        "calibration_json",
        type=Path,
        help="Label Studio export **or** normalized payloads (court_keypoints.py output); selects frame/keypoints",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Use saved homography.npz instead of refitting from export",
    )
    p.add_argument("--geometry", type=Path, default=_CALIB_DIR / "fivb_court_geometry.txt")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--image", type=Path, default=None, help="Local frame path (overrides S3 in export)")
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="S3 region for image fetch",
    )
    p.add_argument(
        "--camera-overlay",
        type=Path,
        nargs="?",
        const=_DEFAULT_OVERLAY_PATH,
        default=None,
        help="Save keypoint/line overlay on camera frame; default path if flag given with no path: %(const)s",
    )
    args = p.parse_args()
    region = os.environ.get("AWS_REGION", args.region)

    if not args.calibration_json.is_file():
        print(f"not found: {args.calibration_json}", file=sys.stderr)
        return 1

    records = load_calibration_records(args.calibration_json)
    if not records or args.task < 0 or args.task >= len(records):
        print("no records or bad --task", file=sys.stderr)
        return 1
    rec = records[args.task]

    if args.npz is not None:
        if not args.npz.is_file():
            print(f"not found: {args.npz}", file=sys.stderr)
            return 1
        H, meta = load_fit_from_npz(args.npz)
        wx_min, wx_max, wy_min, wy_max = meta["world_bounds_xy"]
        wx_min, wx_max, wy_min, wy_max = expand_world_bounds_to_playing_court(
            wx_min, wx_max, wy_min, wy_max, margin_m=_MARGIN_M
        )
        ppm = float(meta.get("pixels_per_metre_requested", _PIXELS_PER_METRE))
    else:
        if not args.geometry.is_file():
            print(f"not found: {args.geometry}", file=sys.stderr)
            return 1
        try:
            H, _mask, _info, img_xy, w_xy, _labels, _ = fit_homography_from_export_task(
                args.calibration_json,
                geometry=args.geometry,
                task_index=args.task,
            )
        except (ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 1
        _ = img_xy
        wx_min, wx_max, wy_min, wy_max = world_canvas_bounds(w_xy, margin_m=_MARGIN_M)
        ppm = _PIXELS_PER_METRE

    try:
        img = load_calibration_image(rec, region=region, local_path=args.image)
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    if args.camera_overlay is not None:
        out_path = args.camera_overlay
        out_path.parent.mkdir(parents=True, exist_ok=True)
        layered = draw_camera_keypoint_overlay(img, rec)
        if not cv2.imwrite(str(out_path), layered):
            print(f"failed to write {out_path}", file=sys.stderr)
            return 1
        print(f"wrote {out_path.resolve()}", flush=True)

    out_w = max(2, int(round((wx_max - wx_min) * ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * ppm)))

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
    mh = _MAX_DISPLAY_HEIGHT
    scale_cam = mh / src_h if src_h > mh else 1.0
    cam_vis_w = max(2, int(round(src_w * scale_cam)))
    cam_vis_h = max(2, int(round(src_h * scale_cam)))

    warp_vis_w = max(2, round(warp_canvas.shape[1] * (cam_vis_h / warp_canvas.shape[0])))
    warp_vis_h = cam_vis_h

    cam_scale_x = cam_vis_w / float(src_w)
    cam_scale_y = cam_vis_h / float(src_h)
    warp_scale_x = warp_vis_w / float(out_w)
    warp_scale_y = warp_vis_h / float(out_h)

    divider = cam_vis_w
    gap_w = _GAP_PX
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

            hud = f"u,v={float(ux):.1f},{float(uy):.1f}px  Wx,Wy={wx:.2f},{wy:.2f}m  topdown=({ox_f:.1f},{oy_f:.1f})"
        else:
            hud = "Left: camera — click / drag. Right: ground plane (metres).  q quit"

        gap = np.full((cam_vis_h, gap_w, 3), 55, dtype=np.uint8)
        split = np.hstack([left, gap, right])
        bar = np.full((hud_h, win_w, 3), 28, dtype=np.uint8)
        cv2.putText(bar, hud, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([split, bar])
        cv2.imshow("homography explorer", canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, _p: object) -> None:
        if y < 0 or y >= cam_vis_h or x < 0 or x >= divider:
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
    print("Window open: drag on CAMERA (left); q closes.", flush=True)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
