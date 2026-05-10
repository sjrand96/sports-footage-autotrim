#!/usr/bin/env python3
"""Experiment: YOLOv8-pose on the calibration frame → two PNGs.

1. **Skeleton** on the camera frame (same style as the notebook: ``plot(..., boxes=False, labels=False, conf=False)``).
2. **Top-down** warp with regulation overlay + ankle-midpoint circles projected to court coordinates.

Uses the same frame as ``data_labeling`` court payloads (S3 bucket/key in JSON) and the saved
``homography.npz`` from ``court_homography.py``.

Requires (same stack as ``notebooks/placeholder_notebook.ipynb``):
    pip install ultralytics torch  # see requirements.txt

Example:
    python cv-pipeline/pose-detection/foot_topdown_experiment.py cv-pipeline/calibration/court_payloads.json \\
        --npz cv-pipeline/calibration/out/homography.npz \\
        --out-skeleton cv-pipeline/pose-detection/out/players_skeleton.png \\
        -o cv-pipeline/pose-detection/out/players_topdown.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CALIB_DIR = _REPO_ROOT / "cv-pipeline" / "calibration"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CALIB_DIR))

from court_homography import (  # noqa: E402
    draw_world_rect_overlay,
    load_calibration_image,
    warp_topdown,
)
from data_labeling.court_keypoints import load_calibration_records  # noqa: E402

# COCO 17 keypoints: 15 left ankle, 16 right ankle (proxy for feet on ground)
_L_ANKLE = 15
_R_ANKLE = 16


def _load_homography_npz(path: Path) -> tuple[np.ndarray, float, float, float, float, int, int]:
    z = np.load(path, allow_pickle=True)
    H = np.asarray(z["H_world_to_pixel"], dtype=np.float64)
    meta = json.loads(bytes(np.asarray(z["meta_json"]).tobytes()).decode("utf-8"))
    wx_min, wx_max, wy_min, wy_max = meta["world_bounds_xy"]
    ppm = float(meta.get("pixels_per_metre_requested", 45.0))
    out_w = max(2, int(round((wx_max - wx_min) * ppm)))
    out_h = max(2, int(round((wy_max - wy_min) * ppm)))
    return H, wx_min, wx_max, wy_min, wy_max, out_w, out_h


def _image_uv_to_world_m(H_world_to_px: np.ndarray, u: float, v: float) -> tuple[float, float]:
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
) -> tuple[int, int]:
    ox = (wx - wx_min) / (wx_max - wx_min) * (out_w - 1)
    oy = (wy_max - wy) / (wy_max - wy_min) * (out_h - 1)
    return int(round(ox)), int(round(oy))


def _foot_uv_from_coco17(xy: np.ndarray, conf: np.ndarray, *, ankle_conf: float) -> tuple[float, float] | None:
    """Mid-ankle in image pixels, or single ankle if only one passes confidence."""
    la = xy[_L_ANKLE]
    ra = xy[_R_ANKLE]
    lc = float(conf[_L_ANKLE])
    rc = float(conf[_R_ANKLE])
    if lc < ankle_conf and rc < ankle_conf:
        return None
    if lc < ankle_conf:
        return float(ra[0]), float(ra[1])
    if rc < ankle_conf:
        return float(la[0]), float(la[1])
    return float((la[0] + ra[0]) / 2.0), float((la[1] + ra[1]) / 2.0)


def main() -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print("Need opencv + ultralytics (and torch). See requirements.txt / notebook.", file=sys.stderr)
        raise SystemExit(1) from e

    p = argparse.ArgumentParser(description="YOLO pose feet → top-down circles on homography warp.")
    p.add_argument("calibration_json", type=Path, help="Court payload or Label Studio export (see court_keypoints)")
    p.add_argument(
        "--npz",
        type=Path,
        default=_CALIB_DIR / "out" / "homography.npz",
        help="homography.npz from court_homography.py",
    )
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--image", type=Path, default=None, help="Override frame (else load from JSON S3 ref)")
    p.add_argument(
        "--out-skeleton",
        type=Path,
        default=_REPO_ROOT / "cv-pipeline" / "pose-detection" / "out" / "players_skeleton.png",
        help="Camera frame with keypoints + skeleton (notebook-style plot)",
    )
    p.add_argument(
        "--out-topdown",
        "-o",
        type=Path,
        default=_REPO_ROOT / "cv-pipeline" / "pose-detection" / "out" / "players_topdown.png",
        help="Top-down warp with foot circles",
    )
    p.add_argument("--weights", type=str, default="yolov8s-pose.pt", help="Ultralytics pose weights (see notebook)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--ankle-conf", type=float, default=0.25, help="Min keypoint conf to use ankles for foot proxy")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = p.parse_args()

    if not args.calibration_json.is_file():
        print(f"not found: {args.calibration_json}", file=sys.stderr)
        return 1
    if not args.npz.is_file():
        print(f"not found: {args.npz}", file=sys.stderr)
        return 1

    records = load_calibration_records(args.calibration_json)
    if not records or args.task < 0 or args.task >= len(records):
        print("bad --task or no calibration records", file=sys.stderr)
        return 1
    rec = records[args.task]

    region = os.environ.get("AWS_REGION", args.region)
    try:
        img = load_calibration_image(rec, region=region, local_path=args.image)
    except (ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 1

    H, wx_min, wx_max, wy_min, wy_max, out_w, out_h = _load_homography_npz(args.npz)
    topdown = warp_topdown(
        img,
        H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).copy()
    draw_world_rect_overlay(topdown, wx_min=wx_min, wx_max=wx_max, wy_min=wy_min, wy_max=wy_max)

    model = YOLO(args.weights)
    results = model(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
    r = results[0]

    # BGR: keypoints + skeleton only (matches notebooks/placeholder_notebook.ipynb)
    skeleton_bgr = r.plot(boxes=False, labels=False, conf=False)
    args.out_skeleton.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out_skeleton), skeleton_bgr):
        print(f"failed to write {args.out_skeleton}", file=sys.stderr)
        return 1

    kp = r.keypoints
    if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
        print("no pose detections", file=sys.stderr)
    else:
        xy = kp.xy.cpu().numpy()
        if kp.conf is not None:
            kconf = kp.conf.cpu().numpy()
        else:
            kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
        n = xy.shape[0]
        for i in range(n):
            foot = _foot_uv_from_coco17(xy[i], kconf[i], ankle_conf=args.ankle_conf)
            if foot is None:
                continue
            wx, wy = _image_uv_to_world_m(H, foot[0], foot[1])
            cx, cy = _world_to_canvas_px(
                wx,
                wy,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                out_w=out_w,
                out_h=out_h,
            )
            hue = int(180 * i / max(n, 1)) % 180
            col = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
            col = (int(col[0]), int(col[1]), int(col[2]))
            cv2.circle(topdown, (cx, cy), 12, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(topdown, (cx, cy), 13, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    args.out_topdown.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out_topdown), topdown):
        print(f"failed to write {args.out_topdown}", file=sys.stderr)
        return 1
    print(args.out_skeleton.resolve())
    print(args.out_topdown.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
