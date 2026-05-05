#!/usr/bin/env python3
"""Draw court line overlays on a calibration frame from Label Studio keypoint export.

From repo root (use the project venv):

    .venv/bin/pip install -e '.[cv]'   # once
    .venv/bin/python cv-pipeline/calibration/court_overlay.py
    # writes cv-pipeline/calibration/court_overlay_preview.png by default
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

from label_studio_keypoints import CalibrationRecord, parse_keypoint_export_file

_CALIB_DIR = Path(__file__).resolve().parent

try:
    import cv2
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing OpenCV/NumPy. From repo root: .venv/bin/pip install -e '.[cv]'"
    ) from e

# BGR
_COLOR_BASELINE = (40, 40, 230)
_COLOR_ATTACK = (60, 200, 80)
_COLOR_CENTER = (230, 160, 60)
_COLOR_NET = (0, 220, 255)
_COLOR_NET_TOP = (80, 140, 255)
_COLOR_SIDE = (200, 200, 200)

# Transverse lines where both endpoints are labeled
_LINE_PAIRS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("far_baseline_left", "far_baseline_right", _COLOR_BASELINE),
    ("near_baseline_left", "near_baseline_right", _COLOR_BASELINE),
    ("far_attack_left", "far_attack_right", _COLOR_ATTACK),
    ("near_attack_left", "near_attack_right", _COLOR_ATTACK),
    ("centerline_left", "centerline_right", _COLOR_CENTER),
    ("net_post_base_left", "net_post_base_right", _COLOR_NET),
    ("net_post_top_left", "net_post_top_right", _COLOR_NET_TOP),
]

# Net posts (vertical in world space; slanted in perspective)
_POST_PAIRS: list[tuple[str, str, tuple[int, int, int]]] = [
    ("net_post_base_left", "net_post_top_left", _COLOR_NET_TOP),
    ("net_post_base_right", "net_post_top_right", _COLOR_NET_TOP),
]

# Far → near along each sideline (intersections with transverse lines only)
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


def fetch_image_bgr(bucket: str, key: str, *, region: str) -> np.ndarray | None:
    """Load image from public S3 object; return BGR array or None."""
    safe_key = requests.utils.quote(key, safe="/")
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{safe_key}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException:
        return None
    buf = np.frombuffer(r.content, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return im


def fetch_image_bgr_boto3(bucket: str, key: str) -> np.ndarray | None:
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    try:
        bc = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        obj = bc.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
    except ClientError:
        return None
    buf = np.frombuffer(body, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_image_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"could not read image: {path}")
    return im


def draw_court_overlay(img_bgr: np.ndarray, rec: CalibrationRecord) -> np.ndarray:
    """Return a copy with lines and keypoint markers; skips edges when a label is missing."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    pts = _pixel_map(rec)

    def line(a: str, b: str, color: tuple[int, int, int], thickness: int) -> None:
        pa, pb = pts.get(a), pts.get(b)
        if pa is None or pb is None:
            return
        cv2.line(out, pa, pb, color, thickness, lineType=cv2.LINE_AA)

    # Sidelines as polylines (use only labeled points, keep far→near order)
    for chain, color in (
        (_SIDELINE_ORDER_LEFT, _COLOR_SIDE),
        (_SIDELINE_ORDER_RIGHT, _COLOR_SIDE),
    ):
        ring = [pts[l] for l in chain if l in pts]
        if len(ring) >= 2:
            arr = np.array([ring], dtype=np.int32)
            cv2.polylines(out, arr, isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    for a, b, c in _LINE_PAIRS:
        line(a, b, c, 3)

    for a, b, c in _POST_PAIRS:
        line(a, b, c, 3)

    # Keypoint markers
    for lbl, (px, py) in pts.items():
        cv2.circle(out, (px, py), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(out, (px, py), 5, (40, 220, 255), -1, lineType=cv2.LINE_AA)
        tx = min(w - 160, px + 8)
        ty = max(16, py - 6)
        cv2.putText(
            out,
            lbl,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            lbl,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Render court line overlay on calibration frame.")
    p.add_argument(
        "export_json",
        nargs="?",
        type=Path,
        default=_CALIB_DIR / "project-7-at-2026-05-05-19-45-119e8837.json",
        help="Label Studio JSON export",
    )
    p.add_argument("--image", type=Path, help="Local frame (overrides S3 fetch from export)")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_CALIB_DIR / "court_overlay_preview.png",
        help=f"default: {_CALIB_DIR / 'court_overlay_preview.png'}",
    )
    p.add_argument("--task", type=int, default=0, help="Task index if export has multiple")
    p.add_argument("--region", default="us-west-2", help="S3 region for HTTPS URL / boto3")
    args = p.parse_args()

    if not args.export_json.is_file():
        print(f"not found: {args.export_json}", file=sys.stderr)
        return 1

    records = parse_keypoint_export_file(args.export_json)
    if not records:
        print("no calibration records in export", file=sys.stderr)
        return 1
    if args.task < 0 or args.task >= len(records):
        print(f"task index out of range (0..{len(records) - 1})", file=sys.stderr)
        return 1

    rec = records[args.task]

    if args.image is not None:
        im = load_image_bgr(args.image)
    else:
        if not rec.image_s3_bucket or not rec.image_s3_key:
            print("missing S3 location; pass --image ", file=sys.stderr)
            return 1
        im = fetch_image_bgr(rec.image_s3_bucket, rec.image_s3_key, region=args.region)
        if im is None:
            im = fetch_image_bgr_boto3(rec.image_s3_bucket, rec.image_s3_key)
        if im is None:
            print(
                "could not download frame from S3 (public URL failed; check network or AWS creds for boto3)",
                file=sys.stderr,
            )
            return 1

    overlay = draw_court_overlay(im, rec)

    out_path = args.output.expanduser().resolve()
    if not cv2.imwrite(str(out_path), overlay):
        print(f"failed to write {out_path}", file=sys.stderr)
        return 1
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
