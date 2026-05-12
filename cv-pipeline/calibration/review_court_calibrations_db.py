#!/usr/bin/env python3
"""Visual QA for ``court_calibrations`` in Supabase: reference frame + keypoint lines | top-down warp.

Left: stored keypoints drawn on the reference image (same style as ``court_homography`` overlays).
Right: top-down using the stored homography and world bounds from the row.

Requires ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` (e.g. from repo ``.env``). S3 read uses the same
HTTPS / boto3 path as other calibration tools.

Keys (window focused): **n** next, **p** previous, **s** save PNG to ``--out-dir``, **q** / Esc quit.

Example (repo root, venv with cv + supabase + dotenv):

    python cv-pipeline/calibration/review_court_calibrations_db.py
    python cv-pipeline/calibration/review_court_calibrations_db.py --source-id dQw4w9WgXcQ
    python cv-pipeline/calibration/review_court_calibrations_db.py --export-dir cv-pipeline/calibration/out/review_db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _CALIB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("OpenCV required: pip install -e '.[cv]'") from e

from court_homography import (  # noqa: E402
    draw_camera_keypoint_overlay,
    draw_world_rect_overlay,
    load_calibration_image,
    warp_topdown,
)
from data_labeling.court_keypoints import calibration_record_from_court_calibrations_row  # noqa: E402
from homography_io import homography_arrays_from_court_calibration_row  # noqa: E402


def _composite(row: dict, *, region: str) -> tuple[np.ndarray, str]:
    rec = calibration_record_from_court_calibrations_row(row)
    img = load_calibration_image(rec, region=region, local_path=None)
    H, wx_min, wx_max, wy_min, wy_max, out_w, out_h = homography_arrays_from_court_calibration_row(row)
    left = draw_camera_keypoint_overlay(img, rec)
    right = warp_topdown(
        img,
        H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).copy()
    draw_world_rect_overlay(right, wx_min=wx_min, wx_max=wx_max, wy_min=wy_min, wy_max=wy_max)

    gap = 12
    mh = 900
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    scale = min(1.0, mh / max(lh, rh))
    lw2, lh2 = max(2, int(round(lw * scale))), max(2, int(round(lh * scale)))
    rw2, rh2 = max(2, int(round(rw * (lh2 / rh)))), lh2
    lrs = cv2.resize(left, (lw2, lh2), interpolation=cv2.INTER_AREA)
    rrs = cv2.resize(right, (rw2, rh2), interpolation=cv2.INTER_AREA)
    g = np.full((lh2, gap, 3), 55, dtype=np.uint8)
    sid = str(row.get("source_id") or "")
    bar_h = 32
    canvas = np.hstack([lrs, g, rrs])
    bar = np.full((bar_h, canvas.shape[1], 3), 28, dtype=np.uint8)
    txt = f"{sid}  |  n next  p prev  s save  q quit"
    cv2.putText(bar, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    out = np.vstack([canvas, bar])
    return out, sid


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("pip install python-dotenv", file=sys.stderr)
        return 1

    from src import db

    p = argparse.ArgumentParser(description="Browse court_calibrations from Supabase (camera | top-down).")
    p.add_argument("--source-id", default=None, help="Only this YouTube id (single row)")
    p.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="If set, write one PNG per row and exit (no GUI)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_CALIB_DIR / "out" / "review_db",
        help="Default directory for --export-dir or per-frame saves (key s)",
    )
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = p.parse_args()

    load_dotenv()
    missing = [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY") if not os.environ.get(k)]
    if missing:
        print(f"missing env: {', '.join(missing)}", file=sys.stderr)
        return 1

    region = os.environ.get("AWS_REGION", args.region)
    client = db.get_supabase_client()

    if args.source_id:
        row = db.get_court_calibration(client, args.source_id.strip())
        if not row:
            print(f"no court_calibrations row for source_id={args.source_id!r}", file=sys.stderr)
            return 1
        rows = [row]
    else:
        rows = db.list_court_calibrations(client)

    if not rows:
        print("no court_calibrations rows to show", file=sys.stderr)
        return 1

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            try:
                comp, sid = _composite(row, region=region)
            except Exception as e:  # noqa: BLE001
                print(f"skip {row.get('source_id')}: {e}", file=sys.stderr)
                continue
            path = args.export_dir / f"{sid}.png"
            if not cv2.imwrite(str(path), comp):
                print(f"failed write {path}", file=sys.stderr)
                return 1
            print(path.resolve(), flush=True)
        return 0

    idx = 0
    win = "court_calibrations review"

    def show() -> None:
        row = rows[idx]
        comp, sid = _composite(row, region=region)
        cv2.imshow(win, comp)
        print(f"[{idx + 1}/{len(rows)}] {sid}", flush=True)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("q")):
            break
        if k in (ord("n"), ord("N")):
            idx = (idx + 1) % len(rows)
            show()
        elif k in (ord("p"), ord("P")):
            idx = (idx - 1) % len(rows)
            show()
        elif k in (ord("s"), ord("S")):
            args.out_dir.mkdir(parents=True, exist_ok=True)
            row = rows[idx]
            comp, sid = _composite(row, region=region)
            path = args.out_dir / f"{sid}.png"
            if cv2.imwrite(str(path), comp):
                print(f"saved {path.resolve()}", flush=True)
            else:
                print(f"failed {path}", file=sys.stderr)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
