#!/usr/bin/env python3
"""Side-by-side MP4: YOLO skeleton (left) + homography top-down per sampled frame (right).

Resolves everything from Supabase ``clips.id`` + ``court_calibrations`` for that clip's ``source_id``:
downloads the clip MP4 from S3 to a temp file, loads ``H`` and world bounds from the DB, then renders.

Example (repo root, ``.env`` with ``SUPABASE_*`` and ``AWS_*`` for S3):

    python cv-pipeline/pose-detection/pose_side_by_side_video.py --clip-id 42 --fps 2

Requires: ultralytics, torch, opencv, boto3, python-dotenv, supabase (see requirements.txt).
``ffmpeg`` on PATH is recommended: writes a short-lived MPEG-4 (mp4v), then transcodes to H.264
unless you pass ``--no-h264-transcode``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CALIB_DIR = _REPO_ROOT / "cv-pipeline" / "calibration"
_POSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_CALIB_DIR))
sys.path.insert(0, str(_POSE_DIR))

from court_homography import draw_world_rect_overlay, warp_topdown  # noqa: E402
from fetch_s3_clip import download_s3_object  # noqa: E402
from homography_io import homography_arrays_from_court_calibration_row  # noqa: E402

_L_ANKLE = 15
_R_ANKLE = 16


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


def _process_frame(
    frame_bgr: np.ndarray,
    *,
    cv2,
    model,
    H: np.ndarray,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    out_w: int,
    out_h: int,
    imgsz: int,
    conf: float,
    ankle_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    results = model(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]
    skeleton_bgr = r.plot(boxes=False, labels=False, conf=False)

    topdown = warp_topdown(
        frame_bgr,
        H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).copy()
    draw_world_rect_overlay(topdown, wx_min=wx_min, wx_max=wx_max, wy_min=wy_min, wy_max=wy_max)

    kp = r.keypoints
    if kp is not None and kp.xy is not None and kp.xy.shape[0] > 0:
        xy = kp.xy.cpu().numpy()
        if kp.conf is not None:
            kconf = kp.conf.cpu().numpy()
        else:
            kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
        n = xy.shape[0]
        for i in range(n):
            foot = _foot_uv_from_coco17(xy[i], kconf[i], ankle_conf=ankle_conf)
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

    return skeleton_bgr, topdown


def _transcode_mp4_to_h264(src: Path, dst: Path) -> None:
    """Re-encode to H.264 + yuv420p + faststart so NLEs (Premiere, Resolve, etc.) reliably import the file."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _stack_panels(
    sk_bgr: np.ndarray,
    td_bgr: np.ndarray,
    *,
    panel_h: int,
    gap_px: int,
    cv2,
) -> np.ndarray:
    sh, sw = sk_bgr.shape[:2]
    sk_r = cv2.resize(sk_bgr, (max(1, int(round(sw * panel_h / sh))), panel_h), interpolation=cv2.INTER_AREA)
    th, tw = td_bgr.shape[:2]
    td_r = cv2.resize(td_bgr, (max(1, int(round(tw * panel_h / th))), panel_h), interpolation=cv2.INTER_AREA)
    gap = np.full((panel_h, gap_px, 3), 55, dtype=np.uint8)
    return np.hstack([sk_r, gap, td_r])


def main() -> int:
    try:
        import cv2
        from dotenv import load_dotenv
        from ultralytics import YOLO
    except ImportError as e:
        print("Need opencv + ultralytics + torch + python-dotenv.", file=sys.stderr)
        raise SystemExit(1) from e

    p = argparse.ArgumentParser(
        description="Render side-by-side pose | top-down MP4 for one clip (Supabase clips.id + S3 + court_calibrations)."
    )
    p.add_argument(
        "--clip-id",
        type=int,
        required=True,
        help="Primary key ``clips.id`` in Supabase",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target sampling rate and output video FPS (frames written per second of source time)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output MP4 (default: cv-pipeline/pose-detection/out/<clip_filename_stem>_pose_side_by_side.mp4)",
    )
    p.add_argument("--panel-h", type=int, default=720, help="Stacked panel height (both panels scaled to this)")
    p.add_argument("--gap", type=int, default=12, help="Gray strip between panels (pixels)")
    p.add_argument("--weights", type=str, default="yolov8s-pose.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--ankle-conf", type=float, default=0.25)
    p.add_argument(
        "--no-h264-transcode",
        action="store_true",
        help="Keep OpenCV's MPEG-4 (mp4v) output only; skips ffmpeg re-encode if you use this flag.",
    )
    p.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-west-2"),
        help="S3 region for boto3 download",
    )
    args = p.parse_args()

    if args.fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 1

    load_dotenv(_REPO_ROOT / ".env")

    sup_keys = ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
    aws_keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    for key in sup_keys + aws_keys:
        if not os.environ.get(key):
            print(f"missing env {key}", file=sys.stderr)
            return 1

    from src import db

    client = db.get_supabase_client()
    clip = db.get_clip_by_id(client, args.clip_id)
    if clip is None:
        print(f"no clips row for id={args.clip_id}", file=sys.stderr)
        return 1

    source_id = str(clip["source_id"])
    cal = db.get_court_calibration(client, source_id)
    if cal is None:
        print(
            f"no court_calibrations for source_id={source_id!r} — push calibration for this source first",
            file=sys.stderr,
        )
        return 1

    H, wx_min, wx_max, wy_min, wy_max, out_w, out_h = homography_arrays_from_court_calibration_row(cal)

    stem = Path(str(clip["filename"])).stem
    out_path = args.out or (_POSE_DIR / "out" / f"{stem}_pose_side_by_side.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    region = os.environ.get("AWS_REGION", args.region)
    bucket = str(clip["s3_bucket"])
    key = str(clip["s3_key"])

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", prefix=f".clip_{args.clip_id}_dl_", dir=str(out_path.parent))
    os.close(fd)
    tmp_video = Path(tmp_name)

    print(f"clip id={args.clip_id} source_id={source_id} s3://{bucket}/{key}", flush=True)
    try:
        download_s3_object(bucket, key, tmp_video, region=region)
    except (RuntimeError, SystemExit) as e:
        print(str(e), file=sys.stderr)
        tmp_video.unlink(missing_ok=True)
        return 1

    cap = cv2.VideoCapture(str(tmp_video))
    if not cap.isOpened():
        print(f"could not open downloaded video: {tmp_video}", file=sys.stderr)
        tmp_video.unlink(missing_ok=True)
        return 1

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval = max(1, int(round(src_fps / args.fps)))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_out = (n_src // frame_interval) if n_src > 0 else "?"
    print(
        f"source_fps≈{src_fps:.2f}  sample_every={frame_interval} frames  output_fps={args.fps:.2f}  est_out_frames≈{est_out}",
        flush=True,
    )

    model = YOLO(args.weights)
    writer: cv2.VideoWriter | None = None
    out_idx = 0
    frame_idx = 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    use_transcode = ffmpeg_ok and not args.no_h264_transcode
    raw_path: Path | None = None
    if use_transcode:
        raw_fd, raw_name = tempfile.mkstemp(suffix=".mp4", prefix=".pose_side_raw_", dir=str(out_path.parent))
        raw_path = Path(raw_name)
        os.close(raw_fd)
        writer_path = raw_path
    else:
        writer_path = out_path
        if not ffmpeg_ok and not args.no_h264_transcode:
            print(
                "ffmpeg not on PATH: writing MPEG-4 (mp4v). Some editors won't import this; "
                "install ffmpeg or re-run with ffmpeg available for an automatic H.264 pass.",
                file=sys.stderr,
            )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            sk, td = _process_frame(
                frame,
                cv2=cv2,
                model=model,
                H=H,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                out_w=out_w,
                out_h=out_h,
                imgsz=args.imgsz,
                conf=args.conf,
                ankle_conf=args.ankle_conf,
            )
            composite = _stack_panels(sk, td, panel_h=args.panel_h, gap_px=args.gap, cv2=cv2)
            h, w = composite.shape[:2]
            w_even = (w // 2) * 2
            h_even = (h // 2) * 2
            if w_even != w or h_even != h:
                composite = cv2.resize(composite, (w_even, h_even))

            if writer is None:
                writer = cv2.VideoWriter(
                    str(writer_path),
                    fourcc,
                    float(args.fps),
                    (w_even, h_even),
                )
                if not writer.isOpened():
                    print(f"VideoWriter failed to open for {writer_path}", file=sys.stderr)
                    return 1

            writer.write(composite)
            out_idx += 1
            if out_idx % 20 == 0:
                print(f"  wrote {out_idx} frames", flush=True)

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        tmp_video.unlink(missing_ok=True)

    if use_transcode and raw_path is not None:
        try:
            _transcode_mp4_to_h264(raw_path, out_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(
                f"H.264 transcode failed ({e!r}); intermediate file kept at: {raw_path}",
                file=sys.stderr,
            )
            return 1
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(
            f"wrote {out_idx} frames -> {out_path.resolve()} (H.264 via ffmpeg; editor-friendly)",
            flush=True,
        )
    else:
        print(f"wrote {out_idx} frames -> {out_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
