"""Per-frame YOLO pose + homography feature extraction (full source fps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_extraction.core.paths import (
    ANKLE_CONF_DEFAULT,
    DET_CONF_DEFAULT,
    IMGSZ_DEFAULT,
    KP_CONF_DEFAULT,
    WEIGHTS_DEFAULT,
    pose_module,
)
from feature_extraction.core.spatial import chunk1_spatial_dict

_NOSE = 0
_L_WRIST = 9
_R_WRIST = 10


def _median_nearest_neighbor_distance(world_xy: np.ndarray) -> float:
    n = int(world_xy.shape[0])
    if n < 2:
        return float("nan")
    diffs = world_xy[:, None, :] - world_xy[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = dists.min(axis=1)
    return float(np.median(nearest))


def _hands_above_head_for_player(xy: np.ndarray, conf: np.ndarray, *, kp_conf_thresh: float) -> bool:
    nose_conf = float(conf[_NOSE])
    lw_conf = float(conf[_L_WRIST])
    rw_conf = float(conf[_R_WRIST])
    if nose_conf < kp_conf_thresh or (lw_conf < kp_conf_thresh and rw_conf < kp_conf_thresh):
        return False
    wrist_ys: list[float] = []
    if lw_conf >= kp_conf_thresh:
        wrist_ys.append(float(xy[_L_WRIST][1]))
    if rw_conf >= kp_conf_thresh:
        wrist_ys.append(float(xy[_R_WRIST][1]))
    if not wrist_ys:
        return False
    nose_y = float(xy[_NOSE][1])
    return min(wrist_ys) < nose_y


def compute_feature_row_from_yolo_result(
    result: Any,
    *,
    H: np.ndarray,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    ankle_conf: float,
    kp_conf_thresh: float,
) -> dict[str, Any]:
    """Numeric features for one frame (keys match ``FEATURE_COLUMNS``)."""
    pose_mod = pose_module()
    kp = result.keypoints
    world_points: list[tuple[float, float]] = []
    camera_world: list[tuple[float, float]] = []
    opposite_world: list[tuple[float, float]] = []
    hands_above_count = 0
    n_pose_instances_raw = 0

    if kp is not None and kp.xy is not None and kp.xy.shape[0] > 0:
        xy = kp.xy.cpu().numpy()
        n_pose_instances_raw = int(xy.shape[0])
        if kp.conf is not None:
            kconf = kp.conf.cpu().numpy()
        else:
            kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

        for i in range(xy.shape[0]):
            foot_uv = pose_mod._foot_uv_from_coco17(xy[i], kconf[i], ankle_conf=ankle_conf)
            if foot_uv is None:
                continue
            wx, wy = pose_mod._image_uv_to_world_m(H, float(foot_uv[0]), float(foot_uv[1]))
            if not (wx_min <= wx <= wx_max and wy_min <= wy <= wy_max):
                continue
            world_points.append((wx, wy))
            if wy < 0.0:
                camera_world.append((wx, wy))
            else:
                opposite_world.append((wx, wy))
            if _hands_above_head_for_player(xy[i], kconf[i], kp_conf_thresh=kp_conf_thresh):
                hands_above_count += 1

    world = np.asarray(world_points, dtype=np.float64)
    n_total = int(world.shape[0])
    if n_total > 0:
        wy_axis = world[:, 1]
        n_camera_side = int(np.sum(wy_axis < 0.0))
        n_opposite_side = int(np.sum(wy_axis >= 0.0))
        n_front_row = int(np.sum(np.abs(wy_axis) < 3.0))
        n_back_row = int(np.sum(np.abs(wy_axis) >= 3.0))
        median_nn = _median_nearest_neighbor_distance(world)
    else:
        n_camera_side = 0
        n_opposite_side = 0
        n_front_row = 0
        n_back_row = 0
        median_nn = float("nan")

    cam_xy = np.asarray(camera_world, dtype=np.float64).reshape(-1, 2)
    opp_xy = np.asarray(opposite_world, dtype=np.float64).reshape(-1, 2)
    spatial = chunk1_spatial_dict(
        n_pose_instances_raw=n_pose_instances_raw,
        camera_world_xy=cam_xy,
        opposite_world_xy=opp_xy,
    )

    return {
        "n_players_total": n_total,
        "n_front_row": n_front_row,
        "n_back_row": n_back_row,
        "n_camera_side": n_camera_side,
        "n_opposite_side": n_opposite_side,
        "median_nearest_neighbor_dist": median_nn,
        "hands_above_head_count": int(hands_above_count),
        **spatial,
    }


def extract_features_for_clip(
    *,
    video_path: Path,
    H: np.ndarray,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    weights: str = WEIGHTS_DEFAULT,
    imgsz: int = IMGSZ_DEFAULT,
    det_conf: float = DET_CONF_DEFAULT,
    ankle_conf: float = ANKLE_CONF_DEFAULT,
    kp_conf_thresh: float = KP_CONF_DEFAULT,
    progress_every: int = 300,
    max_frames: int | None = None,
    frames_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract one row per decoded video frame (~full source fps).

    ``max_frames`` is for local smoke tests only; omit for production full-clip runs.
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Need opencv-python + ultralytics + torch installed.") from exc

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    model = YOLO(weights)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"extract {video_path.name}: source_fps≈{src_fps:.2f} frames≈{n_src} (every frame)", flush=True)

    rows: list[dict[str, Any]] = []
    frame_idx = 0
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frames_dir is not None:
                import cv2

                out_jpg = frames_dir / f"{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_jpg), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            result = model(frame_bgr, imgsz=imgsz, conf=det_conf, verbose=False)[0]
            feats = compute_feature_row_from_yolo_result(
                result,
                H=H,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                ankle_conf=ankle_conf,
                kp_conf_thresh=kp_conf_thresh,
            )
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": float(frame_idx / src_fps),
                    **feats,
                }
            )
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break
            if progress_every > 0 and frame_idx % progress_every == 0:
                print(f"  processed {frame_idx} frames", flush=True)
    finally:
        cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"no frames decoded from video: {video_path}")

    meta = {"source_fps": src_fps, "n_source_frames": n_src}
    return pd.DataFrame(rows), meta
