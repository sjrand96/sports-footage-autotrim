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
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_WRIST = 9
_R_WRIST = 10
_L_HIP = 11
_R_HIP = 12
_L_KNEE = 13
_R_KNEE = 14
_L_ANKLE = 15
_R_ANKLE = 16


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


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _knee_angle_deg(xy: np.ndarray, conf: np.ndarray, *, kp_conf_thresh: float) -> float:
    angles: list[float] = []
    for hip, knee, ankle in ((_L_HIP, _L_KNEE, _L_ANKLE), (_R_HIP, _R_KNEE, _R_ANKLE)):
        if conf[hip] < kp_conf_thresh or conf[knee] < kp_conf_thresh or conf[ankle] < kp_conf_thresh:
            continue
        angles.append(_angle_deg(xy[hip], xy[knee], xy[ankle]))
    if not angles:
        return float("nan")
    return float(np.mean(angles))


def _shoulder_width_px(xy: np.ndarray, conf: np.ndarray, *, kp_conf_thresh: float) -> float | None:
    if conf[_L_SHOULDER] < kp_conf_thresh or conf[_R_SHOULDER] < kp_conf_thresh:
        return None
    return float(np.linalg.norm(xy[_L_SHOULDER] - xy[_R_SHOULDER]))


def _wrist_above_shoulder(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    kp_conf_thresh: float,
) -> tuple[bool, tuple[float, float] | None]:
    if conf[_L_SHOULDER] < kp_conf_thresh and conf[_R_SHOULDER] < kp_conf_thresh:
        return False, None

    shoulder_ys = []
    if conf[_L_SHOULDER] >= kp_conf_thresh:
        shoulder_ys.append(float(xy[_L_SHOULDER][1]))
    if conf[_R_SHOULDER] >= kp_conf_thresh:
        shoulder_ys.append(float(xy[_R_SHOULDER][1]))
    if not shoulder_ys:
        return False, None

    shoulder_y = min(shoulder_ys)
    wrist_points: list[tuple[float, float]] = []
    if conf[_L_WRIST] >= kp_conf_thresh:
        wrist_points.append((float(xy[_L_WRIST][0]), float(xy[_L_WRIST][1])))
    if conf[_R_WRIST] >= kp_conf_thresh:
        wrist_points.append((float(xy[_R_WRIST][0]), float(xy[_R_WRIST][1])))

    if not wrist_points:
        return False, None

    wrist_points.sort(key=lambda p: p[1])
    top_wrist = wrist_points[0]
    if top_wrist[1] < shoulder_y:
        return True, top_wrist
    return False, None


def _high_five_pair_count(wrist_points: list[tuple[float, float]], shoulder_width_px: float) -> int:
    if len(wrist_points) < 2:
        return 0
    thresh = max(10.0, shoulder_width_px * 1.2)
    count = 0
    for i in range(len(wrist_points)):
        for j in range(i + 1, len(wrist_points)):
            dist = float(np.linalg.norm(np.asarray(wrist_points[i]) - np.asarray(wrist_points[j])))
            if dist <= thresh:
                count += 1
    return count


def _flow_mag_stats(prev_gray: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    import cv2

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = mag.astype(np.float64)
    return {
        "flow_mag_mean": float(np.mean(mag)),
        "flow_mag_std": float(np.std(mag, ddof=0)),
        "flow_mag_p90": float(np.percentile(mag, 90)),
        "flow_mag_p95": float(np.percentile(mag, 95)),
    }


def _centroid_speed_stats(
    feats: dict[str, Any],
    *,
    prev_centroids: dict[str, tuple[float, float]] | None,
    prev_inter: float | None,
    prev_ts: float | None,
    cur_ts: float,
) -> dict[str, float]:
    if prev_centroids is None or prev_ts is None:
        return {
            "camera_side_centroid_speed_m": float("nan"),
            "opposite_side_centroid_speed_m": float("nan"),
            "inter_centroid_speed_m": float("nan"),
        }

    dt = cur_ts - prev_ts
    if dt <= 0:
        return {
            "camera_side_centroid_speed_m": float("nan"),
            "opposite_side_centroid_speed_m": float("nan"),
            "inter_centroid_speed_m": float("nan"),
        }

    cam = (float(feats["camera_side_centroid_wx_m"]), float(feats["camera_side_centroid_wy_m"]))
    opp = (float(feats["opposite_side_centroid_wx_m"]), float(feats["opposite_side_centroid_wy_m"]))
    inter = float(feats["inter_centroid_dist_m"])

    def _speed(prev_xy: tuple[float, float], cur_xy: tuple[float, float]) -> float:
        if not np.isfinite(prev_xy[0]) or not np.isfinite(prev_xy[1]) or not np.isfinite(cur_xy[0]) or not np.isfinite(cur_xy[1]):
            return float("nan")
        return float(np.linalg.norm(np.asarray(cur_xy) - np.asarray(prev_xy)) / dt)

    cam_speed = _speed(prev_centroids.get("camera", (float("nan"), float("nan"))), cam)
    opp_speed = _speed(prev_centroids.get("opposite", (float("nan"), float("nan"))), opp)
    if prev_inter is None or not np.isfinite(prev_inter) or not np.isfinite(inter):
        inter_speed = float("nan")
    else:
        inter_speed = float(abs(inter - prev_inter) / dt)

    return {
        "camera_side_centroid_speed_m": cam_speed,
        "opposite_side_centroid_speed_m": opp_speed,
        "inter_centroid_speed_m": inter_speed,
    }


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
    squat_count = 0
    wrists_above_shoulder_count = 0
    n_pose_instances_raw = 0
    knee_angles: list[float] = []
    wrist_points: list[tuple[float, float]] = []
    shoulder_widths: list[float] = []

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

            knee_angle = _knee_angle_deg(xy[i], kconf[i], kp_conf_thresh=kp_conf_thresh)
            if np.isfinite(knee_angle):
                knee_angles.append(knee_angle)
                if knee_angle < 120.0:
                    squat_count += 1

            shoulder_width = _shoulder_width_px(xy[i], kconf[i], kp_conf_thresh=kp_conf_thresh)
            if shoulder_width is not None and np.isfinite(shoulder_width):
                shoulder_widths.append(shoulder_width)

            wrist_ok, wrist_point = _wrist_above_shoulder(xy[i], kconf[i], kp_conf_thresh=kp_conf_thresh)
            if wrist_ok and wrist_point is not None:
                wrists_above_shoulder_count += 1
                wrist_points.append(wrist_point)

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

    if knee_angles:
        knee_angle_mean = float(np.mean(knee_angles))
        knee_angle_min = float(np.min(knee_angles))
    else:
        knee_angle_mean = float("nan")
        knee_angle_min = float("nan")

    squat_ratio = float(squat_count / n_total) if n_total > 0 else 0.0
    shoulder_width_med = float(np.median(shoulder_widths)) if shoulder_widths else 60.0
    high_five_pair_count = _high_five_pair_count(wrist_points, shoulder_width_med)

    return {
        "n_players_total": n_total,
        "n_front_row": n_front_row,
        "n_back_row": n_back_row,
        "n_camera_side": n_camera_side,
        "n_opposite_side": n_opposite_side,
        "median_nearest_neighbor_dist": median_nn,
        "hands_above_head_count": int(hands_above_count),
        "knee_angle_mean_deg": knee_angle_mean,
        "knee_angle_min_deg": knee_angle_min,
        "squat_count": int(squat_count),
        "squat_ratio": float(squat_ratio),
        "wrists_above_shoulder_count": int(wrists_above_shoulder_count),
        "high_five_pair_count": int(high_five_pair_count),
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
    prev_gray: np.ndarray | None = None
    prev_centroids: dict[str, tuple[float, float]] | None = None
    prev_inter: float | None = None
    prev_ts: float | None = None
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
            cur_ts = float(frame_idx / src_fps)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                flow_stats = {
                    "flow_mag_mean": float("nan"),
                    "flow_mag_std": float("nan"),
                    "flow_mag_p90": float("nan"),
                    "flow_mag_p95": float("nan"),
                }
            else:
                flow_stats = _flow_mag_stats(prev_gray, gray)
            prev_gray = gray

            speed_stats = _centroid_speed_stats(
                feats,
                prev_centroids=prev_centroids,
                prev_inter=prev_inter,
                prev_ts=prev_ts,
                cur_ts=cur_ts,
            )
            prev_centroids = {
                "camera": (float(feats["camera_side_centroid_wx_m"]), float(feats["camera_side_centroid_wy_m"])),
                "opposite": (
                    float(feats["opposite_side_centroid_wx_m"]),
                    float(feats["opposite_side_centroid_wy_m"]),
                ),
            }
            prev_inter = float(feats["inter_centroid_dist_m"])
            prev_ts = cur_ts
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": cur_ts,
                    **feats,
                    **flow_stats,
                    **speed_stats,
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
