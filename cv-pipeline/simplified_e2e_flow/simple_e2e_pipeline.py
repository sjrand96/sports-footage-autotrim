#!/usr/bin/env python3
"""Run the simplified single-clip E2E pipeline.

Follows `cv-pipeline/simplified_e2e_flow/simple_e2e_plan.md`:
1) Download clip from S3 (unless cached)
2) Per-frame features from YOLO + homography from Supabase ``court_calibrations``
3) Latest timeline annotation from Supabase → ``is_playing`` labels
4) Placeholder XGBoost + parquet outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_E2E_FLOW_DIR = Path(__file__).resolve().parent
if str(_E2E_FLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_FLOW_DIR))
POSE_DIR = REPO_ROOT / "cv-pipeline" / "pose-detection"
CALIB_DIR = REPO_ROOT / "cv-pipeline" / "calibration"
FETCH_SCRIPT = POSE_DIR / "fetch_s3_clip.py"
POSE_SCRIPT = POSE_DIR / "pose_side_by_side_video.py"
DEFAULT_CACHE_DIR = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "cache"
DEFAULT_BUCKET = "sports-footage-autotrim-bucket"
DEFAULT_REGION = "us-west-2"
DEFAULT_LABEL_FPS = 30.0
DEFAULT_S3_PREFIX = "clips/"
WEIGHTS_DEFAULT = "yolov8s-pose.pt"
IMGSZ_DEFAULT = 1280
DET_CONF_DEFAULT = 0.15
ANKLE_CONF_DEFAULT = 0.25
KP_CONF_DEFAULT = 0.25
POSE_CONF_DEFAULT = 0.3

from e2e_feature_columns import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_FLOAT_FILLNA_COLS,
    INTEGER_E2E_COLUMNS,
    NEW_FEATURE_COLUMNS,
)

from pose_feature_spatial import chunk1_spatial_dict  # noqa: E402

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
_CLIP_KEY_RE = re.compile(r"^clips/(?P<source_id>[^/]+)/(?P=source_id)_(?P<clip_index>\d+)\.mp4$", re.IGNORECASE)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FETCH_MOD = _load_module("fetch_s3_clip_module", FETCH_SCRIPT)
POSE_MOD = _load_module("pose_side_by_side_module", POSE_SCRIPT)

if str(CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_DIR))
from homography_io import homography_arrays_from_court_calibration_row  # noqa: E402


def _local_clip_path(source_id: str, clip_index: int) -> Path:
    filename = f"{source_id}_{clip_index:03d}.mp4"
    return REPO_ROOT / "cv-pipeline" / "pose-detection" / "media" / "clips" / source_id / filename


def _clip_from_s3_key(key: str) -> tuple[str, int] | None:
    m = _CLIP_KEY_RE.match(key.strip())
    if not m:
        return None
    return m.group("source_id"), int(m.group("clip_index"))


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _fetch_all_rows(client: Any, table: str, select_cols: str, *, page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        res = client.table(table).select(select_cols).range(offset, offset + page_size - 1).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_annotated_clip_keys(db_client: Any) -> set[tuple[str, int]]:
    """Return {(source_id, clip_index)} for clips with at least one annotation row."""
    ann_rows = _fetch_all_rows(db_client, "annotations", "clip_id")
    clip_ids = sorted({int(r["clip_id"]) for r in ann_rows if r.get("clip_id") is not None})
    if not clip_ids:
        return set()

    keys: set[tuple[str, int]] = set()
    for chunk in _chunked(clip_ids, chunk_size=500):
        res = db_client.table("clips").select("id,source_id,clip_index").in_("id", chunk).execute()
        for row in res.data or []:
            sid = row.get("source_id")
            cidx = row.get("clip_index")
            if sid is None or cidx is None:
                continue
            keys.add((str(sid), int(cidx)))
    return keys


def list_s3_clips(bucket: str, region: str, prefix: str) -> list[tuple[str, int, str]]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 listing. Install with `pip install boto3`.") from exc

    client = boto3.client("s3", region_name=region)
    paginator = client.get_paginator("list_objects_v2")
    clips: list[tuple[str, int, str]] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents") or []
        for obj in contents:
            key = obj.get("Key")
            if not isinstance(key, str):
                continue
            parsed = _clip_from_s3_key(key)
            if parsed is None:
                continue
            source_id, clip_index = parsed
            clips.append((source_id, clip_index, f"s3://{bucket}/{key}"))

    clips.sort(key=lambda row: (row[0], row[1]))
    return clips


def pick_random_s3_clips(
    *,
    bucket: str,
    region: str,
    prefix: str,
    n: int,
    seed: int,
    allowed_clip_keys: set[tuple[str, int]] | None = None,
) -> list[tuple[str, int, str]]:
    if n <= 0:
        raise ValueError("n must be > 0")

    clips = list_s3_clips(bucket=bucket, region=region, prefix=prefix)
    if allowed_clip_keys is not None:
        clips = [c for c in clips if (c[0], c[1]) in allowed_clip_keys]
    if not clips:
        if allowed_clip_keys is None:
            raise RuntimeError(f"no clip keys found in s3://{bucket}/{prefix}")
        raise RuntimeError(
            f"no annotated clip keys found in s3://{bucket}/{prefix}. "
            "Either there are no annotations yet, or the S3 prefix/bucket doesn't overlap annotated clips."
        )

    take = min(n, len(clips))
    rng = random.Random(seed)
    return rng.sample(clips, k=take)


def ensure_local_clip(*, s3_uri: str, local_path: Path, region: str) -> None:
    if local_path.is_file():
        print(f"clip exists, skipping download: {local_path}")
        return

    bucket, key = FETCH_MOD._parse_s3_uri(s3_uri)
    print(f"downloading clip: s3://{bucket}/{key}")
    FETCH_MOD.download_s3_object(bucket, key, local_path, region=region)
    print(f"downloaded to: {local_path}")


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


def _wrist_above_shoulder(xy: np.ndarray, conf: np.ndarray, *, kp_conf_thresh: float) -> tuple[bool, tuple[float, float] | None]:
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


def compute_e2e_feature_row_from_yolo_result(
    result: Any,
    *,
    H: np.ndarray,
    wx_min: float,
    wx_max: float,
    wy_min: float,
    wy_max: float,
    ankle_conf: float,
    kp_conf_thresh: float,
    pose_conf_thresh: float,
) -> dict[str, Any]:
    """Numeric feature dict for one frame; keys match ``FEATURE_COLUMNS`` (no ``frame_idx`` / ``timestamp_sec``).

    Used by ``extract_features_for_clip`` and by ``pose-based-feature-extraction/feature_lab.py`` for viz.
    """
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
            pose_conf = float(np.mean(kconf[i]))
            if pose_conf < pose_conf_thresh:
                continue
            foot_uv = POSE_MOD._foot_uv_from_coco17(xy[i], kconf[i], ankle_conf=ankle_conf)
            if foot_uv is None:
                continue

            wx, wy = POSE_MOD._image_uv_to_world_m(H, float(foot_uv[0]), float(foot_uv[1]))
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
    target_fps: float,
    weights: str,
    imgsz: int,
    det_conf: float,
    ankle_conf: float,
    kp_conf_thresh: float,
    pose_conf_thresh: float,
) -> pd.DataFrame:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Need opencv-python + ultralytics + torch installed.") from exc

    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    model = YOLO(weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval = max(1, int(round(src_fps / target_fps)))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_out = (n_src // frame_interval) if n_src > 0 else "?"
    print(
        f"source_fps≈{src_fps:.2f} sample_every={frame_interval} output_fps={target_fps:.2f} est_frames≈{est_out}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    frame_idx = 0
    sampled = 0
    prev_gray: np.ndarray | None = None
    prev_centroids: dict[str, tuple[float, float]] | None = None
    prev_inter: float | None = None
    prev_ts: float | None = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            result = model(frame_bgr, imgsz=imgsz, conf=det_conf, verbose=False)[0]
            feats = compute_e2e_feature_row_from_yolo_result(
                result,
                H=H,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                ankle_conf=ankle_conf,
                kp_conf_thresh=kp_conf_thresh,
                pose_conf_thresh=pose_conf_thresh,
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
                "opposite": (float(feats["opposite_side_centroid_wx_m"]), float(feats["opposite_side_centroid_wy_m"])),
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
            sampled += 1
            if sampled % 20 == 0:
                print(f"  processed {sampled} sampled frames", flush=True)

            frame_idx += 1
    finally:
        cap.release()

    return pd.DataFrame(rows)


def _extract_playing_ranges_seconds(payload: dict[str, Any], label_fps: float) -> list[tuple[float, float]]:
    ranges_sec: list[tuple[float, float]] = []
    ann = payload.get("label_studio_annotation")
    if isinstance(ann, dict):
        result = ann.get("result")
        if isinstance(result, list):
            for item in result:
                if not isinstance(item, dict):
                    continue
                value = item.get("value")
                if not isinstance(value, dict):
                    continue
                labels = value.get("timelinelabels")
                if labels != ["Playing"]:
                    continue
                item_ranges = value.get("ranges")
                if not isinstance(item_ranges, list):
                    continue
                for r in item_ranges:
                    if not isinstance(r, dict):
                        continue
                    start = r.get("start")
                    end = r.get("end")
                    if start is None or end is None:
                        continue
                    s = float(min(start, end)) / label_fps
                    e = float(max(start, end)) / label_fps
                    ranges_sec.append((s, e))

    return ranges_sec


def _get_db_helpers() -> Any:
    sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers

    return db_helpers


def fetch_latest_annotation_payload(db_helpers: Any, client: Any, source_id: str, clip_index: int) -> dict[str, Any]:
    clip_row = db_helpers.get_clip(client, source_id, clip_index)
    if clip_row is None:
        raise RuntimeError(f"clip not found in Supabase: source_id={source_id} clip_index={clip_index}")

    clip_id = int(clip_row["id"])
    res = (
        client.table("annotations")
        .select("id,payload,exported_at")
        .eq("clip_id", clip_id)
        .order("exported_at", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(f"no annotations found for clip_id={clip_id} ({source_id}_{clip_index:03d})")

    payload = res.data[0].get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("latest annotation row has invalid payload")
    return payload


def add_ground_truth_labels(df_feat: pd.DataFrame, payload: dict[str, Any], label_fps: float) -> pd.DataFrame:
    ranges_sec = _extract_playing_ranges_seconds(payload, label_fps=label_fps)
    ts = df_feat["timestamp_sec"].to_numpy(dtype=np.float64)
    labels = np.zeros(ts.shape[0], dtype=bool)
    for start_sec, end_sec in ranges_sec:
        labels |= (ts >= start_sec) & (ts <= end_sec)

    out = df_feat.copy()
    out["is_playing"] = labels
    return out


def _float_fillna_cols_for_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in INTEGER_E2E_COLUMNS]


def train_and_predict(
    df_labeled: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required. Install with `pip install xgboost`.") from exc

    from sklearn.metrics import f1_score, precision_score, recall_score

    model_input = df_labeled[feature_columns].copy()
    float_fill_cols = _float_fillna_cols_for_features(feature_columns)
    for _col in float_fill_cols:
        model_input[_col] = model_input[_col].fillna(-1.0)
    X = model_input.to_numpy(dtype=np.float32)
    y = df_labeled["is_playing"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        pred = np.full_like(y, fill_value=int(y[0] if len(y) else 0))
    else:
        # Single-clip placeholder path: train/predict on same clip to validate E2E wiring.
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X, y)
        pred = model.predict(X).astype(int)

    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))

    pred_df = df_labeled[["frame_idx", "timestamp_sec", "is_playing"]].copy()
    pred_df["pred_playing"] = pred.astype(bool)
    return pred_df, {"precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E placeholder: clip MP4 + court_calibrations homography + timeline annotations → parquet + XGBoost."
    )
    p.add_argument(
        "--clip-id",
        type=int,
        default=None,
        help="Supabase clips.id (single-clip mode; mutually exclusive with --random)",
    )
    p.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help="Process N random clips that are annotated and whose source_id has court_calibrations",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would run; no download or parquet writes")
    p.add_argument("--list-only", action="store_true", help="Deprecated: same as --dry-run")
    p.add_argument("--fps", type=float, default=2.0, help="Sampling rate for feature extraction (Hz)")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket used with --random")
    p.add_argument("--prefix", default=DEFAULT_S3_PREFIX, help="S3 key prefix used with --random")
    p.add_argument(
        "--label-fps",
        type=float,
        default=DEFAULT_LABEL_FPS,
        help="Label Studio timeline frame rate for Playing ranges",
    )
    p.add_argument("--skip-download", action="store_true", help="Use existing file under media/clips/…")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --random")
    p.add_argument(
        "--feature-mode",
        choices=["all", "new_only"],
        default="all",
        help="Write all features or only the newly added features to the parquet outputs.",
    )
    p.add_argument(
        "--pose-conf",
        type=float,
        default=POSE_CONF_DEFAULT,
        help="Minimum mean keypoint confidence per person used for feature extraction.",
    )
    return p.parse_args()


def run_clip(
    *,
    source_id: str,
    clip_index: int,
    s3_uri: str,
    args: argparse.Namespace,
    db_helpers: Any,
    db_client: Any,
) -> dict[str, Any]:
    clip_stem = f"{source_id}_{clip_index:03d}"
    local_clip = _local_clip_path(source_id, clip_index)
    features_path = args.cache_dir / f"{clip_stem}_features.parquet"
    predictions_path = args.cache_dir / f"{clip_stem}_predictions.parquet"

    if not args.skip_download:
        ensure_local_clip(s3_uri=s3_uri, local_path=local_clip, region=args.region)
    elif not local_clip.is_file():
        raise RuntimeError(f"--skip-download set but clip is missing: {local_clip}")

    cal = db_helpers.get_court_calibration(db_client, source_id)
    if cal is None:
        raise RuntimeError(f"no court_calibrations row for source_id={source_id!r}")
    H, wx_min, wx_max, wy_min, wy_max, _, _ = homography_arrays_from_court_calibration_row(cal)

    df_feat = extract_features_for_clip(
        video_path=local_clip,
        H=H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        target_fps=args.fps,
        weights=WEIGHTS_DEFAULT,
        imgsz=IMGSZ_DEFAULT,
        det_conf=DET_CONF_DEFAULT,
        ankle_conf=ANKLE_CONF_DEFAULT,
        kp_conf_thresh=KP_CONF_DEFAULT,
        pose_conf_thresh=args.pose_conf,
    )
    df_feat.insert(0, "source_id", source_id)
    df_feat.insert(1, "clip_index", int(clip_index))
    df_feat.insert(2, "clip_s3_uri", s3_uri)
    df_feat.insert(3, "clip_local_path", str(local_clip))

    feature_columns = FEATURE_COLUMNS
    if args.feature_mode == "new_only":
        keep_cols = [
            "source_id",
            "clip_index",
            "clip_s3_uri",
            "clip_local_path",
            "frame_idx",
            "timestamp_sec",
            *NEW_FEATURE_COLUMNS,
        ]
        df_feat = df_feat[keep_cols]
        feature_columns = list(NEW_FEATURE_COLUMNS)

    df_feat.to_parquet(features_path, index=False)
    print(f"wrote features: {features_path}")

    payload = fetch_latest_annotation_payload(db_helpers, db_client, source_id, clip_index)
    df_labeled = add_ground_truth_labels(df_feat, payload, label_fps=args.label_fps)

    pred_df, metrics = train_and_predict(df_labeled, feature_columns)
    pred_df.insert(0, "source_id", source_id)
    pred_df.insert(1, "clip_index", int(clip_index))
    pred_df.insert(2, "clip_s3_uri", s3_uri)
    pred_df.insert(3, "clip_local_path", str(local_clip))

    pred_df.to_parquet(predictions_path, index=False)
    print(f"wrote predictions: {predictions_path}")
    print(
        "metrics "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )

    return {
        "source_id": source_id,
        "clip_index": clip_index,
        "n_sampled_frames": int(len(df_feat)),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "features_path": str(features_path),
        "predictions_path": str(predictions_path),
    }


def main() -> int:
    from dotenv import load_dotenv

    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    if args.clip_id is not None and args.random > 0:
        print("use either --clip-id or --random, not both", file=sys.stderr)
        return 1
    if args.clip_id is None and args.random <= 0:
        print("pass --clip-id <clips.id> or --random N", file=sys.stderr)
        return 1
    if args.random < 0:
        print("--random must be >= 0", file=sys.stderr)
        return 1
    if args.fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    db_helpers = _get_db_helpers()
    db_client = db_helpers.get_supabase_client()
    annotated_clip_keys = fetch_annotated_clip_keys(db_client)
    print(f"annotated clips available in Supabase: {len(annotated_clip_keys)}")
    homography_source_ids = db_helpers.list_court_calibration_source_ids(db_client)
    print(f"source_ids with court_calibrations: {', '.join(sorted(homography_source_ids))}")
    eligible_clip_keys = {k for k in annotated_clip_keys if k[0] in homography_source_ids}
    print(f"eligible annotated clips after court_calibrations filter: {len(eligible_clip_keys)}")
    if not eligible_clip_keys:
        print("no eligible clips (need timeline annotation + court_calibrations for source_id)", file=sys.stderr)
        return 1

    if args.random > 0:
        targets = pick_random_s3_clips(
            bucket=args.bucket,
            region=args.region,
            prefix=args.prefix,
            n=args.random,
            seed=args.seed,
            allowed_clip_keys=eligible_clip_keys,
        )
    else:
        clip_row = db_helpers.get_clip_by_id(db_client, args.clip_id)
        if clip_row is None:
            print(f"no clips row for id={args.clip_id}", file=sys.stderr)
            return 1
        source_id = str(clip_row["source_id"])
        clip_index = int(clip_row["clip_index"])
        if source_id not in homography_source_ids:
            print(f"no court_calibrations for source_id={source_id!r}", file=sys.stderr)
            return 1
        if (source_id, clip_index) not in eligible_clip_keys:
            print(
                f"clip not eligible (needs timeline annotation for this clip): {source_id}_{clip_index:03d}",
                file=sys.stderr,
            )
            return 1
        s3_uri = f"s3://{clip_row['s3_bucket']}/{clip_row['s3_key']}"
        targets = [(source_id, clip_index, s3_uri)]

    print(f"selected {len(targets)} clip(s)")
    for source_id, clip_index, s3_uri in targets:
        print(f"  - {source_id}_{clip_index:03d}  ({s3_uri})")
    if args.dry_run or args.list_only:
        return 0

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for idx, (source_id, clip_index, s3_uri) in enumerate(targets, start=1):
        print(f"\n[{idx}/{len(targets)}] processing {source_id}_{clip_index:03d}")
        try:
            out = run_clip(
                source_id=source_id,
                clip_index=clip_index,
                s3_uri=s3_uri,
                args=args,
                db_helpers=db_helpers,
                db_client=db_client,
            )
            results.append(out)
        except Exception as exc:  # noqa: BLE001
            err = {"source_id": source_id, "clip_index": clip_index, "error": str(exc)}
            failures.append(err)
            print(f"ERROR: {source_id}_{clip_index:03d}: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break

    if results:
        df_summary = pd.DataFrame(results)
        summary_path = args.cache_dir / "last_run_clip_metrics.parquet"
        df_summary.to_parquet(summary_path, index=False)
        print(f"\nwrote run summary: {summary_path}")
        print(
            "mean metrics "
            f"precision={df_summary['precision'].mean():.4f} "
            f"recall={df_summary['recall'].mean():.4f} "
            f"f1={df_summary['f1'].mean():.4f}"
        )

    if failures:
        print(f"\ncompleted with failures: {len(failures)} failed / {len(targets)} total", file=sys.stderr)
        for f in failures:
            print(f"  - {f['source_id']}_{int(f['clip_index']):03d}: {f['error']}", file=sys.stderr)
        return 1

    print(f"\ncompleted successfully: {len(results)} / {len(targets)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
