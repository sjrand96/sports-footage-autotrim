#!/usr/bin/env python3
"""Run the simplified single-clip E2E pipeline.

This script follows `cv-pipeline/simplified_e2e_flow/simple_e2e_plan.md`:
1) Ensure one clip is available locally (download from S3 if missing)
2) Extract per-frame features from YOLO pose + homography projection
3) Fetch latest ground-truth annotation from Supabase
4) Build per-frame labels, train placeholder XGBoost classifier
5) Write features/predictions parquet outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = REPO_ROOT / "cv-pipeline" / "pose-detection"
FETCH_SCRIPT = POSE_DIR / "fetch_s3_clip.py"
POSE_SCRIPT = POSE_DIR / "pose_side_by_side_video.py"
DEFAULT_NPZ = REPO_ROOT / "cv-pipeline" / "calibration" / "out" / "homography.npz"
DEFAULT_CACHE_DIR = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "cache"
DEFAULT_BUCKET = "sports-footage-autotrim-bucket"
DEFAULT_REGION = "us-west-2"
DEFAULT_LABEL_FPS = 30.0
FEATURE_COLUMNS = [
    "n_players_total",
    "n_front_row",
    "n_back_row",
    "n_camera_side",
    "n_opposite_side",
    "median_nearest_neighbor_dist",
    "hands_above_head_count",
]

_NOSE = 0
_L_WRIST = 9
_R_WRIST = 10


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FETCH_MOD = _load_module("fetch_s3_clip_module", FETCH_SCRIPT)
POSE_MOD = _load_module("pose_side_by_side_module", POSE_SCRIPT)


def _s3_uri_for_clip(bucket: str, source_id: str, clip_index: int) -> str:
    return f"s3://{bucket}/clips/{source_id}/{source_id}_{clip_index:03d}.mp4"


def _local_clip_path(source_id: str, clip_index: int) -> Path:
    filename = f"{source_id}_{clip_index:03d}.mp4"
    return REPO_ROOT / "cv-pipeline" / "pose-detection" / "media" / "clips" / source_id / filename


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


def extract_features_for_clip(
    *,
    video_path: Path,
    npz_path: Path,
    target_fps: float,
    weights: str,
    imgsz: int,
    det_conf: float,
    ankle_conf: float,
    kp_conf_thresh: float,
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
    if not npz_path.is_file():
        raise FileNotFoundError(f"homography npz not found: {npz_path}")

    H, wx_min, wx_max, wy_min, wy_max, _, _ = POSE_MOD._load_homography_npz(npz_path)
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

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            result = model(frame_bgr, imgsz=imgsz, conf=det_conf, verbose=False)[0]
            kp = result.keypoints
            world_points: list[tuple[float, float]] = []
            hands_above_count = 0

            if kp is not None and kp.xy is not None and kp.xy.shape[0] > 0:
                xy = kp.xy.cpu().numpy()
                if kp.conf is not None:
                    kconf = kp.conf.cpu().numpy()
                else:
                    kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

                for i in range(xy.shape[0]):
                    foot_uv = POSE_MOD._foot_uv_from_coco17(xy[i], kconf[i], ankle_conf=ankle_conf)
                    if foot_uv is None:
                        continue

                    wx, wy = POSE_MOD._image_uv_to_world_m(H, float(foot_uv[0]), float(foot_uv[1]))
                    if not (wx_min <= wx <= wx_max and wy_min <= wy <= wy_max):
                        continue

                    world_points.append((wx, wy))
                    if _hands_above_head_for_player(xy[i], kconf[i], kp_conf_thresh=kp_conf_thresh):
                        hands_above_count += 1

            world = np.asarray(world_points, dtype=np.float64)
            n_total = int(world.shape[0])
            if n_total > 0:
                wy = world[:, 1]
                n_camera_side = int(np.sum(wy < 0.0))
                n_opposite_side = int(np.sum(wy >= 0.0))
                n_front_row = int(np.sum(np.abs(wy) < 3.0))
                n_back_row = int(np.sum(np.abs(wy) >= 3.0))
                median_nn = _median_nearest_neighbor_distance(world)
            else:
                n_camera_side = 0
                n_opposite_side = 0
                n_front_row = 0
                n_back_row = 0
                median_nn = float("nan")

            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": float(frame_idx / src_fps),
                    "n_players_total": n_total,
                    "n_front_row": n_front_row,
                    "n_back_row": n_back_row,
                    "n_camera_side": n_camera_side,
                    "n_opposite_side": n_opposite_side,
                    "median_nearest_neighbor_dist": median_nn,
                    "hands_above_head_count": int(hands_above_count),
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


def fetch_latest_annotation_payload(source_id: str, clip_index: int) -> dict[str, Any]:
    from dotenv import load_dotenv

    sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers

    load_dotenv(REPO_ROOT / ".env")
    client = db_helpers.get_supabase_client()
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


def train_and_predict(df_labeled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is required. Install with `pip install xgboost`.") from exc

    from sklearn.metrics import f1_score, precision_score, recall_score

    model_input = df_labeled[FEATURE_COLUMNS].copy()
    model_input["median_nearest_neighbor_dist"] = model_input["median_nearest_neighbor_dist"].fillna(-1.0)
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
    p = argparse.ArgumentParser(description="Run simplified single-clip E2E pipeline.")
    p.add_argument("--source-id", default="jZ18INu4LQc")
    p.add_argument("--clip-index", type=int, default=6)
    p.add_argument(
        "--s3-uri",
        default=None,
        help="Override clip URI. Default is built from --source-id/--clip-index.",
    )
    p.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    p.add_argument("--target-fps", type=float, default=2.0)
    p.add_argument("--weights", default="yolov8s-pose.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--det-conf", type=float, default=0.15)
    p.add_argument("--ankle-conf", type=float, default=0.25)
    p.add_argument("--kp-conf", type=float, default=0.25)
    p.add_argument("--label-fps", type=float, default=DEFAULT_LABEL_FPS)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.clip_index <= 0:
        print("--clip-index must be >= 1", file=sys.stderr)
        return 1

    source_id = args.source_id.strip()
    clip_index = int(args.clip_index)
    clip_stem = f"{source_id}_{clip_index:03d}"
    s3_uri = args.s3_uri or _s3_uri_for_clip(DEFAULT_BUCKET, source_id, clip_index)
    local_clip = _local_clip_path(source_id, clip_index)

    features_path = args.cache_dir / f"{clip_stem}_features.parquet"
    predictions_path = args.cache_dir / f"{clip_stem}_predictions.parquet"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        ensure_local_clip(s3_uri=s3_uri, local_path=local_clip, region=args.region)
    elif not local_clip.is_file():
        print(f"--skip-download set but clip is missing: {local_clip}", file=sys.stderr)
        return 1

    df_feat = extract_features_for_clip(
        video_path=local_clip,
        npz_path=args.npz,
        target_fps=args.target_fps,
        weights=args.weights,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        ankle_conf=args.ankle_conf,
        kp_conf_thresh=args.kp_conf,
    )
    df_feat.to_parquet(features_path, index=False)
    print(f"wrote features: {features_path}")

    payload = fetch_latest_annotation_payload(source_id, clip_index)
    df_labeled = add_ground_truth_labels(df_feat, payload, label_fps=args.label_fps)

    pred_df, metrics = train_and_predict(df_labeled)
    pred_df.to_parquet(predictions_path, index=False)
    print(f"wrote predictions: {predictions_path}")
    print(
        "metrics "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
