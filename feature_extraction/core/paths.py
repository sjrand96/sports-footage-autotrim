"""Repo paths and dynamic imports from pose-detection / calibration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_DIR = REPO_ROOT / "cv-pipeline" / "pose-detection"
CALIB_DIR = REPO_ROOT / "cv-pipeline" / "calibration"
FETCH_SCRIPT = POSE_DIR / "fetch_s3_clip.py"
POSE_SCRIPT = POSE_DIR / "pose_side_by_side_video.py"
LOCAL_CLIP_ROOT = POSE_DIR / "media" / "clips"

DEFAULT_BUCKET = "sports-footage-autotrim-bucket"
DEFAULT_REGION = "us-west-2"

WEIGHTS_DEFAULT = "yolov8s-pose.pt"
IMGSZ_DEFAULT = 1280
DET_CONF_DEFAULT = 0.15
ANKLE_CONF_DEFAULT = 0.25
KP_CONF_DEFAULT = 0.25
DEFAULT_LABEL_FPS = 30.0


def local_clip_path(source_id: str, clip_index: int) -> Path:
    return LOCAL_CLIP_ROOT / source_id / f"{source_id}_{clip_index:03d}.mp4"


def clip_stem(source_id: str, clip_index: int) -> str:
    return f"{source_id}_{clip_index:03d}"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_import_paths() -> None:
    for p in (REPO_ROOT, POSE_DIR, CALIB_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


_fetch_mod: Any | None = None
_pose_mod: Any | None = None


def fetch_module() -> Any:
    global _fetch_mod
    ensure_import_paths()
    if _fetch_mod is None:
        _fetch_mod = _load_module("fe_fetch_s3_clip", FETCH_SCRIPT)
    return _fetch_mod


def pose_module() -> Any:
    global _pose_mod
    ensure_import_paths()
    if _pose_mod is None:
        _pose_mod = _load_module("fe_pose_side_by_side", POSE_SCRIPT)
    return _pose_mod


def homography_from_calibration_row(cal_row: dict[str, Any]) -> tuple[Any, ...]:
    ensure_import_paths()
    from homography_io import homography_arrays_from_court_calibration_row

    return homography_arrays_from_court_calibration_row(cal_row)
