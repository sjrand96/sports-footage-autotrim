"""Parse Label Studio exports for court KeyPointLabels (homography calibration).

Export format: JSON array of tasks. Each task has ``data.image`` (S3 or URL) and
``annotations[].result`` entries with ``type == "keypointlabels"``.

Coordinates in the export are **percentages** (0–100) of ``original_width`` /
``original_height``; we also emit pixel coordinates for OpenCV homography.

Future DB (see docs/schema.md): expect something like a new table or an extended
``annotations``-style row with ``clip_id`` (or ``source_id`` + ``clip_index``),
``annotator``, and ``payload`` jsonb — use :func:`calibration_record_to_json`
for a stable object to store or embed inside ``payload``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

_THUMB_KEY_RE = re.compile(
    r"clips/(?P<source_id>[^/]+)/(?P=source_id)_(?P<clip_index>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Keypoint:
    """One labeled court point."""

    label: str
    x_pct: float
    y_pct: float
    x_px: float
    y_px: float


@dataclass
class CalibrationRecord:
    """Normalized view of one submitted keypoint annotation on a still frame."""

    source_id: str
    clip_index: int
    image_s3_bucket: str
    image_s3_key: str
    image_width_px: int
    image_height_px: int
    label_studio_task_id: int
    label_studio_annotation_id: int
    label_studio_project_id: int | None
    lead_time_sec: float | None
    annotation_created_at: str | None
    annotation_updated_at: str | None
    keypoints: list[Keypoint] = field(default_factory=list)
    raw_image_ref: str = ""
    """Original ``data.image`` string from the task (S3 URI or URL)."""


def pick_latest_annotation(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Same policy as ``data_labeling/push_annotations.py`` (non-cancelled, latest by time)."""
    candidates: list[dict[str, Any]] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        if a.get("was_cancelled"):
            continue
        candidates.append(a)
    if not candidates:
        return None
    return max(candidates, key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""))


def parse_image_ref(image_ref: str) -> tuple[str, str, str | None, int | None]:
    """Return (bucket, key, source_id, clip_index) from S3 URI or https URL path.

    ``source_id`` / ``clip_index`` are set when the key matches prep's thumbnail layout.
    """
    ref = unquote((image_ref or "").strip())
    path: str
    bucket: str
    if ref.lower().startswith("s3://"):
        rest = ref[5:]
        slash = rest.find("/")
        if slash == -1:
            return "", "", None, None
        bucket, path = rest[:slash], rest[slash + 1 :]
    else:
        parsed = urlparse(ref)
        host = (parsed.netloc or "").lower()
        path = parsed.path.lstrip("/")
        if ".s3." in host or host.startswith("s3."):
            bucket = host.split(".")[0] if ".s3." in host else ""
        else:
            bucket = ""
        if not bucket and path:
            # path might be "bucket/key"
            parts = path.split("/", 1)
            if len(parts) == 2:
                bucket, path = parts[0], parts[1]

    m = _THUMB_KEY_RE.search(path)
    if not m:
        return bucket, path, None, None
    return bucket, path, m.group("source_id"), int(m.group("clip_index"))


def keypoint_results_to_list(
    result: list[dict[str, Any]],
) -> tuple[list[Keypoint], int, int]:
    """Extract Keypoint rows and image dimensions from a Label Studio ``result`` list."""
    width_px = 0
    height_px = 0
    out: list[Keypoint] = []
    for item in result:
        if not isinstance(item, dict) or item.get("type") != "keypointlabels":
            continue
        w = int(item.get("original_width") or 0)
        h = int(item.get("original_height") or 0)
        if w > 0:
            width_px = w
        if h > 0:
            height_px = h
        val = item.get("value") or {}
        labels = val.get("keypointlabels") or []
        if not labels or not isinstance(labels, list):
            continue
        label = str(labels[0])
        x_pct = float(val.get("x", 0.0))
        y_pct = float(val.get("y", 0.0))
        if width_px <= 0 or height_px <= 0:
            raise ValueError(
                "keypoint entry missing original_width/original_height; cannot convert to pixels"
            )
        x_px = (x_pct / 100.0) * width_px
        y_px = (y_pct / 100.0) * height_px
        out.append(
            Keypoint(label=label, x_pct=x_pct, y_pct=y_pct, x_px=x_px, y_px=y_px)
        )
    return out, width_px, height_px


def task_to_calibration_record(task: dict[str, Any]) -> CalibrationRecord | None:
    """Build a :class:`CalibrationRecord` from one Label Studio task dict."""
    data = task.get("data") or {}
    image_ref = data.get("image")
    if not isinstance(image_ref, str) or not image_ref:
        return None

    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return None
    ann = pick_latest_annotation(annotations)
    if ann is None:
        return None

    result = ann.get("result")
    if not isinstance(result, list):
        return None

    keypoints, wpx, hpx = keypoint_results_to_list(result)
    if not keypoints:
        return None

    bucket, key, sid, cidx = parse_image_ref(image_ref)
    source_id = sid if sid is not None else ""
    clip_index = cidx if cidx is not None else -1

    return CalibrationRecord(
        source_id=source_id,
        clip_index=clip_index,
        image_s3_bucket=bucket,
        image_s3_key=key,
        image_width_px=wpx,
        image_height_px=hpx,
        label_studio_task_id=int(task.get("id") or 0),
        label_studio_annotation_id=int(ann.get("id") or 0),
        label_studio_project_id=int(ann["project"]) if ann.get("project") is not None else None,
        lead_time_sec=float(ann["lead_time"]) if ann.get("lead_time") is not None else None,
        annotation_created_at=str(ann["created_at"]) if ann.get("created_at") else None,
        annotation_updated_at=str(ann["updated_at"]) if ann.get("updated_at") else None,
        keypoints=keypoints,
        raw_image_ref=image_ref,
    )


def parse_keypoint_export_file(path: Path) -> list[CalibrationRecord]:
    """Load a Label Studio JSON export (array of tasks) and parse all submitted keypoint tasks."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("expected JSON array of tasks")
    out: list[CalibrationRecord] = []
    for task in raw:
        if not isinstance(task, dict):
            continue
        rec = task_to_calibration_record(task)
        if rec is not None:
            out.append(rec)
    return out


def calibration_record_to_json(rec: CalibrationRecord) -> dict[str, Any]:
    """Stable dict for DB ``jsonb`` (e.g. nested under ``payload``) or downstream tools."""
    return {
        "schema_version": 1,
        "kind": "court_keypoints_label_studio",
        "source_id": rec.source_id,
        "clip_index": rec.clip_index,
        "frame": {
            "s3_bucket": rec.image_s3_bucket,
            "s3_key": rec.image_s3_key,
            "width_px": rec.image_width_px,
            "height_px": rec.image_height_px,
            "label_studio_image_ref": rec.raw_image_ref,
        },
        "label_studio": {
            "task_id": rec.label_studio_task_id,
            "annotation_id": rec.label_studio_annotation_id,
            "project_id": rec.label_studio_project_id,
            "lead_time_sec": rec.lead_time_sec,
            "created_at": rec.annotation_created_at,
            "updated_at": rec.annotation_updated_at,
        },
        "keypoints": [
            {
                "label": kp.label,
                "x_pct": kp.x_pct,
                "y_pct": kp.y_pct,
                "x_px": round(kp.x_px, 4),
                "y_px": round(kp.y_px, 4),
            }
            for kp in sorted(rec.keypoints, key=lambda k: k.label)
        ],
    }
