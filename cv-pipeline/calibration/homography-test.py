#!/usr/bin/env python3
"""Smoke-test parsing of Label Studio court keypoints export.

From repo root: `.venv/bin/python cv-pipeline/calibration/homography-test.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from label_studio_keypoints import calibration_record_to_json, parse_keypoint_export_file

_REPO_CALIB = Path(__file__).resolve().parent
_DEFAULT_EXPORT = _REPO_CALIB / "project-7-at-2026-05-05-19-45-119e8837.json"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_EXPORT
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    records = parse_keypoint_export_file(path)
    if not records:
        print("no keypoint annotations parsed", file=sys.stderr)
        return 1
    payload = [calibration_record_to_json(r) for r in records]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
