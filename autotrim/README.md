# Autotrim (local inference)

Trim a local sports clip to **predicted playing time** only:

1. Extract pose/homography features at a chosen FPS
2. Run the pretrained **5-fold XGB ensemble** (mean probability)
3. Apply the segment smoother from `eval/segment_metrics.py`
4. **ffmpeg** trim + concat from the original file

No training, Supabase, or S3 in the hot path.

## Requirements

- Python deps from repo root (`.venv`, `opencv-python`, `ultralytics`, `xgboost`, etc.)
- `ffmpeg` / `ffprobe` on `PATH`
- `yolov8s-pose.pt` at repo root (or pass `--weights`)
- On macOS, use default `--yolo-device cpu` (avoids MPS issues); XGBoost is loaded only after extract
- Pretrained CV weights: `models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold/fold-{1..5}/xgb_model.json`

## Example

```bash
.venv/bin/python autotrim/trim_clip.py \
  --input path/to/clip.mp4 \
  --calibration-json path/to/court_cal.json \
  --extract-fps 15 \
  --output path/to/clip_playing.mp4 \
  --keep-sidecar
```

Defaults: `--cv-run models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold`, segment threshold ≈ **0.27** (mean of fold thresholds in `cv_summary.json`).

## Calibration JSON

Same shape as a Supabase `court_calibrations` row (see `cv-pipeline/calibration/homography_io.py`):

```json
{
  "homography_matrix": [[...], [...], [...]],
  "world_wx_min": -9.0,
  "world_wx_max": 9.0,
  "world_wy_min": -4.5,
  "world_wy_max": 4.5,
  "pixels_per_metre": 45.0
}
```

Alternatively use `--calibration-npz` from `cv-pipeline/calibration/court_homography.py` (`homography.npz`).

Export a row from Supabase (one-time setup):

```python
import json
from pathlib import Path
from src.db import get_client, get_court_calibration

client = get_client()
row = get_court_calibration(client, "YOUR_SOURCE_ID")
Path("court_cal.json").write_text(json.dumps(row, indent=2))
```

## Segment smoother flags

Same defaults as `eval/plot_preds_timeseries.py`:

| Flag | Default |
|------|---------|
| `--segment-window-frames` | 15 |
| `--segment-aggregation` | mean |
| `--segment-max-gap-frames` | 45 |
| `--segment-min-segment-frames` | 90 |

## Smoke test

**Fast path** (reuse existing features parquet; still cuts the real MP4):

```bash
.venv/bin/python autotrim/trim_clip.py \
  --input cv-pipeline/pose-detection/media/clips/1rXZJyVXUHU/1rXZJyVXUHU_001.mp4 \
  --features-parquet feature_extraction/_runs/raina_features_local_smoke/train/1rXZJyVXUHU_001.parquet \
  --segment-min-segment-frames 30 \
  --output autotrim/_runs/smoke_test/clip_playing.mp4 \
  --keep-sidecar
```

**Full path** (YOLO extract; use `--max-frames` for a short clip):

```bash
.venv/bin/python autotrim/trim_clip.py \
  --input clip.mp4 \
  --calibration-npz cv-pipeline/calibration/out/homography.npz \
  --extract-fps 15 \
  --max-frames 60 \
  --keep-sidecar
```

Cached debug artifacts go under `autotrim/_runs/` when `--keep-sidecar` is set.

## Timings

Every run prints stage timings to stdout:

```text
timings (sec): extract=45.123 infer=2.456 ffmpeg=3.789 total=51.368
```

With `--keep-sidecar`, the same breakdown is saved in `trim_sidecar.json` under `timings_sec`:

```json
"timings_sec": {
  "extract": 45.123,
  "infer": 2.456,
  "ffmpeg": 3.789,
  "total": 51.368
}
```

- **extract** — YOLO feature extraction (or parquet load)
- **infer** — load XGB folds, ensemble predict, segment smoothing
- **ffmpeg** — trim + concat export
- **total** — end-to-end wall clock
