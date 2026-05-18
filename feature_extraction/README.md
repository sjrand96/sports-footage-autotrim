# Feature extraction

Per-frame pose + homography features → parquets under `{out_dir}/{run_id}/train|test/`.

See [PLAN.md](PLAN.md) for architecture. AWS deploy: **[CLOUD_DEPLOY.md](CLOUD_DEPLOY.md)**. Train/test assignment: **`clip_split.py`**.

## Local run

```bash
# From repo root (uses .venv and .env)
.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --clip-id 64 \
  --skip-download

# List eligible clips + split (no YOLO)
.venv/bin/python feature_extraction/job.py --out-dir /tmp/fe --dry-run --max-clips 5

# Smoke test (first N frames only)
.venv/bin/python feature_extraction/job.py --out-dir feature_extraction/_runs \
  --clip-id 64 --skip-download --max-frames 30 --run-id smoke_test
```

Exit code `1` if any clip failed; inspect `{run_id}/manifest.json`, `run_report.json`, and `timings.json`.

## Timings

Each run writes **`timings.json`** (local + S3) with per-clip stage seconds (`download`, `calibration`, `extract`, `labels`, `parquet_write`, `clip_total`) and derived **`sec_per_frame`** for benchmarking. `manifest.json` includes a short **`timing_summary`** rollup. Logs one line per clip: `timing … extract=…s (… s/frame, … rows)`.

## S3 upload (phase 2)

Uploads to `s3://{bucket}/feature_extraction/{run_id}/` with `train/`, `test/`, `manifest.json`, `run_report.json`, `timings.json`.
Legacy flat files under `feature_extractions/` are unchanged.

```bash
# Extract locally then upload
.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --clip-id 64 --skip-download --max-frames 30 \
  --upload-s3 --run-id my_run_v1

# Upload an existing local run
.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --upload-only --run-id smoke_test
```

Bucket: `--bucket` or `$S3_BUCKET` (default `sports-footage-autotrim-bucket`).

### Full-fps timing mini run (one clip)

```bash
export RUN_ID=mini_fullfps_1clip

.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --clip-id 64 \
  --skip-download \
  --upload-s3 \
  --run-id "$RUN_ID"

# validate
cat "feature_extraction/_runs/$RUN_ID/timings.json" | head -50
aws s3 ls "s3://sports-footage-autotrim-bucket/feature_extraction/$RUN_ID/" --recursive
```

Expect ~30–60+ minutes on CPU for one 60s clip (~1800 frames). Check `derived.sec_per_frame` under `clips[0].timings_sec.extract`.

### Frame JPEG export cost (one clip)

Writes and uploads `s3://…/clips_v2/{source_id}/{clip_index}/frames/*.jpg` during extract (same decode pass as features).

```bash
export RUN_ID=mini_frames_1clip

.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --clip-id 64 \
  --skip-download \
  --write-frames \
  --upload-s3 \
  --run-id "$RUN_ID"

aws s3 ls "s3://sports-footage-autotrim-bucket/clips_v2/1rXZJyVXUHU/1/frames/" | wc -l
grep frames_upload "feature_extraction/_runs/$RUN_ID/timings.json"
```

## Tabular XGBoost (phase 3)

Train on the extract-time split (`train/` + `test/` parquets; `is_playing` is already in each row):

```bash
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id mini_20260517T183530Z \
  --save-report-json feature_extraction/_runs/mini_20260517T183530Z/xgb_report.json

# See models/tabular_xgb/README.md
```

Requires at least one parquet in **both** `train/` and `test/` (use a multi-clip run or adjust `clip_split.py`).

## Container (Step 1 — local Docker)

From repo root (`yolov8s-pose.pt` must exist; Docker Desktop running):

```bash
docker build -f feature_extraction/Dockerfile -t fe-worker .
docker run --rm --env-file .env fe-worker \
  feature_extraction/job.py \
  --out-dir /tmp/fe --clip-id 64 --max-frames 60 --upload-s3 --run-id docker_smoke_60f
```

See [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) for full benchmark steps.
