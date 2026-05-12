# Simplified end-to-end pipeline — run guide

Runs **`cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py`**.

## What it does (per clip)

1. Downloads the MP4 from S3 (unless **`--skip-download`** and the file already exists under `cv-pipeline/pose-detection/media/clips/…`).
2. Samples at **`--fps`**, runs YOLO pose, projects feet to court space using **`court_calibrations`** for that clip’s `source_id` (no local npz/JSON).
3. Loads the latest timeline **`annotations`** row for the clip from Supabase and builds **`is_playing`** from “Playing” ranges.
4. Trains a small XGBoost on the same clip (placeholder) and writes parquet under **`--cache-dir`**.

## Prerequisites

- Repo root, venv with `opencv`, `ultralytics`, `torch`, `pandas`, `xgboost`, `supabase`, `boto3`, `python-dotenv`.
- **`.env`**: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_REGION`.

## CLI (simplified)

| Flag | Role |
|------|------|
| **`--clip-id`** | Single clip: Supabase **`clips.id`**. |
| **`--random N`** | Random **N** clips from S3 that are annotated **and** whose `source_id` has **`court_calibrations`**. |
| **`--dry-run`** | Print selection only (also **`--list-only`** as alias). |
| **`--fps`** | Feature sampling rate (default `2`). |
| **`--cache-dir`** | Output directory (default `cv-pipeline/simplified_e2e_flow/cache`). |
| **`--label-fps`** | Timeline label frame rate (default `30`). |
| **`--bucket`**, **`--prefix`**, **`--region`** | S3 listing for **`--random`** (defaults: project bucket, `clips/`, `us-west-2`). |
| **`--seed`** | RNG seed for **`--random`** (default `42`). |
| **`--skip-download`**, **`--stop-on-error`** | Same as before. |

Pose weights / `imgsz` / detector confs are fixed in the script (aligned with the pose demo defaults).

## Examples

```bash
# Inspect eligibility only
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --clip-id 123 --dry-run

# One clip by DB id
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --clip-id 123 --fps 2

# Random batch
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --random 5 --seed 42
```

## Outputs

Per clip:

- `<source_id>_<clip_index>_features.parquet`
- `<source_id>_<clip_index>_predictions.parquet`

Run summary: **`last_run_clip_metrics.parquet`** in **`--cache-dir`**.

Parquet rows include **`source_id`**, **`clip_index`**, **`clip_s3_uri`**, **`clip_local_path`** on the feature table (and aligned metadata on predictions) for joins without relying on filenames alone.

## Pooled XGBoost from cache

After generating cache files:

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache \
  --save-report-json cv-pipeline/simplified_e2e_flow/cache/pooled_eval_report.json \
  --save-model cv-pipeline/simplified_e2e_flow/cache/pooled_xgb_model.json
```

That script joins paired `*_features.parquet` / `*_predictions.parquet`, uses a clip-level train/test split, and writes metrics plus optional model artifact.
