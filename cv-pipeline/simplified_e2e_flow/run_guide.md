# Simplified E2E Pipeline Run Guide

This guide covers how to run:

- `cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py`

It includes common command scenarios, a concise flag reference, and a high-level path to pooled XGBoost training.

## What the script does

For each selected clip, the pipeline:

1. Ensures the clip is local (downloads from S3 unless `--skip-download` is set)
2. Samples frames at `--target-fps`
3. Runs pose + homography feature extraction
4. Pulls latest annotation for that clip from Supabase
5. Creates per-frame labels (`is_playing`)
6. Trains placeholder XGBoost and writes predictions

Outputs are written to `cv-pipeline/simplified_e2e_flow/cache/` by default.

## Prerequisites

- Run from repo root: `sports-footage-autotrim/`
- Python env with required deps (`opencv`, `ultralytics`, `torch`, `pandas`, `xgboost`, `supabase`, `boto3`, `python-dotenv`)
- `.env` at repo root with:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION` (optional; defaults to `us-west-2`)
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_KEY`

## Quick start

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py
```

This runs one clip using defaults (`jZ18INu4LQc_006`).

## Common run scenarios

### 1) Single known clip (explicit)

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --source-id jZ18INu4LQc \
  --clip-index 6
```

### 2) Single clip, faster smoke run

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --source-id jZ18INu4LQc \
  --clip-index 6 \
  --target-fps 0.2
```

### 3) Skip S3 download if file is already local

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --source-id jZ18INu4LQc \
  --clip-index 6 \
  --skip-download
```

### 4) Preview random S3 clip selection only (no processing)

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 5 \
  --random-seed 42 \
  --list-only
```

### 5) Process random clips from S3

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 5 \
  --random-seed 42 \
  --target-fps 0.5
```

### 6) Stop immediately on first clip error

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 10 \
  --stop-on-error
```

### 7) Use custom cache output directory

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 5 \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache/experiment_a
```

### 8) Override bucket/prefix for sampling

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 5 \
  --s3-bucket sports-footage-autotrim-bucket \
  --s3-prefix clips/
```

## Flag reference

### Clip selection

- `--source-id` (default `jZ18INu4LQc`)
- `--clip-index` (default `6`)
- `--s3-uri` (override exact clip URI)
- `--num-random-clips` (default `0`; when `>0`, random mode is used)
- `--random-seed` (default `42`)
- `--list-only` (show selected clips and exit)

### S3

- `--s3-bucket` (default `sports-footage-autotrim-bucket`)
- `--s3-prefix` (default `clips/`)
- `--region` (default from `AWS_REGION` env, fallback `us-west-2`)
- `--skip-download` (requires local clip file already present)

### Feature extraction

- `--target-fps` (default `2.0`)
- `--weights` (default `yolov8s-pose.pt`)
- `--imgsz` (default `1280`)
- `--det-conf` (default `0.15`)
- `--ankle-conf` (default `0.25`)
- `--kp-conf` (default `0.25`)
- `--npz` (default `cv-pipeline/calibration/out/homography.npz`)

### Labels / outputs / control

- `--label-fps` (default `30.0`, Label Studio frame rate)
- `--cache-dir` (default `cv-pipeline/simplified_e2e_flow/cache`)
- `--stop-on-error` (fail fast in multi-clip runs)

## Output files

Per clip:

- `<source_id>_<clip_index>_features.parquet`
- `<source_id>_<clip_index>_predictions.parquet`

Run-level summary:

- `last_run_clip_metrics.parquet`

Both per-clip parquets include metadata columns so you can join without relying on filenames:

- `source_id`
- `clip_index`
- `clip_s3_uri`
- `clip_local_path`

## High-level path to pooled XGBoost (not implemented here)

Current behavior trains per clip. To move to pooled training across many clips:

1. Run the pipeline over a larger random clip set to generate many per-clip feature + label tables.
2. Build one combined frame table by stacking all clips.
3. Split train/test by clip (group split), not by random frames, to avoid temporal leakage.
4. Train one global XGBoost model on pooled train frames.
5. Predict on held-out clips and write per-frame predictions.
6. Report both:
   - per-clip metrics (for variability across matches/camera conditions)
   - pooled aggregate metrics (overall precision/recall/F1)
7. Persist the trained model artifact and the exact clip split used for reproducibility.

