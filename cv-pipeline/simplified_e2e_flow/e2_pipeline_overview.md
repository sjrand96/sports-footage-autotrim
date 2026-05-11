# Simplified End to End Vision Pipeline Run Guide

This guide is the team reference for running:

- `cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py`

It focuses on the two workflows we use most:

1. run a single known clip
2. run a random sample of clips from S3

## What this pipeline does

For each selected clip, the pipeline:

1. Ensures the clip is local (downloads from S3 unless `--skip-download` is set)
2. Samples frames at `--target-fps`
3. Runs pose + homography feature extraction
4. Pulls the latest annotation for that clip from Supabase
5. Creates per-frame labels (`is_playing`)
6. Trains a clip-local baseline XGBoost and writes predictions

By default, outputs are written to `cv-pipeline/simplified_e2e_flow/cache/`.

## What this pipeline does not do

- It does **not** provide production-grade model versioning or registry workflows.
- It does **not** replace the full CV pipeline; this is a fast end-to-end experiment path.
- It does **not** guarantee benchmark-level metrics unless clip selection, split strategy, and params are controlled.
- It currently depends on local homography artifacts from the data-labeling flow (`homography.npz` + court payload JSON), which is a key reason large-scale multi-clip runs are not fully scalable yet.
- The intent is to remove this local-artifact bottleneck soon by formalizing calibration data in shared storage/database-backed lookup.

## Prerequisites

- Run from repo root: `sports-footage-autotrim/`
- Python env with required deps (`opencv`, `ultralytics`, `torch`, `pandas`, `xgboost`, `supabase`, `boto3`, `python-dotenv`)
- `.env` at repo root with:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION` (optional; defaults to `us-west-2`)
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_KEY`
- Local calibration inputs are expected to exist:
  - `cv-pipeline/calibration/out/homography.npz`
  - `cv-pipeline/calibration/court_payloads.json` (from data labeller output)

## Quick start

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py
```

This runs one clip using defaults (`jZ18INu4LQc_006`).

## Primary run workflows

### 1) Single known clip

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --source-id jZ18INu4LQc \
  --clip-index 6
```

Use this for debugging, targeted checks, and quick iteration.

### 2) Random clips from S3

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py \
  --num-random-clips 5 \
  --random-seed 42 \
  --target-fps 0.5
```

Use this for broader sampling when you want multiple clips in one run.

## Useful optional flags (for both workflows)

- `--target-fps` for faster smoke runs (lower) vs denser frame coverage (higher)
- `--skip-download` when clips already exist locally
- `--list-only` to preview random S3 selection without processing
- `--cache-dir` to isolate outputs for an experiment
- `--stop-on-error` to fail fast in multi-clip runs

## Concise flag reference

### Clip selection

- `--source-id` (default `jZ18INu4LQc`)
- `--clip-index` (default `6`)
- `--s3-uri` (override exact clip URI)
- `--num-random-clips` (default `0`; when `>0`, random mode is used)
- `--random-seed` (default `42`)
- `--list-only` (show selected clips and exit)

### S3 / random sampling

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
- `--court-payloads-json` (default `cv-pipeline/calibration/court_payloads.json`; local data-labeller payload used to determine homography-supported sources)

### Labels / outputs / run control

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

## Pooled XGBoost from cached outputs

After generating cache files with `simple_e2e_pipeline.py`, train a pooled model across clips with:

```bash
.venv/bin/python cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache \
  --save-report-json cv-pipeline/simplified_e2e_flow/cache/pooled_eval_report.json \
  --save-model cv-pipeline/simplified_e2e_flow/cache/pooled_xgb_model.json
```

Current implementation details:

- Reads paired `*_features.parquet` and `*_predictions.parquet` files
- Joins on `source_id`, `clip_index`, and `frame_idx`
- Uses a clip-level train/test split to avoid frame-level leakage
- Reports accuracy/precision/recall/F1 + confusion matrix
- Optionally writes held-out predictions, report JSON, and model artifact

## Next steps

To make this pipeline a stronger long-term team baseline, prioritize:

1. **Formalize homography storage in the database**
   - Persist calibration artifacts/metadata (not just local files) so every run can resolve the correct session/video homography reproducibly.
2. **Expand and iterate on feature engineering**
   - Add richer court-space, pose-derived, and (optionally) tracking-aware features beyond the current baseline set.
3. **Run larger, condition-diverse samples**
   - Increase random S3 sample sizes and evaluate across varied matches/camera conditions before treating metrics as stable.
4. **Conform more explicitly to the staged cache pipeline in `cv_pipeline.md`**
   - Keep file-schema contracts explicit between phases, keep phase outputs cacheable/re-runnable independently, and align outputs toward the broader phase map (calibration -> detection/pose -> tracking -> features/windowing -> classifier evaluation).
5. **Integrate formats with the evaluation pipeline**
   - Standardize/cache schemas and output artifacts so this simplified flow can plug directly into the shared evaluation pipeline with minimal conversion glue.

