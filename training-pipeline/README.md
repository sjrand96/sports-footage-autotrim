# Volleyball Playtime vs Downtime Pipeline

This project builds a PyTorch training pipeline for volleyball playtime vs downtime detection using autotrim exports as the upstream data pipeline.

## Pipeline Steps

Primary S3 parquet workflow:

1. Read paired `*_features.parquet` and `*_predictions.parquet` files from `s3://sports-footage-autotrim-bucket/feature_extractions/`.
2. Build a window manifest from `is_playing` in the predictions parquet.
3. Train the transformer on raw clip frames plus frame-aligned parquet features.
4. Choose `--fusion early` or `--fusion late`.
5. Log metrics and diagnostics to W&B.

Legacy/local workflow:

1. Import autotrim annotations into a canonical clip manifest.
2. Build sliding windows from each 60-second clip.
3. Extract cached visual embeddings and optional pose features.
4. Train/evaluate with cached `.npz` embeddings.

The S3 parquet workflow expects paired files with matching stems, for example:

- `s3://sports-footage-autotrim-bucket/feature_extractions/jZ18INu4LQc_017_features.parquet`
- `s3://sports-footage-autotrim-bucket/feature_extractions/jZ18INu4LQc_017_predictions.parquet`

The feature parquet provides `clip_s3_uri`, so the training manifest does not need
to know clip locations ahead of time. S3 files are downloaded lazily into
`data/s3_cache`.

## Set up Weights & Biases

The training and evaluation scripts can log runs, tables, media, and confusion matrices to W&B. To use it:

1. Create or sign into your W&B account, then copy your API key from W&B account settings.

2. Install the SDK in your active Python environment:

```bash
python -m pip install wandb
```

3. Authenticate once on your machine. This stores your API key locally:

```bash
wandb login
```

If you prefer non-interactive setup, you can also export the key first and then login:

```bash
export WANDB_API_KEY=your_key_here
wandb login
```

4. Use your team as the W&B entity when logging runs. For this repo, the team is `cs348k-sports-footage-autotrim`.

For the sample smoke test, pass the entity explicitly:

```bash
python src/training/sample_wandb_train.py \
  --entity cs348k-sports-footage-autotrim \
  --project volleyball-playtime-sample \
  --run-name sample-data-transformer-smoke-test
```

5. Run training or evaluation with the default W&B project, or set your own project and run name:

```bash
python src/training/train.py --wandb-project volleyball-playtime --wandb-run my-experiment
python src/training/evaluate.py --wandb-project volleyball-playtime --wandb-run my-eval
```

If you want a local-only smoke test, either install W&B and pass `--wandb-mode disabled` to `sample_wandb_train.py`, or skip the dependency entirely and use the disabled mode.

## Setup

Run commands from `training-pipeline/` unless noted otherwise.

Install the training dependencies from the repo root:

```bash
python -m pip install -e ".[training]"
```

If you are using `requirements.txt` instead:

```bash
python -m pip install -r ../requirements.txt
```

Make sure AWS credentials are available through `.env`, environment variables,
or your normal AWS profile. The code uses `AWS_REGION` if set, otherwise
`us-west-2`.

Log in to W&B once:

```bash
wandb login
```

## S3 Parquet Training

### One-command run

This builds the window manifest from S3 parquets and then starts transformer
training with W&B logging:

```bash
scripts/run_s3_parquet_training.sh
```

By default this runs early fusion. To run late fusion:

```bash
FUSION=late scripts/run_s3_parquet_training.sh
```

Useful overrides:

```bash
FUSION=late \
OUTPUT_DIR=outputs/run_s3_late \
WANDB_RUN=raw-video-e2e-late \
scripts/run_s3_parquet_training.sh
```

The script defaults to:

- `PARQUET_PREFIX=s3://sports-footage-autotrim-bucket/feature_extractions`
- `MANIFEST=data/window_manifest_from_feature_extractions.jsonl`
- `WINDOW_SIZES_SEC=2,3,4`
- `STRIDE_SEC=1.0`
- `MIN_POSITIVE_RATIO=0.5`
- `S3_CACHE_DIR=data/s3_cache`
- `CLIP_CACHE_DIR=data/s3_cache/clips`
- `WANDB_PROJECT=volleyball-playtime`
- `WANDB_ENTITY=cs348k-sports-footage-autotrim`

Set `REBUILD_MANIFEST=0` to reuse an existing manifest:

```bash
REBUILD_MANIFEST=0 FUSION=early scripts/run_s3_parquet_training.sh
```

### Manual step 1) Build the window manifest from S3 parquets

```bash
python src/data/build_window_manifest_from_parquets.py \
  --parquet-prefix s3://sports-footage-autotrim-bucket/feature_extractions \
  --output data/window_manifest_from_feature_extractions.jsonl \
  --s3-cache-dir data/s3_cache \
  --window-sizes-sec 2,3,4 \
  --stride-sec 1.0 \
  --min-positive-ratio 0.5
```

This scans for matching `*_features.parquet` and `*_predictions.parquet` files,
uses `is_playing` as the frame-level label source, and writes windows with
`clip_s3_uri` as `clip_path`.

### Manual step 2) Train early fusion

```bash
python src/training/train.py \
  --manifest data/window_manifest_from_feature_extractions.jsonl \
  --output-dir outputs/run_s3_early \
  --use-raw-frames \
  --e2e-features-dir s3://sports-footage-autotrim-bucket/feature_extractions \
  --fusion early \
  --s3-cache-dir data/s3_cache \
  --clip-cache-dir data/s3_cache/clips \
  --config configs/train_config.json \
  --wandb-project volleyball-playtime \
  --wandb-entity cs348k-sports-footage-autotrim \
  --wandb-run raw-video-e2e-early
```

Early fusion concatenates the frozen ResNet frame embedding with the 35-column
E2E feature vector at each sampled timestep, then sends the combined sequence
through one transformer.

### Manual step 3) Train late fusion

```bash
python src/training/train.py \
  --manifest data/window_manifest_from_feature_extractions.jsonl \
  --output-dir outputs/run_s3_late \
  --use-raw-frames \
  --e2e-features-dir s3://sports-footage-autotrim-bucket/feature_extractions \
  --fusion late \
  --s3-cache-dir data/s3_cache \
  --clip-cache-dir data/s3_cache/clips \
  --config configs/train_config.json \
  --wandb-project volleyball-playtime \
  --wandb-entity cs348k-sports-footage-autotrim \
  --wandb-run raw-video-e2e-late
```

Late fusion runs separate transformer branches for raw-video embeddings and
parquet features, then concatenates the pooled branch outputs for classification.

### Manual step 4) Train-only launch script

After building `data/window_manifest_from_feature_extractions.jsonl`, you can use:

```bash
MANIFEST=data/window_manifest_from_feature_extractions.jsonl \
FUSION=early \
OUTPUT_DIR=outputs/run_s3_early \
WANDB_RUN=raw-video-e2e-early \
scripts/train_transformer_with_s3_features.sh
```

For late fusion:

```bash
MANIFEST=data/window_manifest_from_feature_extractions.jsonl \
FUSION=late \
OUTPUT_DIR=outputs/run_s3_late \
WANDB_RUN=raw-video-e2e-late \
scripts/train_transformer_with_s3_features.sh
```

The launch script defaults to:

- `FEATURE_PREFIX=s3://sports-footage-autotrim-bucket/feature_extractions`
- `WANDB_ENTITY=cs348k-sports-footage-autotrim`
- `S3_CACHE_DIR=data/s3_cache`
- `CLIP_CACHE_DIR=data/s3_cache/clips`

## Legacy Local Commands

Use these only if you are training from Label Studio/autotrim exports and local
cached `.npz` visual embeddings rather than the S3 parquet cache.

### 1) Import clips

```bash
python src/data/import_from_autotrim.py \
  --input /path/to/autotrim_export.jsonl \
  --output data/clip_manifest.jsonl
```

The importer also accepts the Label Studio-style `videoLabels` export format used by `../data/project-1-at-2026-04-28-04-06-975ac587.json`:

```bash
python src/data/import_from_autotrim.py \
  --input ../data/project-1-at-2026-04-28-04-06-975ac587.json \
  --output data/sample_clip_manifest.jsonl
```

For production training data, use Label Studio exports from S3-synced tasks. Each task should have `data.video` pointing to `s3://sports-footage-autotrim-bucket/clips/{source_id}/{source_id}_NNN.mp4` or the equivalent S3 HTTPS URL. Local Label Studio upload paths like `/data/upload/...` are not enough to resolve clips back to S3/Supabase.

### 2) Build window manifest from imported annotations

```bash
python src/data/build_window_manifest.py \
  --input data/clip_manifest.jsonl \
  --output data/window_manifest.jsonl \
  --window-sizes-sec 2,3,4 \
  --stride-sec 1.0
```

### 3) Split by match/source

```bash
python src/data/split_manifest.py \
  --input data/window_manifest.jsonl \
  --output-dir data/splits \
  --split-field source_id
```

### 4) Extract cached embeddings

```bash
python src/features/extract_visual_embeddings.py \
  --input data/clip_manifest.jsonl \
  --output-dir data/visual_embeddings \
  --fps 6
```

### 5) (Optional) Extract pose features

```bash
python src/features/extract_pose_features.py \
  --input data/clip_manifest.jsonl \
  --output-dir data/pose_features \
  --fps 6
```

### 6) Train with cached embeddings

```bash
python src/training/train.py \
  --manifest data/window_manifest.jsonl \
  --features-dir data/visual_embeddings \
  --pose-dir data/pose_features \
  --output-dir outputs/run_01 \
  --config configs/train_config.json
```

### 7) Train with raw frames (baseline)

```bash
python src/training/train.py \
  --manifest data/window_manifest.jsonl \
  --output-dir outputs/run_raw \
  --use-raw-frames
```

### 8) Evaluate

```bash
python src/training/evaluate.py \
  --manifest data/window_manifest.jsonl \
  --features-dir data/visual_embeddings \
  --pose-dir data/pose_features \
  --checkpoint outputs/run_01/best.pt \
  --output outputs/run_01/metrics.json
```
