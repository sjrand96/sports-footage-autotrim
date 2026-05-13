# Volleyball Playtime vs Downtime Pipeline

This project builds a PyTorch training pipeline for volleyball playtime vs downtime detection using autotrim exports as the upstream data pipeline.

## Pipeline Steps

1. Import autotrim annotations into a canonical clip manifest.
2. Build sliding windows (2-4 sec) from each 60-second clip.
3. Optionally split windows into train/val/test by source or match.
4. Extract frozen visual embeddings (and optional pose features).
5. Train a transformer classifier.
6. Evaluate window and segment metrics, plus W&B diagnostics.

## Set up Weights & Biases

The training and evaluation scripts can log runs, tables, media, and confusion matrices to W&B. To use it:

1. Install the SDK in your active Python environment:

```bash
python -m pip install wandb
```

2. Authenticate once on your machine:

```bash
wandb login
```

3. Run training or evaluation with the default W&B project, or set your own project and run name:

```bash
python src/training/train.py --wandb-project volleyball-playtime --wandb-run my-experiment
python src/training/evaluate.py --wandb-project volleyball-playtime --wandb-run my-eval
```

If you want a local-only smoke test, either install W&B and pass `--wandb-mode disabled` to `sample_wandb_train.py`, or skip the dependency entirely and use the disabled mode.

## Example Commands

### 0) W&B dashboard smoke test with sample data

Install and log in once:

```bash
python -m pip install wandb
wandb login
```

Run a tiny training job from the repository sample export in `../data`. It parses the Label Studio-style `videoLabels` ranges, builds labeled windows, and logs source segments, sample windows, metrics, a confusion matrix, validation predictions, the example frame, and a model artifact:

```bash
python src/training/sample_wandb_train.py \
  --project volleyball-playtime-sample \
  --run-name sample-data-transformer-smoke-test
```

For a local smoke test without W&B installed:

```bash
python src/training/sample_wandb_train.py --wandb-mode disabled --epochs 2
```

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

### 2) Build window manifest

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
