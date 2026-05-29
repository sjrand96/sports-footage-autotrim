#!/usr/bin/env bash
set -euo pipefail

MANIFEST="training-pipeline/data/window_manifest_full127_test.jsonl"
FEATURES_DIR="training-pipeline/data/embeddings_resnet18"
E2E_FEATURES_DIR="s3://sports-footage-autotrim-bucket/feature_extraction/full_127_limitedconcurr_20260518/test/"
OUTPUT_ROOT="training-pipeline/outputs"
BATCH_SIZE=32
NUM_WORKERS=4
FUSION="early"
S3_CACHE_DIR="training-pipeline/data/s3_cache"
CLIP_CACHE_DIR="training-pipeline/data/s3_cache/clips"
FRAME_LABELS_DIR="s3://sports-footage-autotrim-bucket/feature_extraction/full_127_limitedconcurr_20260518/test/"

SUBSETS=(base all counts pairwise net_dist centroids mocon spatial pose_angles actions temporal)

for subset in "${SUBSETS[@]}"; do
  run_dir="${OUTPUT_ROOT}/e2e_${FUSION}_${subset}"
  checkpoint="${run_dir}/best.pt"
  metrics_out="${run_dir}/test_metrics.json"
  frame_out="${run_dir}/test_frame_metrics.json"
  timeline_out="${run_dir}/test_timeline.png"

  if [[ ! -f "$checkpoint" ]]; then
    echo "=== Skipping ${subset}: checkpoint not found at ${checkpoint} ==="
    continue
  fi

  echo "=== Evaluating subset: ${subset} ==="
  python training-pipeline/src/training/evaluate.py \
    --manifest "$MANIFEST" \
    --features-dir "$FEATURES_DIR" \
    --e2e-features-dir "$E2E_FEATURES_DIR" \
    --fusion "$FUSION" \
    --e2e-feature-subset "$subset" \
    --checkpoint "$checkpoint" \
    --output "$metrics_out" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --s3-cache-dir "$S3_CACHE_DIR" \
    --clip-cache-dir "$CLIP_CACHE_DIR" \
    --frame-eval center \
    --frame-labels-dir "$FRAME_LABELS_DIR" \
    --frame-output "$frame_out" \
    --timeline-output "$timeline_out"

done
