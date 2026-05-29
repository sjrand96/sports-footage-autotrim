#!/usr/bin/env bash
set -euo pipefail

MANIFEST="training-pipeline/data/window_manifest_full127_train.jsonl"
E2E_FEATURES_DIR="s3://sports-footage-autotrim-bucket/feature_extraction/full_127_limitedconcurr_20260518/train/"
FEATURES_DIR="training-pipeline/data/embeddings_resnet18"
OUTPUT_ROOT="training-pipeline/outputs"
BATCH_SIZE=32
NUM_WORKERS=4
FUSION="early"

SUBSETS=(base all counts pairwise net_dist centroids mocon spatial pose_angles actions temporal)

for subset in "${SUBSETS[@]}"; do
  run_dir="${OUTPUT_ROOT}/e2e_${FUSION}_${subset}"
  mkdir -p "$run_dir"

  echo "=== Training subset: ${subset} ==="
  python training-pipeline/src/training/train.py \
    --manifest "$MANIFEST" \
    --features-dir "$FEATURES_DIR" \
    --e2e-features-dir "$E2E_FEATURES_DIR" \
    --fusion "$FUSION" \
    --e2e-feature-subset "$subset" \
    --output-dir "$run_dir" 
    # --batch-size "$BATCH_SIZE" \
    # --num-workers "$NUM_WORKERS"

done
