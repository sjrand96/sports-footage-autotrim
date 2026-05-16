#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ../.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ../.env
  set +a
fi

MANIFEST="${MANIFEST:-data/window_manifest_from_feature_extractions.jsonl}"
FEATURE_PREFIX="${FEATURE_PREFIX:-s3://sports-footage-autotrim-bucket/feature_extractions}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/s3_feature_fusion_$(date +%Y%m%d_%H%M%S)}"
FUSION="${FUSION:-early}"
WANDB_PROJECT="${WANDB_PROJECT:-volleyball-playtime}"
WANDB_ENTITY="${WANDB_ENTITY:-cs348k-sports-footage-autotrim}"
WANDB_RUN="${WANDB_RUN:-transformer-${FUSION}-s3-features}"
S3_CACHE_DIR="${S3_CACHE_DIR:-data/s3_cache}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-data/s3_cache/clips}"

python src/training/train.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --use-raw-frames \
  --e2e-features-dir "${FEATURE_PREFIX}" \
  --fusion "${FUSION}" \
  --s3-cache-dir "${S3_CACHE_DIR}" \
  --clip-cache-dir "${CLIP_CACHE_DIR}" \
  --config configs/train_config.json \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run "${WANDB_RUN}"
