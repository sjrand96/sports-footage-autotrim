#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ../.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ../.env
  set +a
fi

PARQUET_PREFIX="${PARQUET_PREFIX:-s3://sports-footage-autotrim-bucket/feature_extractions}"
MANIFEST="${MANIFEST:-data/window_manifest_from_feature_extractions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/s3_${FUSION:-early}_$(date +%Y%m%d_%H%M%S)}"
FUSION="${FUSION:-early}"
WINDOW_SIZES_SEC="${WINDOW_SIZES_SEC:-2,3,4}"
STRIDE_SEC="${STRIDE_SEC:-1.0}"
MIN_POSITIVE_RATIO="${MIN_POSITIVE_RATIO:-0.5}"
S3_CACHE_DIR="${S3_CACHE_DIR:-data/s3_cache}"
CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-data/s3_cache/clips}"
CONFIG="${CONFIG:-configs/train_config.json}"
WANDB_PROJECT="${WANDB_PROJECT:-volleyball-playtime}"
WANDB_ENTITY="${WANDB_ENTITY:-cs348k-sports-footage-autotrim}"
WANDB_RUN="${WANDB_RUN:-transformer-${FUSION}-s3-parquet}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-1}"

if [[ "${FUSION}" != "early" && "${FUSION}" != "late" ]]; then
  echo "FUSION must be 'early' or 'late'." >&2
  exit 2
fi

python - <<'PY'
missing = []
for module in ("boto3", "pandas", "pyarrow", "torch", "torchvision"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python dependencies: "
        + ", ".join(missing)
        + "\nInstall them from the repo root with: python -m pip install -e \".[training]\""
    )
PY

mkdir -p "$(dirname "${MANIFEST}")" "${OUTPUT_DIR}" "${S3_CACHE_DIR}" "${CLIP_CACHE_DIR}"

if [[ "${REBUILD_MANIFEST}" == "1" || ! -f "${MANIFEST}" ]]; then
  echo "Building window manifest from paired parquets..."
  python src/data/build_window_manifest_from_parquets.py \
    --parquet-prefix "${PARQUET_PREFIX}" \
    --output "${MANIFEST}" \
    --s3-cache-dir "${S3_CACHE_DIR}" \
    --window-sizes-sec "${WINDOW_SIZES_SEC}" \
    --stride-sec "${STRIDE_SEC}" \
    --min-positive-ratio "${MIN_POSITIVE_RATIO}"
else
  echo "Using existing manifest: ${MANIFEST}"
fi

echo "Starting ${FUSION} fusion training..."
python src/training/train.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --use-raw-frames \
  --e2e-features-dir "${PARQUET_PREFIX}" \
  --fusion "${FUSION}" \
  --s3-cache-dir "${S3_CACHE_DIR}" \
  --clip-cache-dir "${CLIP_CACHE_DIR}" \
  --config "${CONFIG}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-run "${WANDB_RUN}"

echo "Done. Output directory: ${OUTPUT_DIR}"
