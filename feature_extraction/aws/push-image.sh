#!/usr/bin/env bash
# Build (linux/amd64 for Fargate x86) and push fe-worker to ECR. Run from repo root.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-214443970313}"
ECR_REPO="${ECR_REPO:-sports-footage-fe-worker}"
IMAGE_LOCAL="${IMAGE_LOCAL:-fe-worker}"
# Fargate task definition uses X86_64; image must be amd64 (not arm64 from Mac default).
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
GIT_TAG="$(git rev-parse --short HEAD)"

echo "Building ${IMAGE_LOCAL} for ${DOCKER_PLATFORM} ..."
docker build --platform "${DOCKER_PLATFORM}" -f feature_extraction/Dockerfile -t "${IMAGE_LOCAL}" .

echo "Tagging ${ECR_URI}:${GIT_TAG} and :latest"
docker tag "${IMAGE_LOCAL}" "${ECR_URI}:${GIT_TAG}"
docker tag "${IMAGE_LOCAL}" "${ECR_URI}:latest"

echo "Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Pushing..."
docker push "${ECR_URI}:${GIT_TAG}"
docker push "${ECR_URI}:latest"

echo ""
echo "=== Paste this into ECS task definition → Container → Image ==="
echo "${ECR_URI}:${GIT_TAG}"
echo ""
echo "Note: amd64 image runs on Fargate; local 'docker run' on Apple Silicon may be slower (emulation)."
