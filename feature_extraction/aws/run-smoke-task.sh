#!/usr/bin/env bash
# One-off Fargate smoke: 60 frames, clip 64, upload timings to S3.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
CLUSTER="${CLUSTER:-default}"
TASK_DEFINITION="${TASK_DEFINITION:-default-sports-footage-fe-worker-e01e:1}"
SUBNET="${SUBNET:-subnet-00f575df9d041d3ab}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-0b23a01f27c7eaf2c}"
CLIP_ID="${CLIP_ID:-64}"
RUN_ID="${RUN_ID:-fargate_smoke_60f}"
MAX_FRAMES="${MAX_FRAMES:-60}"

OVERRIDES_FILE="$(mktemp)"
trap 'rm -f "$OVERRIDES_FILE"' EXIT

cat >"$OVERRIDES_FILE" <<EOF
{
  "containerOverrides": [
    {
      "name": "Main",
      "command": [
        "feature_extraction/job.py",
        "--out-dir", "/tmp/fe",
        "--clip-id", "${CLIP_ID}",
        "--max-frames", "${MAX_FRAMES}",
        "--upload-s3",
        "--run-id", "${RUN_ID}"
      ]
    }
  ]
}
EOF

echo "Overrides:"
cat "$OVERRIDES_FILE"
echo ""

TASK_ARN="$(aws ecs run-task \
  --region "${AWS_REGION}" \
  --cluster "${CLUSTER}" \
  --launch-type FARGATE \
  --task-definition "${TASK_DEFINITION}" \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET}],securityGroups=[${SECURITY_GROUP}],assignPublicIp=ENABLED}" \
  --overrides "file://${OVERRIDES_FILE}" \
  --query 'tasks[0].taskArn' \
  --output text)"

echo "Started: ${TASK_ARN}"
echo ""
echo "Logs:  aws logs tail /ecs/fe-worker --region ${AWS_REGION} --follow"
echo "Check: aws s3 cp s3://sports-footage-autotrim-bucket/feature_extraction/${RUN_ID}/timings.json -"
