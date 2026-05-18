# Fargate setup — sequential checklist

ECR image base: `214443970313.dkr.ecr.us-west-2.amazonaws.com/sports-footage-fe-worker`

Do these in order. Check each box before moving on.

---

## Step 1 — IAM role (one role for everything)

**Console:** IAM → Roles → create **new** role, or open your existing pipeline role.

### 1a. Trust relationship

IAM → your role → **Trust relationships** → **Edit trust policy** → paste entire contents of:

`feature_extraction/aws/ecs-task-role-trust.json`

Save.

### 1b. Permissions

IAM → your role → **Permissions** → **Add permissions** → **Create inline policy** → JSON tab → paste entire contents of:

`feature_extraction/aws/ecs-task-role-permissions.json`

Name it e.g. `fe-worker-ecs-and-s3`. Save.

(If this role already has your old S3-only inline policy, you can **replace** it with this file — it includes the same S3 statements plus ECR pull + logs.)

**Copy the role ARN** (e.g. `arn:aws:iam::214443970313:role/volleyball-pipeline`) — you need it twice in Step 4.

---

## Step 2 — Push Docker image to ECR

From **repo root** (Docker Desktop running):

```bash
chmod +x feature_extraction/aws/push-image.sh
./feature_extraction/aws/push-image.sh
```

Copy the line it prints at the end, e.g.:

`214443970313.dkr.ecr.us-west-2.amazonaws.com/sports-footage-fe-worker:abc1234`

That is your **image URI** for Step 4.

If push fails with ECR auth errors, your **local IAM user** needs ECR push (separate from the role above).

---

## Step 3 — CloudWatch log group

**Console:** CloudWatch → Log groups → **Create log group**

- Name: `/ecs/fe-worker`
- Retention: optional (e.g. 7 days)

---

## Step 4 — ECS task definition

**Console:** ECS → Task definitions → **Create new task definition** → **Create new task definition with JSON** skipped — use form:

| Field | Value |
|-------|--------|
| Task definition family | `fe-worker` |
| Launch type | AWS Fargate |
| OS / Arch | Linux / X86_64 |
| Task size CPU | 4 |
| Task size Memory | 8 GB |
| Task role | your role from Step 1 |
| Task execution role | **same role** from Step 1 |

**Container - 1:**

| Field | Value |
|-------|--------|
| Name | `fe-worker` |
| Image URI | paste from Step 2 (include `:gitsha` tag) |
| Essential | yes |
| Command | leave blank (set at run time in Step 5) |

**Environment variables** (add each):

| Key | Value |
|-----|--------|
| `SUPABASE_URL` | from your `.env` |
| `SUPABASE_SERVICE_KEY` | from your `.env` |
| `AWS_DEFAULT_REGION` | `us-west-2` |

Do **not** set `AWS_ACCESS_KEY_ID` on the task.

**Logging:**

- Log driver: `awslogs`
- Log group: `/ecs/fe-worker`
- Region: `us-west-2`
- Stream prefix: `ecs`

Create → note revision e.g. `fe-worker:1`.

---

## Step 5 — ECS cluster

**Console:** ECS → Clusters → **Create cluster**

- Name: `fe-worker` (or any name)
- Infrastructure: **AWS Fargate (serverless)**

Create.

---

## Step 6 — Run one task (smoke)

**Console:** Clusters → your cluster → **Run new task**

| Field | Value |
|-------|--------|
| Launch type | Fargate |
| Task definition | `fe-worker` (latest revision) |
| Platform version | LATEST |
| Cluster VPC | default VPC |
| Subnets | pick a **public** subnet |
| Security group | default (or one with outbound allowed) |
| Public IP | **Turned on** |

Expand container **fe-worker** → **Environment and secrets** optional; use **Command override** if your console supports it:

Comma-separated (some consoles):

```
feature_extraction/job.py,--out-dir,/tmp/fe,--clip-id,64,--max-frames,60,--upload-s3,--run-id,fargate_smoke_60f
```

Or override via CLI later.

**Run task** → open **Tasks** tab → click task → **Logs** (CloudWatch).

Wait until task **Stopped** with exit code 0.

---

## Step 7 — Verify on S3

```bash
aws s3 cp s3://sports-footage-autotrim-bucket/feature_extraction/fargate_smoke_60f/timings.json -
```

---

## Step 8 — Full clip benchmark

Run task again with command override:

```
feature_extraction/job.py,--out-dir,/tmp/fe,--clip-id,64,--upload-s3,--run-id,fargate_bench_1clip
```

Compare `sec_per_frame` to Docker and local runs.

---

## After code changes

1. `./feature_extraction/aws/push-image.sh` → new `:gitsha` tag  
2. ECS → Task definitions → `fe-worker` → **Create new revision** → update **Image** to new tag  
3. Run task again with a **new** `--run-id`
