# Cloud feature extraction deployment

Run `feature_extraction/job.py` on AWS to benchmark extract throughput and produce the same S3 artifacts as local (`feature_extraction/{run_id}/`, `timings.json`). Train XGBoost locally or on a laptop — not on the extract worker.

**Entrypoint:** `python feature_extraction/job.py` (same CLI as local).

**Benchmark metric:** `timings.json` → `clips[].derived.sec_per_frame` (extract stage only). Compare runs by `run_id`; do not overwrite old runs when iterating on features.

---

## Iteration loop (feature code changes)

Each extractor change should follow this cycle:

1. Edit `feature_extraction/core/` (and bump `extractor_version` / `feature_schema_version` in `core/version.py` when columns or logic change).
2. **Local smoke:** `--max-frames 60` on one `--clip-id` (fast sanity).
3. **Container smoke:** same flags in `docker run` (catches missing deps / paths).
4. **Push image** to ECR (tag with git sha or date; reuse `:latest` only while solo).
5. **Fargate run** with new `--run-id`; pull `timings.json` from S3 and compare to prior run.
6. Full clip(s) only when smoke passes.

New feature logic ⇒ new `run_id` ⇒ new S3 prefix. Old parquets stay for A/B.

---

## Step 1 — Container (local, no AWS)

**Deliverables:** `feature_extraction/Dockerfile`, `feature_extraction/requirements-worker.txt` (slim: opencv, ultralytics, torch **CPU**, pandas, pyarrow, supabase, boto3, python-dotenv, numpy, scipy — no Jupyter/SHAP/XGB).

**Image must include:**

- Repo: `feature_extraction/`, `src/`, `cv-pipeline/calibration/`, `cv-pipeline/pose-detection/` (homography + S3 fetch imports).
- Weights: `yolov8s-pose.pt` at repo root (COPY in build, or download once in Dockerfile).
- `WORKDIR` = repo root; `PYTHONPATH` = `.`

**Test 1 — smoke (inside container):**

```bash
# From repo root; requires Docker Desktop running and yolov8s-pose.pt at repo root
docker build -f feature_extraction/Dockerfile -t fe-worker .

docker run --rm --env-file .env \
  fe-worker \
  feature_extraction/job.py \
    --out-dir /tmp/fe \
    --clip-id 64 \
    --max-frames 60 \
    --upload-s3 \
    --run-id docker_smoke_60f
```

**Test 2 — one full clip (benchmark baseline):**

```bash
docker run --rm --env-file .env fe-worker \
  feature_extraction/job.py \
    --out-dir /tmp/fe \
    --clip-id 64 \
    --upload-s3 \
    --run-id docker_bench_1clip
```

Do **not** pass `--write-frames` for throughput benchmarks (doubles wall time). Do **not** use `--skip-download` in cloud (no pre-staged media).

**Pass criteria:** exit 0; S3 has `feature_extraction/{run_id}/timings.json` and one parquet under `train/` or `test/`.

---

## Step 2 — ECR

Repository: `214443970313.dkr.ecr.us-west-2.amazonaws.com/sports-footage-fe-worker`

```bash
./feature_extraction/aws/push-image.sh
```

Use the printed `…:gitsha` image URI in the ECS task definition (not `:latest` for benchmarks).

**First-time Fargate:** follow **[aws/FARGATE_SETUP.md](aws/FARGATE_SETUP.md)** (IAM JSON, console clicks, run task).

Re-push after every extractor change you want to benchmark in AWS.

---

## Step 3 — Fargate (CPU)

One ECS **task** runs `job.py` for one or more clips in a single process (manifest/timings written once at end). Start with **one clip**; scale to `--max-clips N` in the same task before fan-out.

### 3a — IAM (task role)

Attach to the **task role** (not execution role only):

- `s3:GetObject` on `arn:aws:s3:::{bucket}/clips/*`
- `s3:PutObject` on `arn:aws:s3:::{bucket}/feature_extraction/*`

Execution role: `AmazonECSTaskExecutionRolePolicy` (pull ECR, write CloudWatch Logs).

### 3b — Networking

- Default VPC, **public subnet**, **Assign public IP = ENABLED** (Supabase HTTPS + S3).
- Security group: outbound allowed (default SG is fine).

### 3c — Task definition (starting point)

| Setting | Value |
|---------|--------|
| Launch type | Fargate |
| CPU / memory | **4 vCPU / 8 GB** (adjust after first benchmark) |
| Platform | **LINUX / X86_64** |
| Ephemeral storage | 20–50 GB (one clip + temp; increase if many clips per task) |
| Logs | `awslogs` → CloudWatch log group |

**Environment (task):**

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Clip list, calibration, labels |
| `SUPABASE_SERVICE_KEY` | Same |
| `AWS_DEFAULT_REGION` | `us-west-2` (match bucket) |
| `S3_BUCKET` | Optional; default in code is `sports-footage-autotrim-bucket` |

Do not bake secrets into the image. Task role supplies S3 credentials (no `AWS_ACCESS_KEY_ID` in task env).

### 3d — Run task

**Smoke (first AWS ping):**

```
python feature_extraction/job.py \
  --out-dir /tmp/fe \
  --clip-id 64 \
  --max-frames 60 \
  --upload-s3 \
  --run-id fargate_smoke_60f
```

**Benchmark (one full clip):**

```
python feature_extraction/job.py \
  --out-dir /tmp/fe \
  --clip-id 64 \
  --upload-s3 \
  --run-id fargate_bench_1clip
```

Override via ECS **command** in task definition or `aws ecs run-task` container overrides.

**Verify:**

```bash
aws s3 cp s3://{bucket}/feature_extraction/fargate_bench_1clip/timings.json -
```

Compare `derived.sec_per_frame` to local `mini_fullfps_1clip` (~0.16 s/frame on Mac CPU is a reference; Fargate x86 will differ).

**Pass criteria:** `timings.json` on S3; `run_report.status` ok; `sec_per_frame` documented for go/no-go on CPU path.

### 3e — Tuning (still CPU)

If extract is too slow:

1. Try **8 vCPU / 16 GB** (same image).
2. Then consider GPU path (Step 4) — not more Fargate complexity.

**Multi-clip in one task:** omit `--clip-id`; use `--max-clips 5` (or full eligible list). Same task definition; longer wall clock; one manifest.

---

## Step 4 — GPU (only if Step 3 is too slow)

Fargate cannot run GPU. Reuse the **same** `job.py` and image layout; swap runtime:

| Option | When |
|--------|------|
| **ECS on EC2** (e.g. `g4dn.xlarge`) | Few dozen clips; want same ECS ops model |
| **AWS Batch** (GPU compute env) | Many clips; queue + auto scale |

**Image changes for GPU:**

- `torch` CUDA wheel matching instance NVIDIA driver.
- Optional: set YOLO `device=0` in `extract.py` (today unset → CPU everywhere).

**Benchmark:** same `timings.json` contract; compare `sec_per_frame` vs Fargate CPU before committing to GPU ops cost.

**Pass criteria:** materially lower `sec_per_frame` at acceptable $/clip for your batch size.

---

## Command reference (production-shaped)

| Intent | Flags |
|--------|--------|
| Single-clip AWS benchmark | `--clip-id <id> --upload-s3 --run-id <new>` |
| Multi-clip one task | `--max-clips N --upload-s3 --run-id <new>` |
| Fast sanity | `--max-frames 60` |
| Local-only debug | `--out-dir feature_extraction/_runs` (no `--upload-s3`) |

**IAM paths:** read `s3://{bucket}/clips/{source_id}/{source_id}_{NNN}.mp4`; write `s3://{bucket}/feature_extraction/{run_id}/…`.

**Observability:** CloudWatch Logs (stdout from `job.py`); `run_report.json` + `timings.json` on S3 for pass/fail without SSH.

---

## Checklist (in order)

- [x] Dockerfile + `requirements-worker.txt` (repo root: `docker build -f feature_extraction/Dockerfile -t fe-worker .`)
- [ ] Local `docker` smoke 60 frames (Docker Desktop must be running)
- [ ] Local `docker` one full clip + `timings.json` sanity
- [ ] ECR push
- [ ] ECS cluster + task definition + task role
- [ ] Fargate smoke 60 frames → S3 `timings.json`
- [ ] Fargate one full clip → record `sec_per_frame`
- [ ] (If needed) more vCPU, then GPU Step 4
