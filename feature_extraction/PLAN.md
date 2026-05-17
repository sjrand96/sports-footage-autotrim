# Feature pipeline plan

Working spec for feature extraction on S3, modular models, and AWS workers. Supersedes experimental flows in `cv-pipeline/simplified_e2e_flow/` and `cv-pipeline/pose-based-feature-extraction/` (remove after parity).

---

## Principles

- **Smallest useful change:** new packages + move training; reuse `cv-pipeline/calibration`, `cv-pipeline/pose-detection`, `src/db`.
- **Extractor is versioned and rerunnable:** each run is an immutable snapshot (`run_id`). Iterating on feature code or schema means a **new** `run_id` and a full re-extract — no in-place overwrites. Downstream models reference `feature_extraction_run_id` explicitly.
- **S3:** new prefixes alongside existing `clips/` layout; parquets record full URIs to source video (and optional frames).
- **No training in the extract job** (XGBoost and other heads live under `models/`).

---

## Repo layout

```
feature_extraction/          # Extract + label join + S3 upload
  PLAN.md                    # This file
  core/                      # Column registry, spatial helpers, per-frame row compute
  clip_split.py              # *** TRAIN/TEST ASSIGNMENT — edit here when split logic changes ***
  job.py                     # CLI (local + container); orchestrates clips + finalize report
  manifest.py                # Build/finalize run manifest (incl. run_report)
  timing.py                  # timings.json sidecar + per-clip stage timers
  s3_layout.py               # S3 path helpers
  Dockerfile                 # Fargate image (phase 4)
  latents/                   # PLACEHOLDER — CNN latents later (README only)

models/                      # One subfolder per model family
  tabular_xgb/               # Pooled XGBoost (moved from simplified_e2e_flow cache trainer)
    README.md

eval/                        # PLACEHOLDER — teammate-owned eval tool
  README.md
```

**Unchanged:** `cv-pipeline/calibration/`, `cv-pipeline/pose-detection/`, `src/db.py`, `data_labeling/`, annotation/S3 docs.

**Remove when parity:** `cv-pipeline/simplified_e2e_flow/`, `cv-pipeline/pose-based-feature-extraction/`.

---

## Feature extractor versioning & reruns

The extractor will evolve (new columns, heuristics, bugfixes). Design for **cheap full reruns**:

| Concept | Rule |
|---------|------|
| **`run_id`** | Unique per extract execution (e.g. timestamp + short hash, or W&B run id). Never reuse for different code/schema. |
| **`extractor_version`** | Semantic or git-based string in `manifest.json` (e.g. `0.1.0`, git sha). Bump when logic or `FEATURE_COLUMNS` changes. |
| **`feature_schema_version`** | Integer or string when parquet columns/dtypes change. Trainers validate against manifest. |
| **S3 prefix** | `feature_extraction/{run_id}/` is immutable once finalized. |
| **Rerun** | New extractor → new `run_id` → re-run all clips → new `train/` + `test/` trees. Old runs remain for comparison. |
| **Downstream** | `models/*` and eval take `--feature-run-id` or manifest URI; no implicit “latest” in production scripts. |

Optional later: CLI flag `--reuse-run-id` for dev-only retries of failed clips (same manifest, partial upload) — not required for v1.

---

## S3 layout

```
s3://{bucket}/
  clips/                                    # UNCHANGED
    {source_id}/{source_id}_{NNN}.mp4

  clips_v2/                                 # OPTIONAL (new ingests only; no backfill required)
    {source_id}/{clip_index}/
      video.mp4
      frames/                               # Only if extract job used --write-frames

  feature_extraction/{run_id}/
    manifest.json
    run_report.json
    timings.json
    train/
      {source_id}_{clip_index:03d}.parquet
    test/
      {source_id}_{clip_index:03d}.parquet

  model_prediction_outputs/{run_id}/        # FUTURE — standardized schema TBD (eval)
    {source_id}_{clip_index:03d}.parquet
```

**Parquet (v1):** one file per clip; **one row per decoded frame** (~30 fps). Columns: tabular features + `is_playing` + `source_id`, `clip_index`, `frame_idx`, `timestamp_sec` + provenance (`clip_s3_uri`, paths as needed). Full fps only — downstream models may subsample; extractor does not.

**Split:** at extract time into `train/` and `test/` via `clip_split.py` (see below). Manifest records split method, seed, and resolved clip id lists.

**Latents:** later under `feature_extraction/{run_id}/latents/` or sibling prefix; placeholder in repo only.

---

## Train / test assignment (`clip_split.py`)

**Single place to change split logic.** Module docstring should say: *replace `assign_train_test` implementation when moving from placeholder to hand-curated or automated splits.*

| Phase | Behavior |
|-------|----------|
| **v1 (placeholder)** | `assign_train_test(clips, *, test_fraction, seed)` — random clip-level split (e.g. sklearn-style or simple shuffle). CLI: `--test-fraction`, `--split-seed`. |
| **Next** | Hand-engineered lists: `TRAIN_CLIP_IDS` / `TEST_CLIP_IDS` or JSON file path; function returns explicit partition. |
| **Later** | Grouped split (e.g. by `source_id`), stratified rules, etc. — still only in `clip_split.py`. |

`job.py` calls `assign_train_test` once at run start; never embeds split rules elsewhere. Manifest stores:

- `split_method`: e.g. `"random_placeholder"` → `"explicit_clip_ids"` → `"grouped_source_id"`
- `split_seed`, `test_fraction` (when applicable)
- `train_clip_ids`, `test_clip_ids` (resolved lists for reproducibility)

---

## Per-clip failures & run report

**Do not fail silently.** A missing calibration, missing video, empty pose output, or extract exception must be recorded — but **one bad clip must not abort the whole run** unless `--fail-fast` is set (optional dev flag).

Per clip:

- Wrap extract in try/except; on failure append to `run_report["failures"]` with `clip_id`, `source_id`, `clip_index`, `stage` (`download` | `calibration` | `labels` | `extract` | `upload`), `error` message.
- On success append to `run_report["successes"]` with row count, output path.
- Log each failure at **ERROR** to stdout/stderr (visible in Fargate logs).

End of run (`job.py` + `manifest.py`):

- Write `manifest.json` always (even partial runs), including `run_report`:
  - `n_success`, `n_failed`, `failures[]`, `successes[]`
  - `status`: `"ok"` | `"partial"` | `"failed"` (`failed` = zero successes)
- Print a **summary block** to the console (counts + first N failure messages).
- **Exit code:** `0` only if `n_failed == 0`; `1` if any clip failed (partial or total). Callers/CI treat non-zero as “inspect manifest.”
- Optional: write `run_report.json` beside manifest for quick grep without opening full manifest.

Homography/pose “soft” issues (e.g. zero in-court players on many frames) are out of scope for v1 unless we add explicit row-quality checks later; hard missing prerequisites (no `court_calibrations` row, no annotation, file not found) always go in `failures[]`.

---

## `feature_extraction` job

1. Resolve clips from Supabase (annotated + court calibration).
2. **`clip_split.assign_train_test(...)`** → per-clip `train` | `test`.
3. Download video from existing `clips/` key (or `clips_v2/` when present).
4. Optional `--write-frames` → upload `frames/` under `clips_v2/...` (default off).
5. Every frame: YOLO pose → homography → feature row (current column set).
6. Join latest timeline annotation → `is_playing`.
7. Upload parquet to `feature_extraction/{run_id}/{split}/` (skip upload on clip failure).
8. Finalize `manifest.json` + **run report**; print summary; exit non-zero if any failures. **No training.**

Local dev: same CLI with `--out-dir` instead of S3 until wired.

---

## `models/tabular_xgb`

- Reads `feature_extraction/{run_id}/train` (and test for metrics).
- Validates columns against manifest `feature_schema_version`.
- Logs metrics; references `feature_extraction_run_id` for traceability.

Other model families add sibling folders under `models/`.

---

## `eval/` (placeholder)

Teammate-owned. Will read `model_prediction_outputs/{run_id}/`. README only for now.

---

## AWS

| Layer | v1 | Later |
|--------|-----|--------|
| **Worker** | **ECS Fargate** — one container image; typically **one task per clip** | **AWS Batch** — same image, queue many clips |
| **Orchestration** | Driver script: list clips → `RunTask` per clip → poll → finalize manifest | **Step Functions Map** when fan-out/retries need structure |

**Recommendation:** Fargate per clip + thin driver (local, CI, or CodeBuild). Add Batch when parallel clip count is painful. Step Functions only when retries/DLQ/workflow visibility are worth it.

Batch later: same Docker image and entrypoint; scheduler swap only.

**Env (task):** Supabase credentials, S3 bucket/region, optional W&B API key, `run_id`, clip id, split.

---

## Weights & Biases (later; design for it now)

| When | What | Where |
|------|------|--------|
| Extract run start | `run_id`, `extractor_version`, `feature_schema_version`, split seed, clip counts, git sha, image digest | `manifest.py` / `job.py` |
| Extract run end | Artifact `dataset` → S3 prefix + `manifest.json`; log `run_report.status`, `n_failed` | finalize after all clips |
| XGB train | hyperparams, `feature_extraction_run_id`, metrics | `models/tabular_xgb/` |
| Predict (future) | links to feature + model artifacts | `models/...` → `model_prediction_outputs/` |

Use `job_type` or separate projects (`feature-extraction`, `tabular-xgb`). Every downstream run stores `feature_extraction_run_id`.

---

## Implementation phases

| Phase | Deliverable |
|-------|-------------|
| **1** | `feature_extraction/` package: core, `clip_split.py` (random placeholder), full-fps extract, local parquet + manifest + run report |
| **2** | S3 upload + `timings.json` sidecar (`--upload-s3`, `--upload-only`) |
| **3** | `models/tabular_xgb/`; remove training from any extract path |
| **4** | Dockerfile + Fargate; driver for multi-clip |
| **5** | Delete `simplified_e2e_flow/`, `pose-based-feature-extraction/` |
| **Later** | W&B artifacts; Batch / Step Functions; `clips_v2`; `--write-frames`; latents; eval + `model_prediction_outputs/` |

---

## Non-goals (this pass)

- Migrating or deleting existing `clips/` keys.
- CNN latents, eval implementation, prediction output schema.
- Split-by-`source_id` without new `run_id`.
- Lambda workers.
