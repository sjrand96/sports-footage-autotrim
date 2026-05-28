# Weights & Biases

Project: [cs348k-sports-footage-autotrim/volleyball-playtime](https://wandb.ai/cs348k-sports-footage-autotrim/volleyball-playtime)

Feature parquets stay on S3; W&B stores **references** (dataset artifact `playing-features:{run_id}`) plus local sidecars (`manifest.json`, `timings.json`, `run_report.json`). Training runs log metrics and an `xgb-playing-{run_id}` model artifact.

---

## API key (not project-specific)

W&B has **user** API keys, not per-project keys. The same key works for any team/project your account can write to.

1. Log in at [wandb.ai](https://wandb.ai).
2. Open **[wandb.ai/authorize](https://wandb.ai/authorize)** (or profile menu → **User settings** → **API keys** → copy).
3. Add to repo root `.env`:

   ```bash
   WANDB_API_KEY=your_key_here
   ```

4. Your W&B user must be a **member of the team** with write access. If publish fails with `permission denied`, check **`WANDB_ENTITY`** first (see below)—a wrong entity looks like a permissions problem.

`WANDB_ENTITY` and `WANDB_PROJECT` tell the SDK *where to log*; they are not part of the key.

**Entity vs project:** the team slug is `cs348k-sports-footage-autotrim` (first path segment in the UI URL), not `cs348k`. The project name is `volleyball-playtime`. Using `entity=cs348k` will fail even for full team members.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WANDB_API_KEY` | — | Required to publish or train with W&B |
| `WANDB_ENTITY` | `cs348k-sports-footage-autotrim` | Team slug (first segment of project URL) |
| `WANDB_PROJECT` | `volleyball-playtime` | Project name |

---

## CLI flags

### Fanout (`run_fanout.py`)

| Flag | Default | Purpose |
|------|---------|---------|
| `--wandb-publish` / `--no-wandb-publish` | on if `WANDB_API_KEY` set | After finalize, register `playing-features:{run_id}` |

### One-off publish (`python -m feature_extraction.wandb_publish`)

| Flag | Default | Purpose |
|------|---------|---------|
| `--run-id` | required | Feature extraction run id |
| `--runs-root` | `feature_extraction/_runs` | Local dir with `manifest.json` |
| `--bucket` | from manifest | S3 bucket override |
| `--entity` | `WANDB_ENTITY` | W&B entity override |
| `--project` | `WANDB_PROJECT` | W&B project override |

### Training (`models/tabular_xgb/train.py`)

| Flag | Default | Purpose |
|------|---------|---------|
| `--wandb` | off | Log run to W&B |
| `--wandb-entity` | `WANDB_ENTITY` | Entity override |
| `--wandb-project` | `WANDB_PROJECT` | Project override |
| `--wandb-run-name` | `xgb-{run_id}` | Run display name |

Install: `pip install -e ".[ml]"` (includes `wandb`, `xgboost`, etc.).

---

## Quick commands

```bash
# Backfill artifact for an existing run (after fanout, or if driver died before publish)
.venv/bin/python -m feature_extraction.wandb_publish --run-id full_127_limitedconcurr_20260518

# Train with W&B (uses artifact playing-features:{run_id} when present)
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id full_127_limitedconcurr_20260518 \
  --wandb
```

See also [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) (fanout) and [models/tabular_xgb/README.md](../models/tabular_xgb/README.md) (training).

---

## Dashboard / run count

**Grouped CV and threshold tuning do not create extra W&B runs.** Five XGB fits happen locally inside one `train.py --wandb` invocation → **one** training run with a few scalar metrics (and an optional threshold curve).

**Expected runs per feature extract `run_id`:**

| Action | W&B runs | Notes |
|--------|----------|--------|
| Fanout finalize or `wandb_publish` | **1** (`job_type=publish_features`) | Stable id `fe-publish-{run_id}` — re-publish updates the same row |
| `train.py --wandb` | **1 per invocation** | New run each time unless `--wandb-run-id` is set |
| Artifacts | versions, not runs | `playing-features:{run_id}`, `xgb-playing-{run_id}` |

**Organization:** publish and train runs share `group={feature_run_id}` so the UI groups them together. Filter by `job_type` (`publish_features` vs `train`) or tag `feature_run:…`.

**Avoid clutter:** use `--wandb-run-id xgb-myexperiment` only when you intentionally want to overwrite a prior train log. Skip `--wandb-log-threshold-sweep` unless you need the full curve (scalars `test/f2`, `decision_threshold`, `threshold_tuning/oof_fbeta` are logged by default). Delete stray debug runs from early smoke tests in the UI if needed.
