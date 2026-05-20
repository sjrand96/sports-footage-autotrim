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
