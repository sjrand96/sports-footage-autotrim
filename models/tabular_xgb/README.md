# Tabular XGBoost

Trains on **`feature_extraction/{run_id}/train/`**, evaluates on **`test/`** (split fixed at extract time). Labels are **`is_playing`** in the feature parquets.

## Weights & Biases

Setup, env vars, and flags: **[feature_extraction/WANDB.md](../../feature_extraction/WANDB.md)**.

```bash
pip install -e ".[ml]"

.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id full_127_limitedconcurr_20260518 \
  --wandb \
  --save-report-json feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_report.json \
  --save-model feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_model.json
```

## Usage (local only)

```bash
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id mini_fullfps_1clip \
  --save-report-json feature_extraction/_runs/mini_fullfps_1clip/xgb_report.json \
  --save-model feature_extraction/_runs/mini_fullfps_1clip/xgb_model.json
```

Requires both `train/` and `test/` to contain at least one parquet each (sync from S3 or run locally).
