# Tabular XGBoost

Trains on **`feature_extraction/{run_id}/train/`**, evaluates on **`test/`** (split fixed at extract time). Labels are **`is_playing`** in the feature parquets.

## Usage

```bash
# Local run (e.g. after feature_extraction job)
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id mini_fullfps_1clip \
  --save-report-json feature_extraction/_runs/mini_fullfps_1clip/xgb_report.json \
  --save-model feature_extraction/_runs/mini_fullfps_1clip/xgb_model.json

# Or explicit path
.venv/bin/python models/tabular_xgb/train.py \
  --run-dir feature_extraction/_runs/mini_frames_1clip \
  --feature-subset all
```

Requires both `train/` and `test/` to contain at least one parquet each.
