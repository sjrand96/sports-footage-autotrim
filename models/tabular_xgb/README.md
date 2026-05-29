# Tabular XGBoost

Trains from **`feature_extraction/{run_id}/parquet/`**. By default, train/test assignment comes from run manifest metadata (`train_clip_ids`, `test_clip_ids`); pass `--split-json` to override with explicit clip IDs/keys (for K-fold/custom splits). Labels are **`is_playing`** in the feature parquets.

## Design

**Two-way eval contract per invocation (train / test clips).** We do not hold out a third val folder. Threshold tuning uses **grouped CV on train clips** (`StratifiedGroupKFold` by `clip_key`), so test stays untouched.

**Primary metric: Fβ (default β=2).** β=2 weights recall 4× more than precision—aligned with preferring false negatives (missed “playing”) over false positives.

**Threshold, not a custom XGB loss.** The booster still fits **logloss** (+ `scale_pos_weight`). We tune the **decision threshold** on out-of-fold train probabilities, then **refit on all train clips** and evaluate test with that threshold. This is cheaper and clearer than a custom objective.

**`cv_threshold.py` vs `train.py`.** Tuning logic lives in `cv_threshold.py` (pure numpy/sklearn, no I/O). `train.py` loads parquets, builds the model, calls the tuner, writes reports / W&B.

**Inference contract.** `xgb_report.json` includes `decision_threshold` and `f_beta`. At inference: `pred = proba >= decision_threshold` (see `apply_threshold` in `cv_threshold.py`).

## Weights & Biases

Setup and env vars: **[feature_extraction/WANDB.md](../../feature_extraction/WANDB.md)**.

```bash
pip install -e ".[ml]"

.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id full_127_limitedconcurr_20260518 \
  --wandb \
  --save-report-json feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_report.json \
  --save-model feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_model.json \
  --save-test-preds feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_test_preds.parquet \
  --save-test-csv feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_test_preds.csv
```

### Test predictions CSV (`--save-test-csv`)

One concatenated file for all held-out test frames (for eval viz):

| Column | Meaning |
|--------|---------|
| `clip_key` | `{source_id}_{clip_index}` |
| `clip_s3_uri` | S3 URI to clip media (from feature parquets) |
| `frame_idx` | Frame index in clip |
| `timestamp_sec` | Time in clip (seconds) |
| `prob_playing` | Model P(playing) |
| `pred_playing` | 0/1 at tuned `decision_threshold` |
| `is_playing` | Ground-truth label (0/1) |

## CLI flags (threshold / metrics)

| Flag | Default | Purpose |
|------|---------|---------|
| `--f-beta` | `2` | Fβ for OOF threshold search and primary test metric |
| `--split-json` | — | Optional explicit split file (`train/test` clip IDs or clip keys) |
| `--tune-threshold` / `--no-tune-threshold` | on | Grouped CV threshold tune on train clips |
| `--cv-folds` | `5` | Folds (capped by number of train clips) |
| `--decision-threshold` | — | Fixed threshold; skips CV |
| `--wandb-log-threshold-sweep` | off | Full OOF threshold curve in W&B (optional) |
| `--wandb-run-id` | — | Stable W&B id to resume/overwrite one train run |

With `--no-tune-threshold`, threshold **0.5** is used. Reports include both tuned (or fixed) metrics and **`metrics_at_threshold_0_5`** for comparison.

**W&B:** one run per `train.py --wandb` (CV does not multiply runs). Runs are grouped with feature publish under `group={feature_run_id}`. See [WANDB.md](../../feature_extraction/WANDB.md#dashboard--run-count).

## Usage (local only)

```bash
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id mini_fullfps_1clip \
  --save-report-json feature_extraction/_runs/mini_fullfps_1clip/xgb_report.json \
  --save-model feature_extraction/_runs/mini_fullfps_1clip/xgb_model.json
```

Requires at least one parquet in `parquet/` and a valid split assignment (manifest metadata or `--split-json`).

## FPS sensitivity (`fps_sensitivity.py`)

Simulates lower extract/inference FPS: subsample every *n*th frame **within each clip** (default FPS: 30, 15, 10, 5, 2, 1, 0.5), retrain XGB each time, evaluate with a **fixed** `decision_threshold` (from `xgb_report.json`, else 0.24). Includes an **always predict playing** baseline per FPS (recall=1; shows Fβ from label prevalence). Writes `fps_sensitivity.csv`, `.json`, and `.png`. No W&B.

```bash
.venv/bin/python models/tabular_xgb/fps_sensitivity.py \
  --feature-run-id full_127_limitedconcurr_20260518
```
