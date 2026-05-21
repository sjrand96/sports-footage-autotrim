# Eval

Quick viz for `xgb_test_preds.csv` (from `train.py --save-test-csv`).

**Use `--style strips` (default):** two timeline bands per panel — green = ground truth, red = prediction; filled = playing, gray = downtime. Scan vertically for match vs errors.

```bash
pip install -e ".[eval]"

CSV=feature_extraction/_runs/full_127_limitedconcurr_20260518/xgb_test_preds.csv

# One clip
.venv/bin/python eval/plot_preds_timeseries.py --csv "$CSV" --clip 1rXZJyVXUHU_003 --out eval/output/clip.png

# All test videos (recommended overview)
.venv/bin/python eval/plot_preds_timeseries.py --csv "$CSV" --all-sources --out eval/output/all_sources.png

# All test clips (tall; cap rows)
.venv/bin/python eval/plot_preds_timeseries.py --csv "$CSV" --all-clips --max-panels 8 --out eval/output/clips_sample.png
```

Legacy overlaid step lines: add `--style lines`.
