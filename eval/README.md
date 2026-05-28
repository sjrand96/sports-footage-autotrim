# Eval

Evaluation tools for frame-level prediction CSVs and full-video segment metrics.

## Full-video segment evaluation

Segment F1 should be computed on complete held-out videos, not on mixed train/test
clips from the same `source_id`. Start by creating or editing a source-level split
manifest:

```bash
.venv/bin/python eval/video_splits.py \
  --out eval/source_video_split.json \
  --test-source-id RCbQVAISMcU \
  --test-source-id TRt7udwisVU
```

The default manifest fields are:

- `train_source_ids`: normal training videos.
- `test_source_ids`: complete supervised held-out videos.
- `distribution_shift_source_ids`: complete supervised shift videos, such as beach volleyball or odd-angle footage.
- `unlabeled_source_ids`: full videos used only for behavior diagnostics.

For production-scale extraction, use ECS fanout with the source-level split so
every clip from a held-out video lands in `test/`:

```bash
.venv/bin/python feature_extraction/aws/run_fanout.py \
  --run-id fullvideo_eval_datasheet_v1 \
  --split-manifest eval/source_video_split.json \
  --plan-only

.venv/bin/python feature_extraction/aws/run_fanout.py \
  --run-id fullvideo_eval_datasheet_v1 \
  --start-only \
  --resume \
  --concurrency 10
```

Resume the same run after interruption or failed tasks:

```bash
.venv/bin/python feature_extraction/aws/run_fanout.py \
  --run-id fullvideo_eval_datasheet_v1 \
  --start-only \
  --resume \
  --concurrency 10
```

Fanout uploads parquets to S3. Sync the completed run locally before training
XGBoost:

```bash
aws s3 sync \
  s3://sports-footage-autotrim-bucket/feature_extraction/fullvideo_eval_datasheet_v1/ \
  feature_extraction/_runs/fullvideo_eval_datasheet_v1/
```

For smaller local smoke runs, `feature_extraction/job.py` also accepts the same
split manifest:

```bash
.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --run-id source_eval_xgb_v1 \
  --split-manifest eval/source_video_split.json \
  --skip-download
```

Train/evaluate XGBoost against that feature run and write a prediction CSV:

```bash
.venv/bin/python models/tabular_xgb/train.py \
  --feature-run-id source_eval_xgb_v1 \
  --save-test-csv feature_extraction/_runs/source_eval_xgb_v1/xgb_test_preds.csv \
  --save-report-json feature_extraction/_runs/source_eval_xgb_v1/xgb_report.json
```

Run segment evaluation grouped by full source video:

```bash
.venv/bin/python eval/evaluate_segments.py \
  --csv feature_extraction/_runs/source_eval_xgb_v1/xgb_test_preds.csv \
  --group-by source \
  --out-json feature_extraction/_runs/source_eval_xgb_v1/segment_eval.json
```

Conservative defaults:

```text
threshold: 0.35
window: 15 frames
max_gap: 45 frames
min_segment: 90 frames
boundary_tolerance: 30 frames
max_overlength: max(truth + 150 frames, truth * 2.0)
```

The evaluator reports segment precision/recall/F1, missed and false-positive
segments, boundary errors, frame metrics, kept-frame ratio, and true/predicted
segment duration stats. If `is_playing` is absent, it skips F1 and reports only
prediction behavior diagnostics.

For shift evaluation, populate `distribution_shift_source_ids` and run feature
extraction with:

```bash
.venv/bin/python feature_extraction/job.py \
  --out-dir feature_extraction/_runs \
  --run-id source_eval_xgb_shift_v1 \
  --split-manifest eval/source_video_split.json \
  --split-eval-group shift \
  --skip-download
```

## Suggested next iteration

After the first real source-level XGBoost run, inspect `segment_eval.json` before
adding automation. Good next steps are:

- Add a small grid sweep around `threshold`, `window_frames`, `max_gap_frames`, and `min_segment_frames`.
- Compare regular test videos and distribution-shift videos in separate reports.
- Add unlabeled-video behavior reports once full-video prediction CSVs can be exported without `is_playing`.
- Only add k-fold orchestration after the single source-level split path is trusted.

## Timeline visualization

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
