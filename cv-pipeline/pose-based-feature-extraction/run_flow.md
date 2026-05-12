# CV pose pipeline — overall run flow

This note describes how the **simplified E2E cache**, **pooled XGBoost trainer**, and **`feature_lab`** fit together. It matches the scripts under `cv-pipeline/simplified_e2e_flow/` and `cv-pipeline/pose-based-feature-extraction/feature_lab.py`.

---

## Prerequisites

- **Python environment**: project `requirements.txt` (YOLO / OpenCV / pandas / xgboost / scikit-learn; `shap` + `matplotlib` for `feature_lab pooled-explain`).
- **`.env`**: variables expected by `src.db` (Supabase) and S3 download paths used by `simple_e2e_pipeline.py`.
- **Local video**: E2E can download clips into `cv-pipeline/pose-detection/media/clips/{source_id}/…`, or reuse existing files with `--skip-download`.

---

## 1. Per-clip E2E — `simple_e2e_pipeline.py`

**Role:** For each clip, produce **per-frame feature rows** + **timeline labels** + single-clip sanity predictions, written as **paired parquets** in a cache directory.

**Typical flow per clip:**

1. Resolve the clip (e.g. `--clip-id <clips.id>` or `--random N` from clips that are annotated and whose `source_id` has `court_calibrations`).
2. Ensure the MP4 exists locally (S3 download unless `--skip-download`).
3. Load **homography** and world bounds for that clip’s `source_id` from Supabase (`court_calibrations`).
4. **Sample the video** at `--fps` (e.g. 2 Hz): run YOLO pose, map feet to court coordinates, apply in-bounds rules, compute every column listed in `cv-pipeline/simplified_e2e_flow/e2e_feature_columns.py` (`FEATURE_COLUMNS`: base counts + Chunk 1 spatial fields).
5. Write **`{source_id}_{clip_index:03d}_features.parquet`** (includes `frame_idx`, `timestamp_sec`, ids, paths, etc.).
6. Load the latest **timeline** annotation from Supabase and map “Playing” ranges → per-row **`is_playing`** (`add_ground_truth_labels`).
7. Run a **single-clip** XGBoost train/predict pass (wiring check; not the main benchmark).
8. Write **`{source_id}_{clip_index:03d}_predictions.parquet`** (join keys + `is_playing` + predictions).

**Default cache directory:** `cv-pipeline/simplified_e2e_flow/cache/` (override with `--cache-dir`).

**Examples:**

```bash
# One clip by Supabase id
python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --clip-id 69 --fps 2

# Many random eligible clips
python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --random 50 --fps 2
```

---

## 2. Pooled model — `train_pooled_xgboost_from_cache.py`

**Role:** Train **one** `XGBClassifier` on **all** clips that have matching `*_features.parquet` + `*_predictions.parquet`, with a **clip-level** train/test split (frames from the same clip stay entirely in train or entirely in test).

**Flow:**

1. Discover stems present in both `*_features.parquet` and `*_predictions.parquet`.
2. For each stem, **inner-join** features and predictions on `source_id`, `clip_index`, `frame_idx`; require `is_playing`.
3. Validate that feature parquets contain the columns implied by `--feature-subset`:
   - `all` — full `FEATURE_COLUMNS` from `e2e_feature_columns.py` (default).
   - `base` — original seven columns only (for A/B vs Chunk 1 on the same cache + seed).
4. `train_test_split` on **`clip_key`**, not on rows (`--random-seed`, `--test-size`).
5. Fill **NaN** on float feature columns with `-1.0` (see `FEATURE_FLOAT_FILLNA_COLS` / `float_fillna_cols_for_features`), fit XGBoost with `scale_pos_weight` from training-class balance.
6. Score held-out **test-clip rows**; print accuracy, precision, recall, F1, and confusion matrix.
7. Optionally save test predictions, JSON report, or model (`--save-test-preds`, `--save-report-json`, `--save-model`).

**Examples:**

```bash
python cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache

python cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache --feature-subset base
```

This script is the **primary metric** for “how good is the tabular head on this cache?”

---

## 3. Feature lab — `feature_lab.py`

Optional tooling on top of the same data and (for pooled mode) the **same training code**.

### `frame-viz`

For **one** decoder frame (`--clip-id`, `--frame_idx`), writes a PNG with:

- **Camera + YOLO pose** overlay for that exact frame (full frame rate; not the E2E `--fps` subsample).
- **Top-down** homography warp + court overlay + foot dots + side **centroid crosses** (see script for colors).
- **Text panels:** base seven features + Chunk 1 spatial columns from `compute_e2e_feature_row_from_yolo_result`.
- If `{stem}_predictions.parquet` exists under `--cache-dir` and contains that `frame_idx`, show cached **`is_playing`**; otherwise a short note.

Does **not** run the pooled model or SHAP.

### `pooled-explain`

Loads `train_pooled_xgboost_from_cache` as a module and calls **`load_paired_clip_rows`** + **`train_and_evaluate`** with the same hyperparameters / subset flags you pass — so **metrics match** the standalone trainer for the same inputs.

Then adds:

- `eval_report.json` (always written to `--out-dir`).
- **SHAP** (TreeExplainer) on held-out test rows (optional cap: `--shap-max-samples`).
- **Permutation importance** vs accuracy on the full test matrix (`--permutation-repeats`).
- Plots: `shap_bar.png`, `shap_beeswarm.png`, `shap_mean_abs.json`, `permutation_importance_accuracy.json`.

**Example:**

```bash
python cv-pipeline/pose-based-feature-extraction/feature_lab.py pooled-explain \
  --cache-dir cv-pipeline/simplified_e2e_flow/cache \
  --feature-subset all \
  --out-dir cv-pipeline/pose-based-feature-extraction/outputs/my_run
```

---

## 4. Typical “study” loop

1. **Refresh or grow the cache** — run E2E on the clips you care about, fixed `--fps`, so row semantics stay consistent.
2. **Pooled train / compare** — `train_pooled_xgboost_from_cache.py` with the same `--cache-dir` and `--random-seed`, toggling `--feature-subset base` vs `all`.
3. **Explain / debug** — `feature_lab pooled-explain` with matching flags; **`frame-viz`** on specific frames when geometry or labels look wrong.

---

## 5. Caveats (short)

- **Homography** is stored **per `source_id`** (whole match); **labels** are **per clip**. The pooled split is **per clip** today; many clips can share one `source_id`, so metrics are not automatically “new match” safe (see `pose-feature-extraction-plan.md` §6).
- **Feature schema:** new columns require **regenerated** `*_features.parquet`. Old files missing columns will fail validation for `--feature-subset all`.
- **Frame index vs E2E sample:** `frame-viz` uses any **source** `frame_idx`; E2E only writes rows at the configured `--fps`. Cached `is_playing` may be absent for frames that were never sampled.

---

## 6. Related docs

| Topic | Location |
|--------|----------|
| Feature roadmap, chunks, evaluation goals | `pose-feature-extraction-plan.md` |
| E2E behavior / cache layout | `cv-pipeline/simplified_e2e_flow/simple_e2e_plan.md` |
| Column registry | `cv-pipeline/simplified_e2e_flow/e2e_feature_columns.py` |
