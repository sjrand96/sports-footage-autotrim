# Pose-based play / not-play features — actionable plan

This document is the **working spec** for extending engineered features on top of the stack that already runs in-repo. It is written so chunks can be handed to an LLM with minimal extra context.

---

## 1. Goal

Improve a **binary play vs not-play** (timeline) classifier by **adding and validating tabular features** derived from **YOLO pose + static homography**, without rewriting ingest, DB, or labeling flows.

**Headline metric (target state):** PR-AUC on a **proper grouped split** (see §6). **Today’s quick path:** clip-level holdout + F1 in `train_pooled_xgboost_from_cache.py` is fine for smoke tests only.

---

## 2. Domain & data layout (read before coding)

### 2.1 Court coordinates (`wx`, `wy`)

- Player feet are mapped to **world coordinates in metres** using the same **`H_world_to_pixel`** inverse as in `cv-pipeline/calibration/court_homography.py` (stored in Supabase `court_calibrations.homography_matrix`).
- **Net line in the FIVB layout used here is `wy = 0`.**  
  **Current E2E convention** (matches `simple_e2e_pipeline.py` today):
  - **`n_camera_side`** = count of in-court players with **`wy < 0`**
  - **`n_opposite_side`** = count with **`wy ≥ 0`**
- **Front vs back row** in code uses **`|wy| < 3`** vs **`|wy| ≥ 3`** metres (rough FIVB attack-line bands), not left/right zones.
- When adding “near/far” or “zones”, **reuse the same axis convention** as existing columns, or explicitly document a sign flip. Do not silently invert `wy` relative to the homography already stored in the DB.

### 2.2 Clips vs source video

- **`source_id`** = one YouTube / `source_videos` id (one match recording).
- **`clip_index`** = 1-based index of a **~60 s segment** cut from that source (`clips` table). Many clips share one **`source_id`**.
- **Homography is one row per `source_id`** (`court_calibrations`): **static camera for the whole match**, reused for every clip from that source. Do not refit per clip unless you change the product definition.

### 2.3 Labels

- **Ground truth** comes from **timeline annotations** pushed to Supabase (`annotations.payload`), “Playing” ranges converted with **`--label-fps`** (default 30). See `add_ground_truth_labels` / `_extract_playing_ranges_seconds` in `simple_e2e_pipeline.py`.

### 2.4 Sampling rate

- Features are computed on **uniformly subsampled video frames** (`target_fps` / `--fps`), not every decoded frame. Any “window” length in §7 should be expressed in **seconds** or **number of sampled rows**, then mapped to counts given clip `--fps`.

---

## 3. What already exists (reuse; do not reimplement)

| Piece | Location | Role |
|-------|----------|------|
| Clip download + cache path | `simple_e2e_pipeline.py` — `ensure_local_clip`, `_local_clip_path` | S3 → `cv-pipeline/pose-detection/media/clips/{source_id}/…` |
| Homography from DB | `simple_e2e_pipeline.py` — `homography_arrays_from_court_calibration_row` via `src.db.get_court_calibration` | Same `H` + bounds for all clips of a `source_id` |
| YOLO pose + foot proxy | `simple_e2e_pipeline.py` — `POSE_MOD._foot_uv_from_coco17`, `_image_uv_to_world_m` | Ankle-based foot in image px → `(wx, wy)` |
| In-court filter | `extract_features_for_clip` | Drops detections outside `world_bounds` from calibration row |
| Current feature row | `extract_features_for_clip` + `FEATURE_COLUMNS` | 7 numeric features + `frame_idx` / `timestamp_sec` |
| Labels + parquet cache | `run_clip`, `add_ground_truth_labels` | `*_features.parquet`, `*_predictions.parquet` under `--cache-dir` |
| Pooled training | `train_pooled_xgboost_from_cache.py` | Joins features + preds on `source_id`, `clip_index`, `frame_idx`; **duplicate `FEATURE_COLUMNS` list** — must stay in sync when columns change |

**Implemented today vs prior feature table (high level):**

| # (prior table) | Name | Status vs repo |
|-----------------|------|------------------|
| 1 | Player count per side | **Partial:** `n_players_total`, `n_camera_side`, `n_opposite_side` (no raw detector count before court filter) |
| 1 | Front/back | **Yes:** `n_front_row`, `n_back_row` via `\|wy\|` |
| 3 (pairwise spread) | | **Partial:** `median_nearest_neighbor_dist` only (not mean/std/min/max) |
| 7 (arms overhead) | | **Partial:** `hands_above_head_count` (image y: wrists above nose), **not** per-side / fraction / conf-weighted variants |
| 2,4–6,8–22 | Zones, net distance stats, centroids, MOCON, pose library, windows, tracking | **Not implemented** in E2E yet |

---

## 4. Engineering principles (for all new work)

1. **Extend `extract_features_for_clip`** (or a module it imports) — keep one place that turns `(frame, pose, H, bounds)` → one dict/row per sampled frame.
2. **Update `FEATURE_COLUMNS` in both files:** `simple_e2e_pipeline.py` and `train_pooled_xgboost_from_cache.py` (or extract a shared `e2e_feature_columns.py` once duplication hurts).
3. **Backward compatibility:** new columns are additive until you bump a `features_version` column or path; old parquets in cache can be stale — document “delete cache or new subfolder when schema changes.”
4. **Numerics:** prefer finite floats; use sentinel (e.g. `-1.0` or `NaN` + fillna in trainer) consistently for “undefined” aggregates — match existing `median_nn` pattern.
5. **Tests:** add small unit tests for pure functions (zone bucketing, facing bins) under `cv-pipeline/` or `tests/` if you introduce a separate module.

---

## 5. Prioritized feature table (reference)

The detailed feature list (zones, net stats, MOCON, pose bins, windows, tracking-deferred items) remains the **north star** in rows 1–23 of the original brainstorming table at the top of git history; **§3** maps what is already shipped. Use the original row specs when implementing each new group.

**Pose row pattern (when you add Group B/C):** for each pattern, consider emitting **count**, **fraction of side players**, and **confidence-weighted count** as in the original plan — the current code only has a single global count for hands-above-head.

---

## 6. Evaluation — current vs target

**Today (`train_pooled_xgboost_from_cache.py`):**

- Split is **`GroupKFold`-style by clip** (`clip_key` = `source_id` + padded index), not by match.
- Many clips can share one **`source_id`** → **leakage risk** vs the plan’s “split by match”: train and test can contain different minutes of the **same match** with the **same homography**. Metrics are optimistic for “new match” generalization.
- Acceptable for **feature debugging** and relative comparisons **if** you hold the split seed fixed and only compare runs on the same cache snapshot.

**Target (align with §1 headline):**

- **Primary group:** `source_id` (match) for train/test split, or at least ensure test `source_id`s never appear in train.
- **Report PR-AUC** (and F1 at a chosen threshold) with **`average='weighted'`** or PR-AUC for positive class.
- **SHAP** (TreeSHAP) after meaningful feature sets.

Document benchmark rows in a small CSV (schema from original plan: `sprint_id`, `date`, `n_features`, `cv_pr_auc_mean`, …).

---

## 7. LLM-sized work chunks (execute in order)

Each chunk: **one PR-sized change**, clear files, clear “done when”.

### Chunk 0 — Documentation + column registry (no behavior change)

- **Do:** Add a 10-line comment block above `extract_features_for_clip` documenting `wy` sign, net at 0, and clip vs `source_id`.
- **Do:** Add `FEATURE_SCHEMA_VERSION = 1` constant and optional column in parquet if useful later.
- **Done when:** A new contributor can explain `n_camera_side` without reading FIVB files.

### Chunk 1 — Tier-1 spatial completion (no new pose heuristics)

- **Do:** Add from plan §Group A: raw in-bbox detection count (if exposed by YOLO results), or document why skipped; **pairwise** mean/std/min/max on `(wx,wy)` per side; **distance-to-net** stats per side (`|wy|` to 0 or signed `wy` depending on spec); **centroids** + `inter_centroid_dist`; **MOCON** distances to side centroid (#6).
- **Files:** `simple_e2e_pipeline.py`, `train_pooled_xgboost_from_cache.py` (`FEATURE_COLUMNS`), optional `pose_feature_spatial.py` if `extract_features_for_clip` exceeds ~120 lines.
- **Done when:** New columns appear in `*_features.parquet`; pooled script runs; SHAP optional.

### Chunk 2 — Court zones (12 bins)

- **Do:** Implement `zone_{near,far}_{FL,FC,FR,BL,BC,BR}` counts + fractions from plan; define fixed `wx`/`wy` breakpoints consistent with regulation court (reuse bounds from calibration row).
- **Done when:** Zone columns stable on one labeled clip visual sanity check (print or debug frame).

### Chunk 3 — Group B pose aggregates

- **Do:** Arms overhead **per side** + fractions + conf-weighted; add platform / hands-on-hips / bent-over heuristics from plan with shared conf threshold constant.
- **Reuse:** `_hands_above_head_for_player` pattern; extend with shoulder/hip indices from COCO17.
- **Done when:** At least one new pose aggregate improves CV or test PR-AUC vs Chunk 2 snapshot (same split seed).

### Chunk 4 — Group C orientation (optional / noisier)

- **Do:** Shoulder-line angle → 4 bins; face visibility heuristic; coherence + mutual-facing pairs.
- **Done when:** Separate ablation: with vs without Group C on same cache.

### Chunk 5 — Window aggregates (Group D)

- **Do:** Rolling mean/std/min/max over **K sampled rows** (configurable seconds × fps); condition fractions; selective slopes for top features by SHAP.
- **Note:** Requires either online ring buffer in extraction pass or second pass over saved per-frame parquet — second pass is simpler for LLM work.

### Chunk 6 — Evaluation upgrade

- **Do:** Add `source_id`-grouped split option to pooled trainer (or new `train_pooled_xgboost_grouped.py`); add PR-AUC; optional SHAP export.
- **Done when:** Benchmark CSV row shows test PR-AUC with match-safe split.

### Chunk 7 — Tracking-backed features (deferred)

- **Do:** After ByteTrack (or similar) exists in repo, add plan rows 16–20 (speed, motion, jump).
- **Until then:** leave explicitly out of scope in PR descriptions.

---

## 8. Out of scope (unchanged)

- Ball-only models, full action recognition, permutation importance (unless needed).
- Replacing XGBoost as the default tabular head before feature set stabilizes.

---

## 9. Quick reference — commands

```bash
# Regenerate features for one clip (DB homography)
python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --clip-id <clips.id> --fps 2

# Pooled model on cache (after FEATURE_COLUMNS updated)
python cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py --cache-dir <path> --test-size 0.2
```

---

## 10. File touch map (for LLM prompts)

| Task | Primary files |
|------|----------------|
| New per-frame features | `cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py` (`extract_features_for_clip`, `FEATURE_COLUMNS`) |
| Training / metrics | `cv-pipeline/simplified_e2e_flow/train_pooled_xgboost_from_cache.py` |
| Homography contract | `cv-pipeline/calibration/homography_io.py`, `docs/annotation_process/court_calibration_supabase.md` |
| Label contract | `data_labeling/push_timeline_annotation.py`, `simple_e2e_pipeline.py` (`_extract_playing_ranges_seconds`) |

---

## Appendix A — Prioritized feature table (full spec)

| # | Feature | Tier | Inputs needed | What to emit | Why it's worth implementing | Notes |
|---|---------|------|---------------|--------------|------------------------------|-------|
| 1 | Player count per side | 1 | Detections + homography | `count_near`, `count_far`, `count_total`, `count_detections_raw` | Baseline phase signal; ~6 per side during play | Always emit raw detection count alongside; lets you detect when counts are unreliable due to occlusion |
| 2 | Court-zone occupancy | 1 | Detections + homography | 12 counts: `zone_{near,far}_{FL,FC,FR,BL,BC,BR}` plus 12 fractions of total side count | Captures formation; pre-serve and huddle have very different occupancy patterns | Keep zone boundaries simple (front/mid/back × left/center/right) |
| 3 | Pairwise spread per side | 1 | Detections + homography | Per side: `mean_dist`, `std_dist`, `min_dist`, `max_dist` | Huddles have low min/mean; formations have higher more uniform spread | O(n²) but n ≤ 6 per side |
| 4 | Player-to-net distance stats per side | 1 | Detections + homography | Per side: `mean_net_dist`, `std_net_dist`, `min_net_dist`, `max_net_dist` | Formation depth; pre-serve has high mean and high std | Sign of court-y distinguishes which side |
| 5 | Centroid position per side + inter-centroid distance | 1 | Detections + homography | `centroid_near_{x,y}`, `centroid_far_{x,y}`, `inter_centroid_dist` | During play centroids stay roughly centered; dead-ball clustering shifts them | |
| 6 | Per-player distance to own-team centroid | 1 | Detections + homography | Per side: `mean_dist_to_centroid`, `std_dist_to_centroid`, `max_dist_to_centroid` | The MOCON-style feature from Zahra et al.; one of their top-2 modalities (26% weight). Distinguishes spread formation from clustered huddle directly | Compute team centroid first (#5), then per-player distances to it |
| 7 | Arms-overhead pose | 2 | YOLO pose (wrist, nose keypoints) | Per side: `count_any_arm_up`, `count_both_arms_up`, `frac_any_arm_up`, `frac_both_arms_up`, `confweighted_any_arm_up` | Strongest single pose feature; covers spike, block, set, celebration, ref signal | "Above nose y" is more robust than derived head-top; require keypoint confidence > 0.3 |
| 8 | Platform pose (passing/digging) | 2 | YOLO pose (wrists, shoulders, hips) | Per side: `count_platform`, `frac_platform`, `confweighted_platform` | Specific to passing/digging; strong play signal | Both wrists below shoulders, wrist-to-wrist distance < 0.5× shoulder width, elbows approximately extended |
| 9 | Hands-on-hips pose | 2 | YOLO pose (wrists, hips, shoulders) | `count_hands_hips`, `frac_hands_hips`, `confweighted_hands_hips` | Classic between-point tired posture; strong dead-ball signal | Wrists near hips, elbows out (wrist-x outside shoulder-x line) |
| 10 | Hands-on-knees / bent-over pose | 2 | YOLO pose (shoulders, hips, wrists) | `count_bent_over`, `frac_bent_over`, `confweighted_bent_over` | Another between-point fatigue posture | Torso angle (shoulder-to-hip vector) tilted significantly from vertical, wrists below hip y |
| 11 | Bounding box aspect ratio stats | 2 | Detections (no pose needed) | Per side: `mean_aspect`, `std_aspect`, `min_aspect` | Cheap proxy for stance/crouch when pose is unreliable; works even on far side | Low aspect = crouched OR mid-jump (both are signal) |
| 12 | Body-facing direction, near side | 3 | YOLO pose (shoulders, hips, face keypoints) + homography | Counts in 4 bins: `count_facing_{net,baseline,leftside,rightside}_near`, plus fractions and confidence-weighted versions | Strong play signal: near-side players overwhelmingly face net during play | Resolve front/back ambiguity using face keypoint visibility pattern (nose+eyes = facing camera; only ears = facing away) |
| 13 | Body-facing direction, far side | 3 | YOLO pose + homography | Same as #12 but for far side | Same intent as #12 but expect noisier results | Measure its value separately so you can drop if it doesn't add signal |
| 14 | Facing-direction coherence per side | 3 | Output of #12 + #13 | `facing_coherence_near`, `facing_coherence_far` | High coherence = play; low coherence = scattered attention | Use mode-fraction (largest bin / total) as simplest version |
| 15 | Mutual-facing-pair count | 3 | Output of #12 + #13 + court positions | `count_mutual_facing_pairs` | Strong conversation/huddle indicator; during play players don't face each other | Both facing each other (within tolerance) AND within 2m |
| 16 | Per-player speed | 4 | Tracking + court positions | `max_speed`, `p90_speed`, `mean_speed` | Play has bursts; dead time is slow drift | Smooth over 3-5 frames; cap speed at 10 m/s to filter ID switch artifacts |
| 17 | Count of players in motion | 4 | Tracking + court positions | `count_moving` | Robust simple signal: active play has many players moving | Threshold speed (~1.5 m/s) |
| 18 | Direction-change rate | 4 | Tracking + court positions | `count_direction_changes` | Rally play has frequent sharp cuts; dead-ball walking is smooth | Count players whose direction changed by >45° in last short window |
| 19 | Motion coherence per side | 4 | Tracking + velocities | `motion_coherence_near`, `motion_coherence_far` | During rallies motion is correlated (tracking ball); dead time is uncorrelated | Circular variance of velocity directions |
| 20 | Jump detection | 4 | Tracking + bbox/hip-y | `count_jumping`, `time_since_last_jump` | Any jump is near-certain play signal | Start simple: hip-y dropped X% from player's recent baseline |
| 21 | Window aggregates (mean/std/min/max) | 5 | Tiers 1-4 outputs | 4× multiplier on prior features | Standard temporal summaries | Their paper uses 5-frame windows; you can probably go longer (30-60 frames) for play/not-play |
| 22 | Window trend (linear slope) | 5 | Tiers 1-4 outputs | Slope of ~5-10 selected features | Capture "tightening" vs "spreading" formations over the window | Pick which features to slope-encode after first round of training |
| 23 | Window condition fractions | 5 | Tiers 1-4 outputs | E.g., `frac_frames_with_jump`, `frac_frames_any_arm_up_ge2` | Binary thresholds aggregated to window-level fractions | Robust to per-frame pose noise |

### Notes on pose feature columns (Appendix A table)

For each pose pattern (rows 7–10, 12–13), the original design called for **three** views: raw count, fraction of detected players on that side, and confidence-weighted count. The current E2E only implements a **single** global-style count for arms-above-head (`hands_above_head_count`); extending to the triple pattern is part of Chunk 3.

### Appendix B — Optional sanity check (post-pruning)

After a strong feature set exists, train **logistic regression** on the same pruned features once: if LR is within a few PR-AUC points of XGBoost, features carry linear signal; if much worse, trees are exploiting interactions.

### Appendix C — Feature selection (when feature count explodes)

1. Drop features in bottom quartile by mean |SHAP|.
2. For |Pearson r| > 0.95, drop the lower-|SHAP| member of the pair.
3. Retrain; if CV PR-AUC within ~1 point or better, keep the smaller set.
