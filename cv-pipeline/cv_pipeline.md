# Volleyball CV Annotation Pipeline

A staged computer vision pipeline that extracts handcrafted features from static-camera volleyball footage to feed a downstream "playing vs. not-playing" classifier. Designed so phases can be developed in parallel with cached intermediate outputs.

---

## 1. Pipeline Overview

### 1.1 Working Philosophy: Staged Caches

Each phase of the pipeline reads from saved intermediate files and writes saved intermediate files. Phases don't call each other in code — they communicate through files on disk. This has a few consequences worth understanding before starting:

- **Phases can be developed independently and in parallel.** A collaborator working on feature extraction doesn't have to wait for detection to be finished. They can hand-write a small placeholder file with the right schema and develop against it, then swap in real data when upstream is ready.
- **Expensive steps don't get rerun unnecessarily.** Running detection on a full match takes a long time. Once it's saved, downstream phases just read the file. You only rerun detection when the detection code itself changes.
- **The file schema becomes the contract between phases.** Agreeing on column names and types upfront is the most important coordination step.
- **Cache invalidation is manual.** If you change a phase's code in a way that changes its output, delete that phase's cached file before rerunning. Nothing tracks this automatically — be deliberate about it.

For tabular intermediate outputs (detections, poses, features) we'll use **Parquet** files. Parquet is a columnar, typed, compressed file format — think of it as a much faster and smaller replacement for CSV. Pandas reads and writes it natively (`pd.read_parquet`, `df.to_parquet`); you'll need `pyarrow` installed. Key properties:

- Reading only the columns you need is fast (it skips the others on disk)
- Files are typically 5–10× smaller than equivalent CSVs
- Types are preserved — no re-parsing strings into floats every load
- Loading is roughly an order of magnitude faster than CSV for the data sizes we'll be dealing with

For non-tabular outputs (the homography matrix, model weights), use whatever format is natural — `.npz`, pickle, etc.

### 1.2 Phase Map

| # | Phase | Cadence | Owner | Output |
|---|-------|---------|-------|--------|
| 0 | Frame extraction | per-frame | shared utility | decoded frames + timestamp index |
| 1 | Court calibration & spatial projection | once + per-detection | Collaborator A | homography, court-space positions |
| 2 | Player detection & pose estimation | per-frame | Collaborator B | bounding boxes + keypoints |
| 3 | Player tracking | per-frame | Collaborator C | track IDs across frames |
| 4 | Per-frame feature extraction | per-frame | Collaborator A or B | feature vector per frame |
| 5 | Window aggregation | per-window | same as Phase 4 | feature matrix per window |
| 6 | Classifier training & evaluation | — | Collaborator C | trained model + metrics |
| 7 | End-to-end vision model | — | teammate (separate track) | trained video model |

Cadence tags:
- **per-frame** runs on every processed frame (recommend ~10 FPS, not full video FPS)
- **per-window** runs on a sliding temporal window
- **once** runs once per recording session (camera setup)

### 1.3 Suggested Cache Layout

```
data/
  <video_id>/
    frames/                   # phase 0
    detections.parquet        # phase 2
    poses.parquet             # phase 2
    court_positions.parquet   # phase 1
    tracks.parquet            # phase 3
    frame_features.parquet    # phase 4
    window_features.parquet   # phase 5
  court_calibration/
    <session_id>_homography.npz
    <session_id>_keypoints.json
  eval/
    <stage>_eval_set.parquet  # see Section 2: Benchmarking
models/
  classifier_v1.pkl           # phase 6
  vision_model_v1/            # phase 7
```

### 1.4 Suggested Development Order

1. **Week 1:** Phase 0 utility done, Phase 1 & 2 in parallel on real footage
2. **Week 2:** Phase 3 if needed; placeholder-driven work begins on Phases 4, 5, 6
3. **Week 3:** Real outputs flow into Phases 4–6; first end-to-end run on one match
4. **Week 4+:** Iterate, hand off feature outputs to Phase 7 work, pursue stretch goals

---

## 2. Benchmarking Approach

We want to know whether each pipeline change makes things better — both at the stage level (did detection improve?) and at the end-to-end level (did the classifier improve?). We don't want benchmarking to become a project of its own.

### 2.1 Guiding Principles

- **One number per stage.** Each stage gets one headline metric that goes up (or down) over time. Track it in a shared spreadsheet or CSV. More metrics are useful for debugging but only the headline number gets reported.
- **Small fixed eval sets, annotated once, reused forever.** ~30 frames per stage is enough to detect meaningful changes. The real cost of evaluation is annotation; once that's paid, re-running a benchmark is cheap.
- **Eval sets sampled across conditions.** When picking the 30 frames, deliberately include hard cases: net occlusion, players close together, lighting changes, fast motion. A balanced eval set surfaces failure modes; a lazy random sample doesn't.
- **End-to-end metric is the ground truth.** Stage-level metrics are diagnostics. The thing that actually matters is whether the classifier's segment-level IoU improves on a held-out match. If a stage metric improves but end-to-end doesn't, the stage metric was measuring the wrong thing.
- **Skip benchmarking what's not worth it.** If a stage is "off-the-shelf model with default params and we're not iterating on it," there's no need to benchmark it beyond an initial sanity check.

### 2.2 Per-Stage Benchmarks

| Phase | Headline Metric | Eval Set | Annotation Effort |
|---|---|---|---|
| 1 — Calibration | Mean reprojection error (px) at far baseline | 1 ref frame per session, court keypoints already annotated | none beyond what's already required |
| 2a — Detection | Recall on visible court players (precision as secondary) | 30 frames, all visible players boxed | ~1 hour total |
| 2b — Pose | % of keypoints within 10px of ground truth (PCK@10px) | 20 player crops, 17 keypoints each | ~1.5 hours |
| 3 — Tracking | ID switches per rally | 2-3 short rally clips | ~1 hour |
| 4 — Features | qualitative: feature timeline plots show separation | 1 rally + 1 between-points clip | none, visual check |
| 6 — Classifier | Per-window F1 + segment-level IoU | held-out match | comes from main labels, no extra |

Notes:

- **Phase 1 reprojection error** is essentially free — once the homography is fit, project the canonical court grid back onto the reference image and measure pixel distance from grid lines to painted court lines at known points. Far-baseline error is the most sensitive indicator.
- **Phase 2a recall** is the only thing that really matters for detection in this pipeline — false positives are filtered downstream by the court polygon. Annotating 30 frames with all visible players takes a couple hours. Use any tool — Label Studio, CVAT, or even an existing public volleyball dataset (see Prior Work) for the initial sanity check.
- **Phase 2b PCK@10px** is the standard pose metric. For a quick version, just visually check 20 skeletons against the source image — count keypoints that are clearly off. This isn't rigorous but it's useful for spotting regressions when you change models.
- **Phase 3 ID switches** are cheap to count manually for short clips. If population-level features are what Phase 4 ends up using, this benchmark may not matter much — defer it until Phase 4 commits to per-player features.
- **Phase 4 is qualitative.** A feature is "good" if its timeline visibly differs between playing and not-playing segments. This is a judgment call and trying to formalize it isn't worth the effort. If you want a quantitative version, compute the AUC of each feature alone as a single-feature classifier — features with AUC near 0.5 are useless and can be dropped.
- **Phase 6 metrics are the real ones.** Improvements in upstream stages should eventually show up here. If they don't, the upstream metric was measuring the wrong thing.

### 2.3 What to Track Over Time

A simple `benchmarks.csv` in the repo:

```
date,commit,phase,metric,value,notes
2025-09-15,a1b2c3d,detection,recall,0.91,YOLOv8m baseline
2025-09-22,e4f5g6h,detection,recall,0.94,YOLOv8x with conf=0.15
2025-09-22,e4f5g6h,classifier,segment_iou,0.71,baseline LightGBM
2025-09-29,i7j8k9l,classifier,segment_iou,0.78,added pose-derived features
```

This is the lightest-weight version of experiment tracking that's still useful. Anything fancier (MLflow, Weights & Biases) is fine if someone enjoys setting it up, but a CSV is sufficient.

### 2.4 What to Skip

- **No per-phase confidence intervals or statistical significance.** Eval sets are small and the signal we care about is large differences. If a metric moves by 1 percentage point we're not going to celebrate; if it moves by 5+ we are. Statistical rigor is overhead this project doesn't need.
- **No automated regression testing.** Just rerun the benchmarks manually when you make a change you expect to matter.
- **No benchmarking of frame extraction or window aggregation.** These are deterministic, simple, and unlikely to break in subtle ways. A unit test that checks output schemas is sufficient.

---

## 3. Prior Work

References organized by which phase they inform. None of these is a turnkey solution for our problem (single-camera static volleyball, playing/not-playing classification), but they're useful design references and citation sources.

### 3.1 Group Activity Recognition on Volleyball

The **Volleyball Dataset** (Ibrahim et al., CVPR 2016) is the canonical benchmark in this space — 4830 clips from 55 YouTube videos, with 8 group activity labels (left/right spike, set, pass, win-point) and 9 individual action labels (waiting, setting, digging, jumping, etc.). It's been used in dozens of follow-up papers. Worth knowing about even though our label space is simpler (binary playing/not-playing). Many of the architectural ideas in this lineage — hierarchical modeling of individual actions feeding into group activity, attention over actor relations — are directly relevant to your teammate's Phase 7 model.

- Ibrahim et al., *A Hierarchical Deep Temporal Model for Group Activity Recognition*, CVPR 2016. [arXiv:1607.02643](https://arxiv.org/abs/1607.02643)
- Han et al., *Dual-AI: Dual-path Actor Interaction Learning for Group Activity Recognition*, CVPR 2022. [arXiv:2204.02148](https://arxiv.org/abs/2204.02148) — strong recent baseline on the Volleyball Dataset
- Kim et al., *Detector-Free Weakly Supervised Group Activity Recognition*, CVPR 2022. [arXiv:2204.02139](https://arxiv.org/abs/2204.02139) — useful if your label data is sparse
- Chappa et al., *SoGAR: Self-supervised Spatiotemporal Attention-based Social Group Activity Recognition*, 2023. [arXiv:2305.06310](https://arxiv.org/abs/2305.06310)

For volleyball-specific action recognition (more granular than playing/not-playing — bump, set, spike, etc.):

- Waltner, Mauthner, Bischof, *Indoor Activity Detection and Recognition for Sport Games Analysis*, 2014. [arXiv:1404.6413](https://arxiv.org/abs/1404.6413) — the Graz volleyball activity dataset (serve, reception, setting, attack, block, stand, defense/move) is publicly available and could be useful as a Phase 2 detection sanity-check dataset
- Shih & Hsu, *Real-Time Action Detection in Volleyball Matches Using DETR Architecture*, MMM 2025 — recent, frames volleyball action detection as transformer-based detection

### 3.2 Sports Field Registration / Homography (Phase 1)

Most of this literature is from soccer because of broadcast-style moving cameras, which doesn't directly apply to our static setup. But the keypoint-detection-then-homography framing is the standard approach we're following.

- Chu et al., *Sports Field Registration via Keypoints-aware Label Condition*, CVPR Workshops 2022. [PDF](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf) — uses a grid of keypoints rather than sparse corners; relevant if we ever want to handle camera shift
- Theiner & Ewerth, *TVCalib: Camera Calibration for Sports Field Registration in Soccer*, WACV 2023. [arXiv:2207.11709](https://arxiv.org/abs/2207.11709) — full camera calibration rather than just homography; useful if 3D reasoning (jump heights in meters) becomes important
- Gutiérrez-Pérez & Agudo, *No Bells, Just Whistles: Sports Field Registration by Leveraging Geometric Properties*, CVPR Workshops 2024. [arXiv:2404.08401](https://arxiv.org/abs/2404.08401) — combines keypoint and line detection
- Drews et al., *Video-based Sequential Bayesian Homography Estimation for Soccer Field Registration*, 2024. [arXiv:2311.10361](https://arxiv.org/abs/2311.10361) — for the moving-camera case; not needed for our static setup but worth a citation

### 3.3 Player Detection & Tracking (Phases 2–3)

- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) — primary reference for Phase 3
- Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, 2022. [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) — extends ByteTrack with ReID and camera-motion compensation; the stretch option for Phase 3
- Cui et al., *SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes*, ICCV 2023. [arXiv:2304.05170](https://arxiv.org/abs/2304.05170) — standard sports tracking benchmark, includes volleyball
- Scott et al., *TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos*, CVPR Workshops 2024 — sports-specific MOT benchmark

### 3.4 Pose Estimation (Phase 2b)

- Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023. [arXiv:2303.07399](https://arxiv.org/abs/2303.07399) — good speed/accuracy default for Phase 2b
- Xu et al., *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation*, NeurIPS 2022. [arXiv:2204.12484](https://arxiv.org/abs/2204.12484) — stronger but slower alternative
- Lu et al., *RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation*, CVPR 2024 — one-stage option, simpler integration than top-down

### 3.5 End-to-End Video Models (Phase 7)

For your teammate's track:

- Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022. [arXiv:2203.12602](https://arxiv.org/abs/2203.12602)
- Wang et al., *VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking*, CVPR 2023. [arXiv:2303.16727](https://arxiv.org/abs/2303.16727)
- Feichtenhofer et al., *SlowFast Networks for Video Recognition*, ICCV 2019. [arXiv:1812.03982](https://arxiv.org/abs/1812.03982) — the multi-rate sampling idea is particularly relevant to volleyball (fast events embedded in slow context)
- Wang et al., *InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding*, 2024. [arXiv:2403.15377](https://arxiv.org/abs/2403.15377)

### 3.6 Public Datasets That Might Be Useful

- **Volleyball Dataset** (Ibrahim et al.) — 4830 clips, group activity labels. Useful for pretraining or for Phase 2 detection sanity-check.
- **Graz Volleyball Activity Dataset** (Waltner et al.) — individual action classes with bounding boxes. Useful for Phase 2 sanity-check.
- **SportsMOT** — general sports tracking benchmark including volleyball.
- **COCO** — pretraining target for Phases 2a and 2b detectors/pose models out of the box.

---

## 4. Phase Details

### Phase 0 — Frame Extraction

Shared utility, not a research task. OpenCV's `VideoCapture` is fine.

- Decode at ~10 FPS (not full video frame rate; full rate is overkill for this task and expensive downstream)
- Emit `(frame_idx, timestamp_sec, image)` — either save JPEGs to disk or yield in-memory depending on what downstream phases prefer
- Save a timestamp index so frame indices can always be mapped back to video time

**Deliverable:** a small Python module both collaborators can import. ~50 lines of code, no real milestones.

---

### Phase 1 — Court Calibration & Spatial Projection
**Owner:** Collaborator A

Combines one-time annotation, homography fitting, and the per-detection projection that makes positions interpretable. Owning this end-to-end keeps the spatial logic in one place.

#### Subgoals

**1a. Define and document the canonical court coordinate system.**
Origin under the net center. X along the net, Y along the long court axis, units in meters. Reference points for a standard 18m × 9m court (verify against the actual court being filmed):

| Point | Court (X, Y) |
|---|---|
| Far baseline corners | (±4.5, 9.0) |
| Near baseline corners | (±4.5, -9.0) |
| Far attack line endpoints | (±4.5, 3.0) |
| Near attack line endpoints | (±4.5, -3.0) |
| Centerline endpoints | (±4.5, 0) |
| Net post bases | (±5.0, 0) |

Net post tops add Z = 2.43m if you decide to extend to 3D later. Pin down whether you're using the inner or outer edge of the line for "the corner" and stick to the convention.

**1b. Annotate court keypoints on a reference frame.**
Pick one clean frame per recording session. The minimum is 4 visible non-collinear points; 8+ is recommended for a robust least-squares fit. Skip points that are out of frame.

Label Studio is one option for this: use a **separate** image project with `KeyPointLabels` (not the timeline project used for Playing/Downtime). The repo documents a ready-made template — label names, colors, import steps, and court semantics — in [docs/annotation_process/label-studio-setup.md](../docs/annotation_process/label-studio-setup.md#2b-optional-court-keypoints-project-homography). Other options: a quick matplotlib clicker, CVAT, or whatever annotation tool the team is comfortable with. The output just needs to be (label, x_pixel, y_pixel) pairs.

Best practices regardless of tool:
- Annotate at full video resolution
- Click line *intersections*, not centers
- For net post bases, click where the post meets the floor
- Pick a frame with no players occluding the lines

**1c. Compute and save the homography.**
`cv2.findHomography` with RANSAC is the standard approach. Save the matrix plus the source annotations so the calibration can be re-validated or re-fit later.

**1d. Project detections to court space.**
Given a bounding box, compute a foot point (midpoint of bottom edge is a reasonable default; using ankle keypoints from Phase 2 when confident is a stretch improvement). Apply the homography. Tag each projected position with which side of the net it's on. Filter detections whose court coordinates fall well outside the court — those are spectators or referees.

#### Validation
- **Homography sanity check:** project a canonical court grid back onto the reference image. Lines should overlay the painted court within ~3 pixels at the far baseline. Errors compound with perspective distance, so far-baseline error usually means a near-side keypoint was misclicked.
- **Position sanity check:** plot all per-frame court positions over a top-down court diagram for one rally. Players should appear in plausible places; no spectators should leak through the filter.

#### Headline Benchmark
Mean reprojection error in pixels at the far baseline. Target: < 5px. Track per session.

#### Baseline / Stretch
- **Baseline:** manual one-time annotation per session, foot-of-bbox projection, hard polygon filter for off-court detections.
- **Stretch:** use ankle keypoints from Phase 2 when they're high-confidence; train a keypoint detector for automatic re-calibration if the team ends up with footage from multiple courts.

#### Deliverables
- `homography.npz` per session
- A side-by-side visualization (reference frame with overlay + warped top-down view)
- `court_positions.parquet` schema: `frame_idx`, `det_idx`, `court_x`, `court_y`, `side`
- A top-down animation of player positions for one rally

---

### Phase 2 — Player Detection & Pose Estimation
**Owner:** Collaborator B

These two tasks live together because pose estimation typically runs on detection boxes (top-down pose), so they share an interface and benefit from being tuned together.

#### Subgoals

**2a. Player detection.**
Pretrained person detectors are the obvious starting point — YOLOv8, YOLOv10, RT-DETR, or similar. Whatever is convenient and runs at acceptable speed on the available hardware. Tunable knobs: confidence threshold, NMS IoU, input resolution.

Volleyball-specific failure modes to watch: players partially behind the net, distant players with small bounding boxes, spectators at frame edges. The first two are mitigated by lowering the confidence threshold (especially if Phase 3 tracking will be used). The third is best handled in Phase 1's court polygon filter rather than at detection time.

**2b. Pose estimation.**
Top-down pose models (run on each detection box) are recommended because the boxes are already filtered to court players. Many reasonable choices: RTMPose, ViTPose, MediaPipe, HRNet. Speed/accuracy tradeoffs differ; pick based on what runs at adequate speed on available hardware.

Use the COCO 17-keypoint format (most models output this natively) so downstream feature code has a stable schema.

**2c. Joint output schema.**
Detections and poses share a `(frame_idx, det_idx)` join key. Downstream code should be able to read either independently or join them.

#### Validation
- **Detection:** overlay boxes on 20 sample frames covering a mix of rally and non-rally moments. Manually count missed players and false positives.
- **Pose:** overlay skeletons on a similar sample. Spot-check during fast actions (spike, dive) where pose models often degrade. Check that low-confidence keypoints are correctly flagged in the output.

#### Headline Benchmarks
- **Detection:** recall on visible court players over a 30-frame eval set. Target: ≥95%.
- **Pose:** PCK@10px (% of keypoints within 10 pixels of ground truth) on 20 player crops. Target: depends on baseline; track over time.

#### Baseline / Stretch
- **Baseline:** off-the-shelf pretrained models, no fine-tuning, fixed confidence thresholds.
- **Stretch:** fine-tune detector on hand-labeled volleyball frames; swap in a stronger pose model for known-hard frames; add temporal smoothing across frames for poses.

#### Deliverables
- `detections.parquet` schema: `frame_idx`, `det_idx`, `x1`, `y1`, `x2`, `y2`, `conf`
- `poses.parquet` schema: `frame_idx`, `det_idx`, `keypoint_name`, `x`, `y`, `conf` (long format) *or* one column per keypoint (wide format) — agree with Phase 4 owner before committing
- Visualization scripts for both

---

### Phase 3 — Player Tracking
**Owner:** Collaborator C

Tracking adds stable identities across frames so downstream features can compute per-player quantities (velocity, time-stationary, etc.). Whether this is needed depends on whether the eventual feature set requires per-player tracking, which is partly an open question — population-level features (count, mean velocity, spread) don't need it.

#### Subgoals

**3a. Run a tracker over the cached detections.**
ByteTrack is the recommended starting point: pairs with any detector, no appearance model needed, handles occlusion well by associating low-confidence detections in a second pass. Other reasonable choices: BoT-SORT (adds optional ReID), OC-SORT, StrongSORT.

If using ByteTrack via Ultralytics, the key parameter is a low confidence threshold (e.g. `conf=0.1`) — the default defeats the point of the algorithm. Track buffer worth increasing for volleyball (~60 frames) since net occlusions can last over a second.

**3b. Post-process for volleyball-specific sanity.**
A track that crosses the net is almost certainly an ID switch (players don't change sides mid-rally). Use court coordinates from Phase 1 to detect and split these. Filter very short tracks as likely noise.

#### Validation
Render frames with track ID labels for one rally. Manually count ID switches.

#### Headline Benchmark
ID switches per rally on 2-3 hand-checked rally clips. Target depends on what downstream features need.

#### Baseline / Stretch
- **Baseline:** ByteTrack with default-ish parameters, no appearance model.
- **Stretch:** BoT-SORT with a ReID model fine-tuned on volleyball data; per-side track gating; integration with jersey number OCR if visible.

#### Deliverables
- `tracks.parquet` schema: extends detections with `track_id`
- Tracking visualization video for one rally

---

### Phase 4 — Per-Frame Feature Extraction
**Owner:** Collaborator A or B (whoever has bandwidth after their primary phase)

Compute interpretable features per frame from the cached upstream outputs. This is the phase that can start earliest with placeholder inputs — synthetic court positions and poses with hand-crafted "playing" and "not-playing" patterns let you develop and test feature logic before real data is ready.

#### Suggested feature categories

**From court positions:**
- Player counts (total, per side)
- Position spread per side (std, convex hull area)
- Distances from net (mean, min, max per side)

**From poses:**
- Hands-above-head count (any wrist above nose) — strong signal for active play
- Arms-platform count (both wrists at hip level, close together) — passing pose
- Highest joint y-position per side — for jump detection
- Mean torso angle — leaning/diving indicator

**From tracks (if available):**
- Per-player velocity, acceleration
- Stillness duration

These are suggestions, not a fixed list. The Phase 4 owner should treat the feature set as a research question and iterate based on what visibly separates playing from not-playing in the validation plots.

#### Validation
Plot each feature over time on a clip with known playing/not-playing segments. Features that visibly differ between segments are good candidates; features that look noisy across both are candidates to drop.

#### Headline Benchmark
Qualitative — does the feature timeline visibly separate playing from not-playing? If you want a quantitative version, compute single-feature AUC against the binary label; features with AUC near 0.5 are useless.

#### Baseline / Stretch
- **Baseline:** the categories above, no tracking-dependent features.
- **Stretch:** formation classification, inter-player distance features, pose-derived action detection (hit/set/dig), tracking-dependent per-player features.

#### Deliverables
- `frame_features.parquet`
- Feature timeline plots for one rally and one between-points segment

---

### Phase 5 — Window Aggregation
**Owner:** same as Phase 4

Aggregate per-frame features over short temporal windows to produce the feature vectors the classifier will train on.

#### Approach
Sliding windows (e.g., 1s window, 0.5s stride is a reasonable starting point — revisit based on what window length captures volleyball events without losing temporal resolution). For each window, compute summary stats over each per-frame feature (mean, std, min, max are obvious choices) plus temporal-only features:

- Count of frames with full team present
- Count of "hands-above-head" events in window
- Peak jump height in window

#### Validation
Verify windows align correctly with timestamps. Spot-check feature values for windows that span a known rally start — features should change visibly across the boundary.

#### Headline Benchmark
None at this phase — the next phase's classifier metric serves as the test of whether window aggregation is doing the right thing.

#### Baseline / Stretch
- **Baseline:** fixed 1s windows, mean/std/min/max per feature.
- **Stretch:** multi-scale windows (0.5s, 1s, 2s) concatenated; learned temporal aggregation via a small 1D CNN over per-frame features.

#### Deliverables
- `window_features.parquet`

---

### Phase 6 — Classifier Training & Evaluation
**Owner:** Collaborator C (or whoever owns the classifier track)

Train a binary classifier on the window features to predict playing vs. not-playing.

#### Approach
Gradient-boosted trees (LightGBM, XGBoost) are a strong baseline for this kind of feature-based binary classification. Logistic regression as an even simpler sanity check.

Two evaluation choices that matter more than model architecture:

- **Group splits by match.** Random window-level splits leak heavily because adjacent windows are highly correlated. Use `GroupKFold` on `match_id`.
- **Two metrics: per-window F1 and segment-level IoU.** Per-window metrics measure raw classification; segment-level measures whether predicted "playing" segments (after merging adjacent positives) align with true segments. The end user probably cares about segment quality.

Temporal post-processing at inference (median filter or HMM smoothing over the prediction sequence) typically improves segment IoU substantially at minimal cost. Worth including in the baseline.

#### Validation
- Confusion matrix and per-class precision/recall
- Segment-level IoU on a held-out match
- Qualitative review: build a debug video showing original frame, detected players, per-frame features, and predicted vs true label as a color bar at the bottom. This is the highest-leverage debugging artifact for the whole project — build it early.

#### Headline Benchmark
Segment-level IoU on the held-out match. This is the project's primary number — the one upstream improvements should ultimately move.

#### Baseline / Stretch
- **Baseline:** LightGBM with default hyperparameters, GroupKFold by match, median filter post-processing.
- **Stretch:** hyperparameter tuning, calibration, feature importance analysis to feed back into Phase 4, more sophisticated temporal smoothing.

#### Deliverables
- Trained model file
- Evaluation report (metrics + plots)
- Debug video for one test match

---

### Phase 7 — End-to-End Vision Model
**Owner:** teammate on separate track

Parallel effort that trains a video model directly on raw clips with the same playing/not-playing labels. The handcrafted-feature pipeline (Phases 0–6) and this end-to-end pipeline are evaluated against each other on the same test split.

#### High-level notes for coordination
- Likely architectures: VideoMAE-v2, InternVideo, SlowFast variants. Multi-rate temporal sampling is probably useful given the mix of fast events (spikes) and slow context (rotations).
- Outputs from Phases 1–4 (homography, court positions, poses) can be fed in as auxiliary inputs or used for cropping/attention guidance, but a strong video model should mostly learn from raw pixels.
- **Coordination point:** agree on clip format, label schema, and train/val/test splits with the Phase 6 owner before training starts. Both models must train and evaluate on identical splits or comparison is meaningless.

---

## 5. Open Questions

- **Frame rate:** 10 FPS is the suggested baseline. Revisit if jump-detection misses peak heights — might need 15–20 FPS.
- **Ball detection:** intentionally excluded from baseline. Can be added later as an extension to Phase 2 if features plateau.
- **Multi-session generalization:** the homography approach assumes one calibration per session. If matches from different gyms are mixed, calibrate separately and store per-video.
- **Label granularity:** define exactly what counts as "playing" before Phase 6. Does the toss before a serve count? Does a long rally with a pause count as one segment or two?
- **Tracking necessity:** revisit after Phase 4 baseline — if no compelling features need stable IDs, Phase 3 may not be worth the investment.
