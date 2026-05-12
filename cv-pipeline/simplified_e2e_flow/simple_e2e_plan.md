# Simplified end-to-end flow

A single-clip walkthrough of the full pipeline: pull a video, extract per-frame features, then train and evaluate a placeholder classifier against ground truth. Reuses existing helpers from elsewhere in the repo — don't reimplement S3, homography, pose, or Supabase plumbing.

Reference clip for the first run:
- `source_id = "jZ18INu4LQc"`, `clip_index = 6`
- S3: `s3://sports-footage-autotrim-bucket/clips/jZ18INu4LQc/jZ18INu4LQc_006.mp4` (us-west-2)
- Local cache: `cv-pipeline/pose-detection/media/clips/jZ18INu4LQc/jZ18INu4LQc_006.mp4`
- Homography: `cv-pipeline/calibration/out/homography.npz`

## 1. Per-frame feature extraction

A script that, for one clip, does the following:

- **Pull down the MP4.** Reuse `cv-pipeline/pose-detection/fetch_s3_clip.py` (`download_s3_object(bucket, key, dest, region)`). It loads AWS creds from the repo-root `.env` via `python-dotenv` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`). Skip the download if the destination already exists.
- **Yield frames at a target FPS.** Open the MP4 with OpenCV (`cv2.VideoCapture`), read `CAP_PROP_FPS`, and step `frame_interval = round(src_fps / target_fps)`. Default `target_fps = 2.0` (matches `cv-pipeline/pose-detection/pose_side_by_side_video.py`). For each yielded frame, record `frame_idx` (source frame index) and `timestamp_sec = frame_idx / src_fps`.
- **Get the homography for the clip.** For now, load `cv-pipeline/calibration/out/homography.npz` directly using the existing `_load_homography_npz` helper in `cv-pipeline/pose-detection/pose_side_by_side_video.py` (returns `H, wx_min, wx_max, wy_min, wy_max, out_w, out_h`). This is the placeholder source; later the per-clip homography will come from the DB.
- **Run the pose detector.** Use the same setup as `cv-pipeline/pose-detection/pose_side_by_side_video.py`: `ultralytics.YOLO("yolov8s-pose.pt")` with `imgsz=1280`, `conf=0.15`. For each frame, take `results[0].keypoints.xy` (shape `(N, 17, 2)` in pixel coords) and `results[0].keypoints.conf` (shape `(N, 17)`). COCO17 keypoint indices used here: `0` nose, `9` left wrist, `10` right wrist, `15` left ankle, `16` right ankle.
- **Project each player to court coordinates.** For each detection, compute the foot pixel `(u, v)` from the ankles (reuse `_foot_uv_from_coco17` from `cv-pipeline/pose-detection/pose_side_by_side_video.py`, `ankle_conf=0.25`), then `wx, wy = _image_uv_to_world_m(H, u, v)` (also from that file).
- **Drop players outside the court bounds.** Keep a detection only if `wx_min <= wx <= wx_max and wy_min <= wy <= wy_max` (from the NPZ meta). Revisit later.
- **Compute the per-frame features** (one row per yielded frame; column names and thresholds in the schema below):
    - `n_players_total`: count of in-court detections.
    - `n_camera_side` / `n_opposite_side`: split on `wy < 0` vs `wy >= 0`.
    - `n_front_row` / `n_back_row`: split on `abs(wy) < 3.0` vs `>= 3.0`.
    - `median_nearest_neighbor_dist`: for each in-court player, distance in metres to its closest other in-court player (Euclidean on `(wx, wy)`); take the median across players. `NaN` if `< 2` players.
    - `hands_above_head_count`: count of in-court players where `min(left_wrist_y, right_wrist_y) < nose_y` in **pixel** space (smaller y = higher in image). Skip a player if nose conf or both wrist confs are below `0.25`.
- **Cache the table.** Write one parquet per clip to `cv-pipeline/simplified_e2e_flow/cache/<source_id>_<clip_index:03d>_features.parquet`, matching the staged-cache convention in `cv-pipeline/cv_pipeline.md`.

## 2. Placeholder classifier

- **Look up the clip in Supabase.** Use `src.db.get_supabase_client()` (reads `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from `.env`), then `src.db.get_clip(client, source_id, clip_index)` to get `clip_id`.
- **Fetch the latest annotation for that clip.** Query `annotations` filtered by `clip_id`, ordered by `exported_at desc`, take the first row's `payload`. The payload shape is `{"label_studio_task": ..., "label_studio_annotation": ...}` (written by `data_labeling/push_timeline_annotation_export.py`).
- **Convert the LS annotation into per-frame `is_playing`.** From the chosen annotation's `result` list, take entries whose `value.timelinelabels == ["Playing"]`; each has `value.ranges = [{"start": <frame>, "end": <frame>}, ...]`. Frame numbers are at the LS template's `frameRate="30"` (see `docs/annotation_process/label-studio-setup.md`), so convert to seconds with `/ 30.0`. For each yielded frame in the feature table, set `is_playing = True` iff its `timestamp_sec` falls inside any Playing range, else `False`. Anything unlabeled is downtime by convention.
- **Join** ground truth onto the per-frame feature table on `frame_idx`.
- **Train an XGBoost classifier** on the feature columns directly, holding out one whole clip as the test set so adjacent frames don't leak labels across the split. (For the very first run with only one clip wired up, train and predict on the same clip just to validate the path end-to-end.)
- **Predict on the held-out clip** and store the per-frame predictions to `cv-pipeline/simplified_e2e_flow/cache/<source_id>_<clip_index:03d>_predictions.parquet` with columns `frame_idx, timestamp_sec, is_playing, pred_playing`.
- **Report precision / recall / F1** as the first number on the board.

Not yet, but later: run a ~1-second median filter over the predictions to kill flicker, and hand the per-frame predictions parquet to the video editor to consume.

## 3. Per-frame feature schema

| column | dtype | description |
| --- | --- | --- |
| `frame_idx` | int | join key |
| `timestamp_sec` | float | for label lookup |
| `n_players_total` | int | detections inside court polygon |
| `n_front_row` | int | `|court_y| < 3.0` |
| `n_back_row` | int | `|court_y| >= 3.0` |
| `n_camera_side` | int | `court_y < 0` |
| `n_opposite_side` | int | `court_y >= 0` |
| `median_nearest_neighbor_dist` | float | team-huddle signal, NaN if <2 players |
| `hands_above_head_count` | int | wrist y above nose y |
