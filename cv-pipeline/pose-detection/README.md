# Pose detection (experiments)

This folder holds **ad-hoc experiments** around player pose and court top-down views: local scripts, one-off outputs, and manual media under `media/` and `out/` (both gitignored). Nothing here is the canonical pipeline yet.

Over time we expect this work to **line up with the staged CV spec** in [`cv-pipeline/cv_pipeline.md`](../cv-pipeline/cv_pipeline.md): stable cache paths (e.g. `data/<video_id>/poses.parquet`), agreed schemas, and phase boundaries instead of bespoke MP4s and PNGs.

**Scripts (today)**

- `fetch_s3_clip.py` — download a clip from S3 into `media/`.
- `foot_topdown_experiment.py` — single-frame YOLO pose + homography top-down (skeleton / top-down images).
- `pose_side_by_side_video.py` — sample a local clip at `--fps`, run pose + top-down per frame, write a side-by-side MP4 (H.264 via `ffmpeg` when available).

Dependencies follow the repo’s Python env (see root `requirements.txt`; needs OpenCV, Ultralytics/Torch, and optional `ffmpeg` on PATH for editor-friendly MP4 output).
