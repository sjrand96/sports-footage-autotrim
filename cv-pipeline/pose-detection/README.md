# Pose detection (experiments)

Ad-hoc pose and court top-down experiments; `media/` and `out/` are gitignored.

**Homography** comes from Supabase `court_calibrations` for the clip's `source_id` (same row as `data_labeling/push_court_calibration.py`).

**`pose_side_by_side_video.py`** — one command per DB clip: resolves `clips` row + `court_calibrations`, downloads the MP4 from S3 (needs `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`), runs YOLO + top-down, writes a side-by-side MP4 (H.264 via `ffmpeg` when available).

```bash
python cv-pipeline/pose-detection/pose_side_by_side_video.py --clip-id 42 --fps 2
```

Optional: `-o path/out.mp4`, `--panel-h`, `--weights`, `--no-h264-transcode`, etc. Defaults load `.env` from repo root (`SUPABASE_*`, `AWS_*`).

**Visual QA of calibrations** (still frame | top-down, no video): [`cv-pipeline/calibration/review_court_calibrations_db.py`](../calibration/review_court_calibrations_db.py).

**`fetch_s3_clip.py`** — manual S3 download into `media/` if you only need the file.

See root `requirements.txt` (OpenCV, Ultralytics/Torch, boto3, supabase, dotenv; optional `ffmpeg`).
