# Court calibration (homography)

This directory holds the **shared geometry and homography code** used when a still frame is labeled with court keypoints and we need a world→image transform for top-down warps and downstream pose experiments.

**Canonical path into production data:** `data_labeling/push_court_calibration.py` imports `court_homography.fit_calibration_record_for_db`, fits from the Label Studio export (or normalized payloads), and upserts `court_calibrations` in Supabase. The fitter and FIVB point table live **here**; the push script wires env, S3 metadata, and the DB row.

**Library-style modules**

- `court_homography.py` — planar correspondences from `fivb_court_geometry.txt`, RANSAC homography, `warp_topdown`, overlays, optional CLI → `out/homography.npz` + `topdown.png` for local work.
- `homography_io.py` — load the same `(H, world bounds, canvas size)` tuple from **`homography.npz`** or a **`court_calibrations`** dict (used by pose scripts and review tooling).
- `image_io.py` — reference still from local path or S3 (HTTPS / boto3).

**Scripts**

- `review_court_calibrations_db.py` — visual QA of rows already in Supabase (camera | top-down); see `cv-pipeline/pose-detection/README.md`.
- `court_homography_interactive.py` — OpenCV explorer on an export or npz (click camera → world).
- `court_homography.py` (as `__main__`) — one-off fit + npz from an export.

**Docs / schema:** `docs/schema.md` (`court_calibrations`), design notes in `docs/annotation_process/court_calibration_supabase.md`.

**`out/`** — default drop for npz/PNG experiments; gitignored. Not required for DB-backed workflows if you only push and review from Supabase.
