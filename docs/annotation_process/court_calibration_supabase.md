# Court calibration (homography) — Supabase design

Reference for teammates and implementation. One **static camera per source video** (`source_id` = YouTube id): one calibration row per source, **upsert / overwrite** when redoing.

## Label Studio workflow

Calibration uses **still images** (KeypointLabels on `data.image`), not full video scrubbing.

**Recommended (simple):** one Label Studio task per `source_id`. `data.image` points at a single S3 HTTPS URL for a reference still (typically a clip **thumbnail** from ingest: `clips/{source_id}/{source_id}_NNN.jpg`). The annotator labels visible court keypoints on that frame. If the frame is unusable (occlusion, bad angle), swap the image URL to another thumbnail or exported still and re-label — then re-import.

**Optional (multiple candidates):** several tasks per source, each with a different `data.image`. The annotator completes keypoints on the clearest task only. The import script ingests **the completed task** for that `source_id` (enforce one winning row in DB via upsert on `source_id`).

**Resolution:** keypoints are stored in **pixel coordinates of the reference image** (`ref_image_width_px` / `ref_image_height_px` must match the image at `data.image`). No scaling to full clip resolution for now; later scaling should update image + points together.

## What lives in the database

| Area | Stored | Purpose |
|------|--------|--------|
| Contract | Reference image S3 location + dimensions | Reproducibility; same pixel space as clicks |
| Contract | `keypoints` jsonb | Human labels (`label`, `x_px`, `y_px`) |
| Contract | `homography_matrix` jsonb (3×3) + world bounds + `pixels_per_metre` | Same information the CV stack today reads from `homography.npz` / `meta_json` (`H_world_to_pixel`, `world_bounds_xy`, ppm) — pipelines load without refitting |
| Audit | Label Studio ids + `annotator` | Trace which export produced the row |
| Optional | `raw_label_studio_export` jsonb | Small full task/annotation snapshot for debugging |

Fit runs **at import time** (after export): read keypoints + geometry from code, compute `H`, write row. Pipelines **read** stored `H` and bounds; refit only when you change the fitter and re-import.

## Table: `court_calibrations`

- **Primary key:** `source_id` — one row per source; overwrite = `upsert` on conflict.
- **Foreign key:** `source_id` → `source_videos(id)` ON DELETE CASCADE (see `docs/schema.md`).

### `keypoints` jsonb shape

Array of objects, sorted by `label` in application if desired:

```json
[
  {"label": "example_key", "x_px": 123.45, "y_px": 678.9}
]
```

Labels must match the planar court geometry used by the homography fitter (same convention as `cv-pipeline/calibration/court_homography.py`).

### `homography_matrix` jsonb shape

3×3 row-major array of floats, **world → pixel** (same tensor name as `H_world_to_pixel` in `homography.npz`).

### World bounds

Match `meta_json.world_bounds_xy` from the calibration fitter: axis-aligned court window in **meters** used for top-down rendering (`wx_min`, `wx_max`, `wy_min`, `wy_max`). `pixels_per_metre` matches the fitter default (e.g. 45) unless changed at import.

## SQL (Supabase SQL editor)

Run on the project after `source_videos` exists (`docs/schema.md`).

```sql
-- Court homography: one row per source video (overwrite via upsert on source_id).

create table if not exists court_calibrations (
  source_id text primary key references source_videos(id) on delete cascade,

  ref_image_s3_bucket text not null,
  ref_image_s3_key text not null,
  ref_image_width_px int not null check (ref_image_width_px > 0),
  ref_image_height_px int not null check (ref_image_height_px > 0),
  ref_clip_index int,

  keypoints jsonb not null,
  homography_matrix jsonb not null,
  world_wx_min numeric not null,
  world_wx_max numeric not null,
  world_wy_min numeric not null,
  world_wy_max numeric not null,
  pixels_per_metre numeric not null default 45,

  label_studio_task_id bigint,
  label_studio_annotation_id bigint,
  label_studio_project_id bigint,
  annotator text not null,

  raw_label_studio_export jsonb,
  schema_version smallint not null default 1,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists court_calibrations_updated_at_idx
  on court_calibrations (updated_at desc);
```

**`updated_at`:** bump in application on each successful upsert (`updated_at = now()`), or add a `BEFORE UPDATE` trigger if you prefer it automatic.

## Related code (today)

- Normalized keypoint payloads from exports: `data_labeling/court_keypoints.py` (`calibration_record_to_json`, `kind: court_keypoints_label_studio`).
- Fit + npz layout: `cv-pipeline/calibration/court_homography.py`, `_load_homography_npz` in `cv-pipeline/pose-detection/pose_side_by_side_video.py`.
