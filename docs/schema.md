-- Sports Footage Autotrim — Database Schema (canonical DDL)
--
-- Column semantics, S3 layout, and how ingest_youtube_source / push_timeline_annotation / push_court_calibration use these tables:
--   docs/annotation_process/annotation_schema_and_systems.md
--
-- Court calibration (homography) column contracts and workflow:
--   docs/annotation_process/court_calibration_supabase.md
--
-- Run this in the Supabase SQL Editor on a fresh project to create all tables.
-- Idempotent: safe to re-run; uses IF NOT EXISTS where possible.
--
-- Notes:
--   - All timestamps are timestamptz (UTC internally; displayed in your
--     browser timezone in the Supabase UI).
--   - This schema relies on the Supabase service role key for access. The
--     anon key will see empty results because RLS is enabled by default on
--     new Supabase tables. Add policies if you ever need anon access.

-- ============================================================
-- source_videos
-- One row per YouTube video that has been ingested.
-- ============================================================

create table if not exists source_videos (
  id              text primary key,             -- YouTube video ID (e.g. 'dQw4w9WgXcQ')
  url             text not null,                -- full YouTube URL
  display_name    text,                         -- human-readable, e.g. 'USCG vs Stanford 4/15'
  duration_sec    numeric,                      -- total length of the source video
  fps_original    numeric,                      -- fps of the original (before re-encode)
  downloaded_at   timestamptz not null default now(),
  downloaded_by   text                          -- whoever ran ingest_youtube_source.py
);

-- ============================================================
-- clips
-- One row per 60-second clip cut from a source video.
-- s3_key is unique to prevent two rows pointing at the same object.
-- ============================================================

create table if not exists clips (
  id                  bigserial primary key,
  source_id           text not null references source_videos(id) on delete cascade,
  clip_index          int not null,             -- 1-based index within the source
  filename            text not null,            -- e.g. 'dQw4w9WgXcQ_001.mp4'
  s3_bucket           text not null,
  s3_key              text not null unique,     -- e.g. 'clips/dQw4w9WgXcQ/dQw4w9WgXcQ_001.mp4'
  thumbnail_s3_key    text,                     -- e.g. 'clips/dQw4w9WgXcQ/dQw4w9WgXcQ_001.jpg', nullable
  start_sec           numeric not null,         -- offset in source video where clip begins
  end_sec             numeric not null,
  duration_sec        numeric not null,
  uploaded_at         timestamptz not null default now(),
  unique (source_id, clip_index)
);

create index if not exists clips_source_id_idx on clips (source_id);

-- ============================================================
-- annotations
-- Append-only log of annotation exports from any collaborator's local
-- Label Studio. Multiple rows per clip are expected (different annotators,
-- repeated sessions). Use MAX(exported_at) per (clip_id, annotator) to
-- get the latest version of someone's labels for a given clip.
-- ============================================================

create table if not exists annotations (
  id                       bigserial primary key,
  clip_id                  bigint not null references clips(id) on delete cascade,
  label_studio_task_id     bigint,
  label_studio_project_id  bigint,
  annotator                text not null,           -- ANNOTATOR_NAME from each person's .env
  lead_time_sec            numeric,
  exported_at              timestamptz not null default now(),
  payload                  jsonb not null            -- raw Label Studio task JSON
);

create index if not exists annotations_clip_id_idx on annotations (clip_id);
create index if not exists annotations_annotator_idx on annotations (annotator);
create index if not exists annotations_clip_annotator_idx on annotations (clip_id, annotator);

-- ============================================================
-- court_calibrations
-- One row per source video: reference still + keypoints + fitted homography
-- (world→image) and court bounds. Written by push_court_calibration.py (upsert on source_id).
-- ============================================================

create table if not exists court_calibrations (
  source_id text primary key references source_videos(id) on delete cascade,

  ref_image_s3_bucket text not null,
  ref_image_s3_key text not null,
  ref_image_width_px int not null check (ref_image_width_px > 0),
  ref_image_height_px int not null check (ref_image_height_px > 0),
  ref_clip_index int,                              -- thumbnail clip index when path is clips/{id}/{id}_NNN.jpg

  keypoints jsonb not null,                        -- [{label, x_px, y_px}, ...] in ref image pixel space
  homography_matrix jsonb not null,                -- 3×3 world→pixel (same role as H_world_to_pixel in npz)
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

create index if not exists court_calibrations_updated_at_idx on court_calibrations (updated_at desc);

-- ============================================================
-- Verify
-- ============================================================
-- select table_name from information_schema.tables
--   where table_schema = 'public'
--   order by table_name;
-- Should return: annotations, clips, court_calibrations, source_videos
