-- Sports Footage Autotrim — Database Schema
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
  downloaded_by   text                          -- whoever ran the prep script
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
-- Verify
-- ============================================================
-- select table_name from information_schema.tables
--   where table_schema = 'public'
--   order by table_name;
-- Should return: annotations, clips, source_videos
