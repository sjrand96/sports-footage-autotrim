# Sports Footage Autotrim — Architecture

## Purpose

A shared system for downloading volleyball footage from YouTube, segmenting it into uniform clips, labeling **Playing** segments in Label Studio (unlabeled timeline = **downtime**), and tracking everything in a queryable database.

Designed for a small team of technical collaborators (CS master's students) sharing AWS and Supabase credentials, but each running Label Studio locally.

## System components

- **Local ingest script** (Python). Runs on each collaborator's machine. Handles download, segmentation, S3 upload, and Supabase metadata insert (`ingest_youtube_source.py`).
- **AWS S3** — `sports-footage-autotrim-bucket` (us-west-2). Public-read. Stores source videos and clips.
- **Supabase Postgres** — project URL `https://jlcgaxesuwcehqyfbbur.supabase.co`. Tracks all metadata.
- **Label Studio (per-person, local)**. Each collaborator runs their own Label Studio instance on `localhost:8080`. They share AWS credentials to read from the same S3 bucket but have independent task lists and annotation history.
- **Timeline export script** (Python). Imports Label Studio **timeline** JSON exports into Supabase (`push_timeline_annotation_export.py`).
- **Coordination spreadsheet** (Google Sheet). Lightweight ledger of which source videos are claimed, in progress, and done.

## High-level data flow

```
YouTube URL
    │
    ▼
[ingest_youtube_source.py] ── yt-dlp ──▶ source.mp4 (local temp)
    │                            │
    │                            ▼
    │                      ffmpeg segment
    │                            │
    │                            ▼
    │                      clips/ (local temp)
    │                            │
    │                            ▼
    │                       S3 upload
    │                            │
    └────── insert ─────▶  Supabase (source_videos, clips)

[Coordination Sheet] ◀── manual ──── new row added when video uploaded
       │
       │ collaborator claims a row
       ▼
[Local Label Studio (per person)]
       │  configures S3 source storage with prefix matching claimed source_id
       │  syncs, annotates
       ▼
[push_timeline_annotation_export.py](../../data_labeling/push_timeline_annotation_export.py)
       │  reads Label Studio JSON export
       │  resolves S3 URL → clip_id via Supabase lookup
       │  stamps with ANNOTATOR_NAME from env
       ▼
   Supabase (annotations)
```

## Identifiers and naming

The YouTube video ID is the canonical identifier for a source video. It is already unique, stable, and URL-safe.

- **Source ID**: 11-character YouTube video ID, e.g. `dQw4w9WgXcQ`
- **Display name**: human-readable label, lives in DB and coordination sheet, e.g. `"USCG vs Stanford 4/15"`
- **Clip filenames**: `{source_id}_{NNN}.mp4`, zero-padded 3-digit index starting at 001
- **S3 keys**: `clips/{source_id}/{source_id}_{NNN}.mp4`

Example: a 17-minute video with YouTube ID `abc123XYZ45` produces keys:
- `s3://sports-footage-autotrim-bucket/sources/abc123XYZ45.mp4`
- `s3://sports-footage-autotrim-bucket/clips/abc123XYZ45/abc123XYZ45_001.mp4`
- ... through `_017.mp4`

Public URLs follow the pattern:
```
https://sports-footage-autotrim-bucket.s3.us-west-2.amazonaws.com/clips/abc123XYZ45/abc123XYZ45_001.mp4
```

## Clip parameters

All clips are normalized to identical encoding properties so Label Studio's frame counting works consistently:

- Container: MP4
- Video codec: H.264 — `libx264` (default / `--software-encode`) or `h264_videotoolbox` on Apple Silicon when available; high profile, level 4.0, yuv420p for software path
- Audio codec: AAC, 128 kbps
- Frame rate: constant 30 fps
- Segment length: 60 seconds (last clip may be shorter)

## Database tables

Three tables:

- **`source_videos`** — one row per ingested YouTube video (`id` = video id).
- **`clips`** — one row per 1-minute segment (`source_id`, `clip_index`), S3 keys, optional thumbnail key, timing fields.
- **`annotations`** — rows written by `data_labeling/push_timeline_annotation_export.py` (`clip_id` FK, Label Studio ids, `annotator`, `lead_time_sec`, `payload` JSON).

**Executable DDL** (creates tables and indexes; idempotent `IF NOT EXISTS`): use [schema.md](../schema.md) — paste into the Supabase SQL Editor when bootstrapping a new project. For collaborators on an existing project, tables are already present; this file is the reference when migrations are needed.

Semantics:

- `source_videos.id` is the YouTube ID (text), not a serial — straightforward joins from URLs and paths.
- `annotations.payload` stores a JSON object with `label_studio_task` and `label_studio_annotation` (full export context for the pushed row). Evolves without schema migrations when the LS export shape changes.
- `annotations.annotator` is set from each collaborator’s `ANNOTATOR_NAME` in `.env` — local Label Studio’s internal user id is not distinctive across machines.
- Multiple rows per `clip_id` are normal (different annotators or re-pushes). `data_labeling/push_timeline_annotation_export.py` skips inserting a duplicate for the same `(clip_id, label_studio_task_id, annotator)`; otherwise each run can add new rows.

## S3 layout

```
s3://sports-footage-autotrim-bucket/
  sources/
    {source_id}.mp4              # original full-length download
  clips/
    {source_id}/
      {source_id}_001.mp4
      {source_id}_001.jpg          # middle-frame thumbnail (same stem as clip)
      {source_id}_002.mp4
      ...
```

Bucket is public-read. Footage is non-sensitive (already public on YouTube). Label Studio reads via direct S3 URLs.

## Reprocessing policy

**Source videos and clips are mutable, but reprocessing overwrites in place.** If a video is reprocessed:

- Same `source_id` (YouTube ID is stable)
- Same S3 keys (overwritten)
- Same `clips` rows (updated, not duplicated — handled via `INSERT ... ON CONFLICT (source_id, clip_index) DO UPDATE`)
- Existing annotations remain attached to the same clip rows

Trade-off accepted: annotations made against a previous version of a clip will appear to apply to the new version. In practice, reprocessing should be rare; when it happens, collaborators are responsible for noting whether existing annotations need re-review (use the `Notes` column in the coordination sheet).

## Coordination model

Because each collaborator runs their own Label Studio instance, work assignment is coordinated externally via a Google Sheet rather than through Label Studio's task queue.

### Coordination spreadsheet structure

One row per source video:

| Source ID | Display name | Status | Assignee | Started | Finished | Notes |
|---|---|---|---|---|---|---|
| dQw4w9WgXcQ | USCG vs Stanford 4/15 | Annotating | alice | 2026-05-03 | | 17 clips |
| abc123XYZ45 | UCLA vs USC 4/22 | Done | bob | 2026-04-28 | 2026-05-02 | All clips pushed |
| def456PQR78 | Pepperdine 4/29 | Available | | | | Clips uploaded |

Status values: `Available`, `Claimed`, `Annotating`, `Done`, `Issues`.

### Lifecycle of a source video

1. Someone runs `data_labeling/ingest_youtube_source.py` to download, segment, and upload a new video. They add a row to the sheet with status `Available`.
2. An annotator picks an `Available` row, changes status to `Claimed` with their name in `Assignee`, sets `Started` date.
3. They configure their local Label Studio's S3 source storage prefix to `clips/{source_id}/`, click Sync. Status moves to `Annotating`.
4. They annotate all clips for that source (Playing-only convention; see [workflow_overview.md](workflow_overview.md)).
5. They export JSON and run `python data_labeling/push_timeline_annotation_export.py export.json`. Status moves to `Done`, `Finished` date is filled in.

### Conflict avoidance

Two people should not claim the same source video simultaneously. Status transitions are advisory but the sheet is the source of truth. If two people accidentally annotate the same clips, both sets of annotations are preserved in Supabase (different `annotator` values, both attached to the same `clip_id`) — no data loss, just some duplicated work.

## Workflows

### W1: Add a new video

Run on a collaborator's local machine from the repo root (after `pip install -e .` so `boto3` / `supabase` are available):

```
python data_labeling/ingest_youtube_source.py <youtube_url> --display-name "USCG vs Stanford 4/15"
```

Optional flags: `--force` (redo download, clips, S3, DB), `--software-encode` (force `libx264` instead of Apple `h264_videotoolbox` when available).

The script:

1. Extracts the YouTube video ID from the URL
2. If `source_videos.id` already exists in Supabase, logs a reminder; idempotent steps still skip unchanged work unless you pass `--force`
3. Downloads with yt-dlp (1080p60 H.264 + m4a audio, merged to MP4)
4. Segments + re-encodes with ffmpeg to 60s / 30fps clips in a local temp directory
5. Uploads source video and clips to S3 (skipping any that already exist with matching size, for idempotency)
6. Upserts `source_videos` row and `clips` rows in Supabase
7. Cleans up local temp files
8. Prints a reminder to add the source to the coordination sheet

The script is **idempotent**: running it twice in a row with the same URL is a no-op on the second run (modulo timestamps).

### W2: Claim and label

Each collaborator does this in their own local Label Studio:

1. Find an `Available` row in the coordination sheet, claim it
2. In local Label Studio: Project → Settings → Cloud Storage → Edit S3 source storage → set Bucket Prefix to `clips/{source_id}/`
3. Click Sync
4. Tasks for that source's clips appear
5. Annotate all tasks (convention: **Playing** regions only; unlabeled = downtime — see [workflow_overview.md](workflow_overview.md))
6. Export JSON from Label Studio (Data Manager → Export → **JSON**), then run `python data_labeling/push_timeline_annotation_export.py` (below)
7. Mark `Done` in the sheet

### W3: Push Label Studio export to Supabase

Run after you have **submitted** annotations and exported **JSON** (Label Studio common format) from your local Label Studio project.

```bash
python data_labeling/push_timeline_annotation_export.py /path/to/project-N-at-....json
python data_labeling/push_timeline_annotation_export.py /path/to/export.json --dry-run   # no DB writes
```

**`.env` required for this script:** `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `ANNOTATOR_NAME` (your stable handle — same idea as for ingest’s `downloaded_by`).

**What it does:**

1. Reads the export file (a JSON **array of tasks**).
2. For each task that has at least one **non-cancelled** submitted annotation, picks the **latest** annotation by `updated_at` (then `created_at`).
3. Parses `data.video` (`s3://…/clips/{source_id}/{source_id}_NNN.mp4` or equivalent HTTPS path) to get `source_id` and `clip_index`. Non-`.mp4` objects (e.g. thumbnails) are skipped.
4. Looks up `clips` by `(source_id, clip_index)`.
5. **Idempotency:** if a row already exists with the same `(clip_id, label_studio_task_id, annotator)`, that task is **skipped** (no duplicate row).
6. Otherwise **inserts** one `annotations` row: `annotator = ANNOTATOR_NAME`, `lead_time_sec` from Label Studio, `payload` = `{ "label_studio_task": <full task>, "label_studio_annotation": <chosen annotation> }`.

Re-exporting and re-running after you **edit** the same task in Label Studio still **skips** that task until you delete the existing Supabase row or change annotator — by design for strict one row per task per annotator. To replace labels, delete the old annotation row in Supabase (or add a future “force” flag).

Concurrent runs by different collaborators are safe: each uses their own `ANNOTATOR_NAME`, so distinct rows for the same clip/task are allowed when intended.

## Credentials

A shared `.env` template (gitignored, distributed via password manager) plus per-person additions:

```
# Shared — everyone identical
SUPABASE_URL=https://jlcgaxesuwcehqyfbbur.supabase.co
SUPABASE_SERVICE_KEY=<from Supabase project settings>
AWS_ACCESS_KEY_ID=<from IAM user volleyball-pipeline>
AWS_SECRET_ACCESS_KEY=<from IAM user>
AWS_REGION=us-west-2
S3_BUCKET=sports-footage-autotrim-bucket

# Personal — each collaborator fills in their own
ANNOTATOR_NAME=<your handle, e.g. spencer>
```

- AWS: one IAM user (`volleyball-pipeline`) with read/write to the bucket. Same credentials used by everyone's ingest script and everyone's local Label Studio S3 connection.
- Supabase: service role key. Bypasses RLS, full table access. Acceptable because the `.env` is shared only with trusted collaborators.
- Label Studio: each person runs their own instance and uses the UI; **`data_labeling/ingest_youtube_source.py` and `data_labeling/push_timeline_annotation_export.py` do not read any Label Studio URL or API key** — only the JSON export file for push. If you later add a script that calls the Label Studio REST API, you would introduce those variables then.

If a collaborator leaves: rotate the IAM access key and Supabase service key, redistribute shared `.env` values. Their local Label Studio instance has no shared state to revoke.

## Repository layout

```
sports-footage-autotrim/
├── README.md
├── docs/
│   ├── annotation_process/
│   │   ├── README.md                        # index of labeling docs
│   │   ├── workflow_overview.md             # setup + per-video steps + diagram
│   │   ├── label-studio-setup.md            # local LS install, S3, template XML
│   │   ├── court_calibration_supabase.md  # court homography table + SQL
│   │   └── annotation_schema_and_systems.md # this document
│   └── schema.md                            # executable SQL only (DDL)
├── pyproject.toml                           # pipeline deps (boto3, supabase, …)
├── requirements.txt                         # optional notebook / CV stack
├── src/
│   └── db.py                                # Supabase helpers (ingest + timeline push)
├── data_labeling/
│   ├── README.md                            # scripts hub → docs
│   ├── ingest_youtube_source.py             # W1: YouTube → segments, S3, Supabase
│   ├── push_timeline_annotation_export.py   # W3: LS timeline export → annotations
│   └── court_keypoints.py                   # court LS export → normalized JSON
└── ...
```

Local ingest uses `./workdir/{youtube_id}/` (download + `clips/`); that directory is removed after a successful run.

## Out of scope (for now)

- Shared Label Studio instance with unified task queue (deferred — adds operational overhead, current scale doesn't need it)
- Per-user authentication / RLS policies in Supabase
- Webhook-based annotation sync (manual export script is fine)
- Read-only dashboards (Supabase web UI sufficient for ad-hoc queries)
- Backfilling existing clips (none exist)
- Annotation versioning beyond append-only rows

## Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Database | Supabase Postgres (us-west-2) | Hosted, free tier, multi-writer, web UI |
| Object storage | AWS S3, public-read, us-west-2 | Standard, integrates with Label Studio |
| Source identifier | YouTube video ID | Unique, stable, URL-safe, no sanitization |
| Reprocessing | Overwrite in place | Simpler model; annotations stay attached to logical clip |
| Auth | Shared service credentials in `.env` | Small trusted team, simplest path, easy to rotate |
| Annotation sync | Manual `data_labeling/push_timeline_annotation_export.py` | Avoids webhook infrastructure |
| Label Studio deployment | Per-person local | Avoids EC2 setup overhead; team is small enough that coordination via spreadsheet works |
| Work coordination | Google Sheet | Lightweight, human-readable, no infra |
| Annotator identity | `ANNOTATOR_NAME` env var stamped on each row | Local LS gives everyone annotator ID `1`; can't distinguish without an explicit name |
| Clip length | 60 seconds | Convenient unit for labeling, matches typical rally cadence |
| Frame rate | 30 fps CFR | Required for Label Studio frame-count accuracy |
