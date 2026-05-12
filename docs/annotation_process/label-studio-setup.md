# Label Studio Setup Guide

End-to-end setup for the local Label Studio instance each collaborator runs on their own machine. Plan on ~20 minutes.

## Prerequisites

You should already have:

- The shared `.env` file populated with AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Access to the project's S3 bucket: `sports-footage-autotrim-bucket` in `us-west-2`
- Python 3.10+ installed

## 1. Install and start Label Studio

Use a dedicated venv to avoid dependency conflicts with the project venv.

```
python3 -m venv ~/.label-studio-env
source ~/.label-studio-env/bin/activate
pip install label-studio
```

Start it:

```
label-studio start
```

A browser tab opens at `http://localhost:8080`. First time only:

- Sign up with email + password (the email doesn't need to be real, but make it memorable)
- Save credentials in your personal password manager

To stop Label Studio later: `Ctrl+C` in the terminal. To restart: re-activate the venv and run `label-studio start` again. Your data persists at `~/.local/share/label-studio/`.

## 2. Create the labeling project

- Click **Create**
- **Project Name**: `Volleyball Action Labels`
- Skip **Data Import** for now
- Open **Labeling Setup** → click **Custom template**
- Paste this configuration exactly:

```xml
<View>
  <Header value="Mark Playing segments. Leave Downtime unlabeled."/>
  <Video name="video" value="$video" frameRate="30" timelineHeight="100"/>
  <TimelineLabels name="videoLabels" toName="video">
    <Label value="Playing" background="#1BB500"/>
  </TimelineLabels>
</View>
```

- Click **Save**

### Why this config

- **One label, not two.** We only mark `Playing` segments. Anything unlabeled is implicitly Downtime. This avoids off-by-one frame issues at boundaries and roughly halves labeling effort.
- **`frameRate="30"`** matches what the ingest script produces. If this doesn't match the actual video framerate, the timeline math will be wrong and your annotations will land on the wrong frames.
- **`timelineHeight="100"`** makes the timeline panel taller (default is 64). Much more comfortable to drag on.
- **`name="videoLabels"`** is the field name that will appear in your exported JSON. Keep this consistent across collaborators so the push script works.

## 2b. Optional: court keypoints project (homography)

This is **separate** from **Volleyball Action Labels**. Timeline tasks use `$video` and are synced from `.mp4` objects; keypoint tasks use `$image`. You reuse the **same bucket and `clips/{source_id}/` prefix** as the video project—`data_labeling/ingest_youtube_source.py` uploads a `.jpg` next to each clip (middle-frame still, same resolution as the clip). Point this project’s S3 storage at those JPEGs with an image-only filename filter so tasks sync the same way as clips. Use this when you need a reference frame for court calibration (see Phase 1 in [cv-pipeline/cv_pipeline.md](../../cv-pipeline/cv_pipeline.md)).

Exports from this project are **not** ingested by `data_labeling/push_timeline_annotation.py` (that script only understands clip `.mp4` URLs). Use `data_labeling/court_keypoints.py` to parse an export into stable JSON payloads, then fit homography with `cv-pipeline/calibration/court_homography.py` (writes `cv-pipeline/calibration/out/homography.npz`; see Phase 1 in [cv_pipeline.md](../../cv-pipeline/cv_pipeline.md)). Target Supabase contract and DDL: [court_calibration_supabase.md](court_calibration_supabase.md).

### Create the project

- Click **Create**
- **Project Name**: e.g. `Volleyball Court Keypoints`
- Skip **Cloud Storage** until the next subsection (unless you combine setup order however you prefer)
- **Labeling Setup** → **Custom template** → paste:

```xml
<View>
  <Header value="Still is middle-of-clip from ingest — use the synced task where the court is clearest. One point per label (skip off-screen/occluded). Line intersections; net posts at floor and top."/>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="kp" toName="img" strokeWidth="3">
    <Label value="far_baseline_left" background="#E60026"/>
    <Label value="far_baseline_right" background="#FF7F00"/>
    <Label value="near_baseline_left" background="#FFD700"/>
    <Label value="near_baseline_right" background="#A0522D"/>
    <Label value="far_attack_left" background="#00C04B"/>
    <Label value="far_attack_right" background="#006837"/>
    <Label value="near_attack_left" background="#00CED1"/>
    <Label value="near_attack_right" background="#1E3A8A"/>
    <Label value="centerline_left" background="#8A2BE2"/>
    <Label value="centerline_right" background="#FF1493"/>
    <Label value="net_post_base_left" background="#000000"/>
    <Label value="net_post_base_right" background="#808080"/>
    <Label value="net_post_top_left" background="#4B0082"/>
    <Label value="net_post_top_right" background="#FFFFFF"/>
  </KeyPointLabels>
</View>
```

- **Save**

### Sync reference frames from S3 (recommended)

The ingest script already writes `{source_id}_{index}.jpg` alongside each `{source_id}_{index}.mp4` under `clips/{source_id}/`.

**Volleyball Court Keypoints** → **Settings → Cloud Storage → Add Source Storage → Amazon S3**, then:

**Step 1 — Configure Connection** (same bucket and auth as clip labeling; see [§3 → Values at a glance](#values-at-a-glance)):

| Field | Value |
|---|---|
| **Storage Title** | any (e.g. `Court thumbnails`, `Volleyball Db`) |
| **Bucket Name** | `sports-footage-autotrim-bucket` |
| **Region Name** | `us-west-2` |
| **S3 Endpoint** | default — leave blank, or `https://s3.amazonaws.com` |
| **Access Key ID** / **Secret Access Key** | from `.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) |
| **Session Token** | leave blank |
| **Use pre-signed URLs** | **On** |
| **Expire pre-signed URLs (minutes)** | `15` |

**Step 2 — Import Settings & Preview**

| Field | Value |
|---|---|
| **Bucket Prefix** | `clips/{source_id}/` (same rule as clips: real YouTube id, e.g. `clips/jZ18INu4LQc/`) |
| **Import Method** | **Files - Automatically creates a task for each storage object** |
| **File Name Filter** | `.*\.jpg$` (only JPEGs—**not** `.mp4`) |
| **Scan all sub-folders** | **enabled** |

**Step 3** → **Save**, then **Sync Storage** on the Cloud Storage page.

You get **one labeling task per clip thumbnail**—same ordering as clips. Prefer the task whose middle frame shows the clearest court (or annotate multiple clips if you need several homographies).

If **Data Manager** previews look wrong or keypoints UI won’t attach to the canvas, open a task’s raw data and confirm the field is **`image`** (the template binds `$image` to that).

### Fallback: manual still

If thumbnails are missing or you need a timestamp that is not mid-clip: extract locally and **Import → upload**.

`ffmpeg -ss 5 -i path/to/{source_id}_000.mp4 -frames:v 1 -q:v 2 court_ref.jpg`

Adjust `-ss` as needed.

### Annotate

Open a task → select each label in the sidebar → click the matching spot → **Submit** when all **visible** points are placed (skip labels whose features are out of frame).

### What each label means

Naming is **camera-relative**: “far” is the far end of the court from the camera, “near” is the close end. Left/right are as seen on screen.

| Label | Court feature |
|------|----------------|
| `far_baseline_left`, `far_baseline_right` | Far baseline where it meets the sidelines |
| `near_baseline_left`, `near_baseline_right` | Near baseline where it meets the sidelines |
| `far_attack_left`, `far_attack_right` | Far attack line at the sidelines |
| `near_attack_left`, `near_attack_right` | Near attack line at the sidelines |
| `centerline_left`, `centerline_right` | Center line (under the net) at the sidelines |
| `net_post_base_left`, `net_post_base_right` | Net post where it meets the floor |
| `net_post_top_left`, `net_post_top_right` | Top of the net post (or net–post junction), same left/right as the bases |

Canonical meters-based coordinates for these points are documented in [cv-pipeline/cv_pipeline.md](../../cv-pipeline/cv_pipeline.md) (Phase 1a). Match the same corner/edge convention the team uses there.

## 3. Connect S3 source storage

Now point Label Studio at the shared S3 bucket. This is a multi-step wizard.

### Values at a glance

Use these everywhere Label Studio asks for bucket settings (clip project, court-thumbnail project, tests):

| | |
|---|---|
| **Bucket Name** | `sports-footage-autotrim-bucket` |
| **Region Name** | `us-west-2` |
| **S3 Endpoint** | **Default** — leave blank; the UI may show `https://s3.amazonaws.com (default)`. If you must type an endpoint, use `https://s3.amazonaws.com` |
| **Access Key ID** | paste from repo `.env`: `AWS_ACCESS_KEY_ID` |
| **Secret Access Key** | paste from repo `.env`: `AWS_SECRET_ACCESS_KEY` |
| **Session Token** | leave **empty** |
| **Use pre-signed URLs** | **On** |
| **Expire pre-signed URLs (minutes)** | `15` |
| **Bucket Prefix** (Step 2) | `clips/{source_id}/` with the real 11-character YouTube id, e.g. `clips/jZ18INu4LQc/` |
| **Import Method** (Step 2) | **Files - Automatically creates a task for each storage object** |
| **Scan all sub-folders** (Step 2) | **enabled** |
| **File Name Filter** (Step 2) | **Video project:** `.*\.mp4$` — **Court keypoints project:** `.*\.jpg$` (see [§2b](#2b-optional-court-keypoints-project-homography)) |

**Storage Title** is only a label in your UI (e.g. `Volleyball Clips`, `Court thumbnails`, `Volleyball Db`).

### 3a. Step 1 — Configure Connection

- **Project → Settings → Cloud Storage → Add Source Storage**
- Select **Amazon S3** as the storage type
- Fill in **Step 1** of the wizard to match [Values at a glance](#values-at-a-glance) (same fields as **Edit Source Storage**: bucket, region, endpoint, keys, pre-signed **On**, expire **15**)

About pre-signed URLs: even though the bucket is publicly readable, we leave this **on** because it's the default and avoids any browser-cache or CORS edge cases. Each time you open a clip to label, Label Studio generates a short-lived signed URL on the fly. It works the same way for the labeler — clips just stream from S3.

- Click **Test Connection** — should show success
- Click **Next**

### 3b. Step 2 — Import Settings & Preview

- **Bucket Prefix**: set this to **`clips/{source_id}/`** using the **real** YouTube source id for the video you are labeling (11 characters, e.g. `jZ18INu4LQc`). Example: `clips/jZ18INu4LQc/`.

  **Do not** use a broad prefix like `clips/` alone. Scoping to one `source_id` is how we avoid pulling everyone’s clips into one project and how we align with who is working on which source. When you switch to a different video, edit this storage entry and change the prefix to the new `clips/{source_id}/`, then sync again.

- **Import Method**: select **Files - Automatically creates a task for each storage object**

  This is the right choice for our setup: each `.mp4` in S3 becomes one labeling task. The other option ("JSON files describing tasks") is for cases where you've prepared task JSON files separately, which we don't.

- **File Name Filter**: `.*\.mp4$`

  Critical for our setup. The ingest script uploads both `.mp4` clips and `.jpg` thumbnails to the same prefix. This regex tells Label Studio to ignore the thumbnails so they don't become labeling tasks.

  The default `.*\.(mp4|avi|mov|wmv|webm)$` also works, but I prefer the narrower one — we know everything is mp4.

- **Scan all sub-folders**: ✅ **enabled**

  Required because clips live under `clips/{source_id}/`, not directly in `clips/`.

- Click **Load Preview** — should list the `.mp4` files for that source (or "No files found" if ingest has not been run for that id yet)
- Click **Next**

### 3c. Step 3 — Review & Confirm

- Confirm the settings look right
- Click **Save**, then **Sync Storage** on the Cloud Storage page when you are ready to import tasks

## 4. Set your annotator name

In your local `.env` (repo root, same file the pipeline uses):

```
ANNOTATOR_NAME=spencer
```

Use a short identifier — your first name in lowercase is fine. **Once set, never change it.** This gets stamped on every annotation row you push to Supabase, and it's the only way we tell whose work an annotation represents.

## 5. Verify everything works

You should have at least:

```
# Shared (example — use values from the team template)
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
S3_BUCKET=sports-footage-autotrim-bucket

# Personal
ANNOTATOR_NAME=...
```

Checks:

- Local Label Studio is reachable: open `http://localhost:8080` and sign in
- After sync, **Data Manager** shows one task per clip `.mp4` for the `source_id` you used in the prefix

## 6. Daily labeling workflow

### Claim a source

Mark your claim in the **Google Sheet** (status / assignee / dates as your team agrees).

### Point Label Studio at that source's clips

- **Project → Settings → Cloud Storage** → edit your S3 source
- Set **Bucket Prefix** to `clips/{source_id}/` with the **actual** YouTube id for that row (same idea as in setup)
- Save → **Sync Storage**
- Tasks for that source’s clips appear in **Data Manager**

Update the sheet again when you move to **annotating** (if your team tracks that).

### Annotate

- Open a task
- Click the green **Playing** label to activate it
- On the timeline, **click and drag** across each rally / playing segment
- Don't worry about exact frame boundaries — generally don't waste time tweaking edges by 1–2 frames
- Press **Submit** (bottom-right) when done with a clip
- Move to the next task

### Push annotations to the database

After you **submit** tasks and **Export → JSON** from Label Studio, from the repo:

```
cd path/to/sports-footage-autotrim
source .venv/bin/activate
python data_labeling/push_timeline_annotation.py /path/to/your-export.json
```

Optional: `--dry-run` first. See [workflow_overview.md](workflow_overview.md) and W3 in [annotation_schema_and_systems.md](annotation_schema_and_systems.md).

When the source is fully labeled and pushed, mark it **Done** (or equivalent) in the **Google Sheet**.
