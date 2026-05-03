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
- **`frameRate="30"`** matches what the prep script produces. If this doesn't match the actual video framerate, the timeline math will be wrong and your annotations will land on the wrong frames.
- **`timelineHeight="100"`** makes the timeline panel taller (default is 64). Much more comfortable to drag on.
- **`name="videoLabels"`** is the field name that will appear in your exported JSON. Keep this consistent across collaborators so the push script works.

## 3. Connect S3 source storage

Now point Label Studio at the shared S3 bucket. This is a multi-step wizard.

### 3a. Step 1 — Configure Connection

- **Project → Settings → Cloud Storage → Add Source Storage**
- Select **Amazon S3** as the storage type
- Fill in:

| Field | Value |
|---|---|
| **Storage Title** | `Volleyball Clips` (or any name — just for your reference) |
| **Bucket Name** | `sports-footage-autotrim-bucket` |
| **Region Name** | `us-west-2` |
| **S3 Endpoint** | leave blank (default) |
| **Access Key ID** | from your `.env` (`AWS_ACCESS_KEY_ID`) |
| **Secret Access Key** | from your `.env` (`AWS_SECRET_ACCESS_KEY`) |
| **Session Token** | leave blank |
| **Use pre-signed URLs** | ✅ **leave ON** (the default) |
| **Expire pre-signed URLs (minutes)** | `15` (default is fine) |

About pre-signed URLs: even though the bucket is publicly readable, we leave this **on** because it's the default and avoids any browser-cache or CORS edge cases. Each time you open a clip to label, Label Studio generates a short-lived signed URL on the fly. It works the same way for the labeler — clips just stream from S3.

- Click **Test Connection** — should show success
- Click **Next**

### 3b. Step 2 — Import Settings & Preview

- **Bucket Prefix**: set this to **`clips/{source_id}/`** using the **real** YouTube source id for the video you are labeling (11 characters, e.g. `jZ18INu4LQc`). Example: `clips/jZ18INu4LQc/`.

  **Do not** use a broad prefix like `clips/` alone. Scoping to one `source_id` is how we avoid pulling everyone’s clips into one project and how we align with who is working on which source. When you switch to a different video, edit this storage entry and change the prefix to the new `clips/{source_id}/`, then sync again.

- **Import Method**: select **Files - Automatically creates a task for each storage object**

  This is the right choice for our setup: each `.mp4` in S3 becomes one labeling task. The other option ("JSON files describing tasks") is for cases where you've prepared task JSON files separately, which we don't.

- **File Name Filter**: `.*\.mp4$`

  Critical for our setup. The prep script uploads both `.mp4` clips and `.jpg` thumbnails to the same prefix. This regex tells Label Studio to ignore the thumbnails so they don't become labeling tasks.

  The default `.*\.(mp4|avi|mov|wmv|webm)$` also works, but I prefer the narrower one — we know everything is mp4.

- **Scan all sub-folders**: ✅ **enabled**

  Required because clips live under `clips/{source_id}/`, not directly in `clips/`.

- Click **Load Preview** — should list the `.mp4` files for that source (or "No files found" if prep has not been run for that id yet)
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
python scripts/push_annotations.py /path/to/your-export.json
```

Optional: `--dry-run` first. See [workflow_overview.md](workflow_overview.md) and W3 in [annotation_schema_and_systems.md](annotation_schema_and_systems.md).

When the source is fully labeled and pushed, mark it **Done** (or equivalent) in the **Google Sheet**.
