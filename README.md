# sports-footage-autotrim

**CS348K —** Build toward **automatic volleyball trims**: long casual match footage (lots of downtime) → **when play is happening** → a shorter **gameplay-focused** cut. Human labels on short clips are the ground truth for models and evaluation later.

## Where things live

| Path | What it is |
|------|------------|
| [docs/annotation_process/README.md](docs/annotation_process/README.md) | Index of labeling docs (playing timeline + court calibration) |
| [docs/annotation_process/workflow_overview.md](docs/annotation_process/workflow_overview.md) | **Start here for day-to-day work:** one-time setup, ingest → Label Studio → export → push to DB, flow diagram |
| [docs/annotation_process/label-studio-setup.md](docs/annotation_process/label-studio-setup.md) | Local Label Studio: venv, project template, S3 prefix per `source_id` |
| [docs/annotation_process/annotation_schema_and_systems.md](docs/annotation_process/annotation_schema_and_systems.md) | Architecture: S3 layout, Supabase tables (semantics), credentials, reprocessing, workflows W1–W3 |
| [docs/annotation_process/court_calibration_supabase.md](docs/annotation_process/court_calibration_supabase.md) | Court homography: Label Studio stills, `court_calibrations` DDL + contract |
| [docs/schema.md](docs/schema.md) | Executable **SQL** for Supabase (DDL only) |
| [weekly-updates/](weekly-updates/) | **Project scope, roadmap, milestones, evaluation** (e.g. [initial_proposal.md](weekly-updates/initial_proposal.md)) |
| [data_labeling/README.md](data_labeling/README.md) | Scripts hub (ingest, push timeline annotations, push court calibration, court_keypoints) + links to docs |
| [data_labeling/ingest_youtube_source.py](data_labeling/ingest_youtube_source.py) | YouTube URL → 60 s / 30 fps clips → S3 + `source_videos` / `clips` |
| [data_labeling/push_timeline_annotation.py](data_labeling/push_timeline_annotation.py) | Label Studio **timeline** export (Playing) → `annotations` rows |
| [data_labeling/push_court_calibration.py](data_labeling/push_court_calibration.py) | Court keypoint export → fit → `court_calibrations` upsert |
| [data_labeling/court_keypoints.py](data_labeling/court_keypoints.py) | Court **KeyPointLabels** export → normalized JSON (`calibration_record_to_json`) |
| [src/db.py](src/db.py) | Shared Supabase helpers for the scripts above |
| [pyproject.toml](pyproject.toml) | Python deps for the pipeline (`pip install -e .`) |

## Quick start (collaborators)

1. Clone the repo, Python 3.10+, `brew install yt-dlp ffmpeg` (or equivalent).
2. `python -m venv .venv && source .venv/bin/activate` then `pip install -e .`
3. Add `.env` using the team template (see [workflow overview](docs/annotation_process/workflow_overview.md#one-time-setup)).
4. Follow **[docs/annotation_process/workflow_overview.md](docs/annotation_process/workflow_overview.md)** for ingesting a source and labeling.

Modeling, baselines, and UI work are **out of scope for this README**; see **weekly-updates** and course milestones for that thread.
