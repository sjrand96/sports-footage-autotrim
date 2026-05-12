# `data_labeling` — scripts

Python entrypoints for **human labeling workflows** (different Supabase tables per flow). Deep docs: [docs/annotation_process/README.md](../docs/annotation_process/README.md).

| Script | Needs `.env` | Doc |
|--------|----------------|-----|
| [ingest_youtube_source.py](ingest_youtube_source.py) | AWS, `S3_BUCKET`, Supabase | [workflow_overview.md](../docs/annotation_process/workflow_overview.md) |
| [push_timeline_annotation_export.py](push_timeline_annotation_export.py) | Supabase, `ANNOTATOR_NAME` | [annotation_schema_and_systems.md](../docs/annotation_process/annotation_schema_and_systems.md) (W3) |
| [court_keypoints.py](court_keypoints.py) | — (parse only) | [label-studio-setup.md](../docs/annotation_process/label-studio-setup.md) (court project), [court_calibration_supabase.md](../docs/annotation_process/court_calibration_supabase.md) |

Run from repo root, e.g. `python data_labeling/ingest_youtube_source.py '<youtube_url>'`. Shared DB helpers: [src/db.py](../src/db.py).
