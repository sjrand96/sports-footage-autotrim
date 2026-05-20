# Labeling docs

| Doc | Use |
|-----|-----|
| [workflow_overview.md](workflow_overview.md) | Day-to-day: ingest → Label Studio → export → push timeline to DB |
| [label-studio-setup.md](label-studio-setup.md) | Local LS, S3 prefix, timeline + optional court keypoints project |
| [annotation_schema_and_systems.md](annotation_schema_and_systems.md) | S3 layout, Supabase semantics, credentials, workflows |
| [court_calibration_supabase.md](court_calibration_supabase.md) | Court homography: LS stills, `court_calibrations` table + SQL; run `python data_labeling/push_court_calibration.py export.json` after export |

Executable DDL for core tables: [schema.md](../schema.md). Code: [data_labeling/README.md](../../data_labeling/README.md).
