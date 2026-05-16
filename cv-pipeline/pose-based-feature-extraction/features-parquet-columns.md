# `*_features.parquet` column reference

Parquet files under `cv-pipeline/simplified_e2e_flow/cache/` (e.g. `1rXZJyVXUHU_001_features.parquet`) are written by `simple_e2e_pipeline.py`. Column semantics and naming follow **`pose-feature-extraction-plan.md`** (court coords, net at `wy = 0`, clip vs `source_id`, sampling at `--fps`). The **model** uses only the columns listed in `e2e_feature_columns.py` (`FEATURE_COLUMNS`); the leading columns are row identity and provenance.

**World axes:** feet are mapped to metres with the match homography; **`wy < 0`** = camera side (near), **`wy ≥ 0`** = opposite (far); **net line** is **`wy = 0`**. **Front vs back row** uses **`|wy| < 3` m** vs **`|wy| ≥ 3` m** (plan §2.1).

---

## Identifiers and provenance (not in `FEATURE_COLUMNS`)

| Column | One-sentence meaning |
|--------|----------------------|
| `source_id` | YouTube / `source_videos` id for the full match recording (plan §2.2). |
| `clip_index` | 1-based index of the ~60 s segment within that source (`clips` table). |
| `clip_s3_uri` | S3 URI used when the clip was downloaded for this run. |
| `clip_local_path` | Local filesystem path to the cached video segment used for extraction. |
| `frame_idx` | Source-video frame index of this sample (not every frame: subsampled by `--fps`). |
| `timestamp_sec` | Wall-clock time in the source file, `frame_idx / source_fps` (plan §2.4). |

---

## Numeric features (`FEATURE_COLUMNS` — training / pooled join)

Order matches `e2e_feature_columns.py`: **base (7)** then **Chunk 1 spatial** (plan §7 Chunk 1 / Appendix A #1 raw count, #3–#6).

### Base tier (original seven)

| Column | One-sentence meaning |
|--------|----------------------|
| `n_players_total` | Number of pose detections with feet inside the calibration court bounds (in-court players for this frame). |
| `n_front_row` | Of those, count with **`|wy| < 3` m** (rough front-row band vs back row; plan §2.1). |
| `n_back_row` | Of those, count with **`|wy| ≥ 3` m**. |
| `n_camera_side` | In-court count on **camera side of net** (`wy < 0`; plan §2.1). |
| `n_opposite_side` | In-court count on **opposite side** (`wy ≥ 0`). |
| `median_nearest_neighbor_dist` | Median pairwise Euclidean distance (m) between in-court foot positions—spread vs tight cluster (plan §3 / Appendix #3 partial). |
| `hands_above_head_count` | Detections where wrists are above the nose in image space with enough keypoint confidence—global arms-up proxy for spike/block/set/celebration (plan Appendix #7, partial implementation). |

### Chunk 1 — spatial (pairwise, net distance, centroids, MOCON)

| Column | One-sentence meaning |
|--------|----------------------|
| `n_pose_instances_raw` | Raw YOLO pose instance count before dropping detections outside the court polygon (Appendix A #1 “raw detection count”). |
| `camera_side_pairwise_mean_m` | Mean pairwise foot distance (m) among **camera-side** in-court players (Appendix #3). |
| `camera_side_pairwise_std_m` | Standard deviation of those pairwise distances. |
| `camera_side_pairwise_min_m` | Minimum pairwise foot distance on camera side. |
| `camera_side_pairwise_max_m` | Maximum pairwise foot distance on camera side. |
| `opposite_side_pairwise_mean_m` | Same **mean** pairwise distance for **opposite-side** feet. |
| `opposite_side_pairwise_std_m` | Same **std** for opposite side. |
| `opposite_side_pairwise_min_m` | Same **min** for opposite side. |
| `opposite_side_pairwise_max_m` | Same **max** for opposite side. |
| `camera_side_net_dist_mean_m` | Mean **`|wy|`** (m) to net line `wy = 0` for camera-side feet—depth along court (Appendix #4). |
| `camera_side_net_dist_std_m` | Std of **`|wy|`** on camera side. |
| `camera_side_net_dist_min_m` | Min **`|wy|`** on camera side. |
| `camera_side_net_dist_max_m` | Max **`|wy|`** on camera side. |
| `opposite_side_net_dist_mean_m` | Mean **`|wy|`** for opposite-side feet. |
| `opposite_side_net_dist_std_m` | Std of **`|wy|`** on opposite side. |
| `opposite_side_net_dist_min_m` | Min **`|wy|`** on opposite side. |
| `opposite_side_net_dist_max_m` | Max **`|wy|`** on opposite side. |
| `camera_side_centroid_wx_m` | Mean **wx** (m) of camera-side in-court feet—side centroid X (Appendix #5). |
| `camera_side_centroid_wy_m` | Mean **wy** (m) of camera-side feet—side centroid Y. |
| `opposite_side_centroid_wx_m` | Mean **wx** (m) of opposite-side feet. |
| `opposite_side_centroid_wy_m` | Mean **wy** (m) of opposite-side feet. |
| `inter_centroid_dist_m` | Distance (m) between the two side foot centroids (Appendix #5). |
| `camera_side_mocon_mean_m` | Mean distance (m) from each camera-side foot to **that side’s** centroid—spread vs huddle (MOCON-style; Appendix #6). |
| `camera_side_mocon_std_m` | Std of those distances to the camera-side centroid. |
| `camera_side_mocon_max_m` | Max distance to the camera-side centroid. |
| `opposite_side_mocon_mean_m` | Mean distance from each opposite-side foot to the **opposite** centroid. |
| `opposite_side_mocon_std_m` | Std of distances to the opposite-side centroid. |
| `opposite_side_mocon_max_m` | Max distance to the opposite-side centroid. |

---

## Related files

- Column registry: `cv-pipeline/simplified_e2e_flow/e2e_feature_columns.py`
- Extraction: `cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py` (`extract_features_for_clip`, `compute_e2e_feature_row_from_yolo_result`)
- Spatial aggregates: `cv-pipeline/simplified_e2e_flow/pose_feature_spatial.py`
- Spec and roadmap: `cv-pipeline/pose-based-feature-extraction/pose-feature-extraction-plan.md`

If a cached parquet predates a schema bump, **regenerate** with `simple_e2e_pipeline.py` for that clip or clear stale files (plan §4 backward compatibility).
