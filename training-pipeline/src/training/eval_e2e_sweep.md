# E2E sweep evaluation script

This doc describes how to run the E2E subset evaluation sweep and what artifacts it writes.

## Script

Location: training-pipeline/scripts/eval_e2e_sweep.sh

The script evaluates each trained subset checkpoint and writes:

- test_metrics.json
- test_frame_metrics.json (center-window frame eval)
- test_timeline.png (timeline visualization)

Outputs are written under:

training-pipeline/outputs/e2e_<fusion>_<subset>/

## How to run

```bash
bash training-pipeline/scripts/eval_e2e_sweep.sh
```

## Parameters baked into the script

- manifest: training-pipeline/data/window_manifest_full127_test.jsonl
- features-dir: training-pipeline/data/embeddings_resnet18
- e2e-features-dir: s3://sports-footage-autotrim-bucket/feature_extraction/full_127_limitedconcurr_20260518/test/
- fusion: early
- batch-size: 32
- num-workers: 4
- frame-eval: center
- frame-labels-dir: same as e2e-features-dir
- s3-cache-dir: training-pipeline/data/s3_cache
- clip-cache-dir: training-pipeline/data/s3_cache/clips

## Subsets swept

- base
- all
- counts
- pairwise
- net_dist
- centroids
- mocon
- spatial
- pose_angles
- actions
- temporal

- **base**:  
  `n_players_total`, `n_front_row`, `n_back_row`, `n_camera_side`, `n_opposite_side`, `median_nearest_neighbor_dist`, `hands_above_head_count`, `knee_angle_mean_deg`, `knee_angle_min_deg`, `squat_count`, `squat_ratio`, `wrists_above_shoulder_count`, `high_five_pair_count`

- **counts**:  
  `n_players_total`, `n_front_row`, `n_back_row`, `n_camera_side`, `n_opposite_side`, `hands_above_head_count`, `n_pose_instances_raw`, `squat_count`, `wrists_above_shoulder_count`, `high_five_pair_count`

- **pairwise**:  
  `camera_side_pairwise_mean_m`, `camera_side_pairwise_std_m`, `camera_side_pairwise_min_m`, `camera_side_pairwise_max_m`,  
  `opposite_side_pairwise_mean_m`, `opposite_side_pairwise_std_m`, `opposite_side_pairwise_min_m`, `opposite_side_pairwise_max_m`

- **net_dist**:  
  `camera_side_net_dist_mean_m`, `camera_side_net_dist_std_m`, `camera_side_net_dist_min_m`, `camera_side_net_dist_max_m`,  
  `opposite_side_net_dist_mean_m`, `opposite_side_net_dist_std_m`, `opposite_side_net_dist_min_m`, `opposite_side_net_dist_max_m`

- **centroids**:  
  `camera_side_centroid_wx_m`, `camera_side_centroid_wy_m`, `opposite_side_centroid_wx_m`, `opposite_side_centroid_wy_m`, `inter_centroid_dist_m`

- **mocon**:  
  `camera_side_mocon_mean_m`, `camera_side_mocon_std_m`, `camera_side_mocon_max_m`,  
  `opposite_side_mocon_mean_m`, `opposite_side_mocon_std_m`, `opposite_side_mocon_max_m`

- **spatial** (all chunk1 spatial):  
  `n_pose_instances_raw` plus all `pairwise`, `net_dist`, `centroids`, and `mocon` columns listed above

- **pose_angles**:  
  `knee_angle_mean_deg`, `knee_angle_min_deg`

- **actions**:  
  `squat_count`, `squat_ratio`, `wrists_above_shoulder_count`, `high_five_pair_count`

- **temporal** (flow/motion):  
  `flow_mag_mean`, `flow_mag_std`, `flow_mag_p90`, `flow_mag_p95`,  
  `camera_side_centroid_speed_m`, `opposite_side_centroid_speed_m`, `inter_centroid_speed_m`

- **all**:  
  `base` + `spatial` + `temporal` (full list: `FEATURE_COLUMNS`)

If a checkpoint is missing for a subset, the script skips it.



| Subset | Precision | Recall | F1 | F2 | Accuracy | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.7362 | 0.6816 | 0.7078 | 0.6919 | 0.8094 | 9830 | 3523 | 24620 | 4592 |
| counts | 0.4314 | 0.9727 | 0.5977 | 0.7775 | 0.5563 | 14028 | 18492 | 9651 | 394 |
| net_dist | 0.8966 | 0.2550 | 0.3970 | 0.2975 | 0.7376 | 3677 | 424 | 27719 | 10745 |
| centroids | 0.8368 | 0.5644 | 0.6741 | 0.6037 | 0.8151 | 8140 | 1587 | 26556 | 6282 |
| mocon | 0.6545 | 0.7788 | 0.7113 | 0.7503 | 0.7858 | 11232 | 5929 | 22214 | 3190 |
| spatial | 0.6319 | 0.8227 | 0.7148 | 0.7758 | 0.7775 | 11865 | 6912 | 21231 | 2557 |
| pose_angles | 0.7779 | 0.5793 | 0.6641 | 0.6105 | 0.8014 | 8355 | 2385 | 25758 | 6067 |
| temporal | 0.6367 | 0.7577 | 0.6919 | 0.7299 | 0.7714 | 10927 | 6235 | 21908 | 3495 |
