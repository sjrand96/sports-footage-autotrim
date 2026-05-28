"""Feature columns produced by cv-pipeline/simplified_e2e_flow parquets."""

from __future__ import annotations

FEATURE_COLUMNS_BASE: list[str] = [
    "n_players_total",
    "n_front_row",
    "n_back_row",
    "n_camera_side",
    "n_opposite_side",
    "median_nearest_neighbor_dist",
    "hands_above_head_count",
    "knee_angle_mean_deg",
    "knee_angle_min_deg",
    "squat_count",
    "squat_ratio",
    "wrists_above_shoulder_count",
    "high_five_pair_count",
]

FEATURE_COLUMNS_CHUNK1_SPATIAL: list[str] = [
    "n_pose_instances_raw",
    "camera_side_pairwise_mean_m",
    "camera_side_pairwise_std_m",
    "camera_side_pairwise_min_m",
    "camera_side_pairwise_max_m",
    "opposite_side_pairwise_mean_m",
    "opposite_side_pairwise_std_m",
    "opposite_side_pairwise_min_m",
    "opposite_side_pairwise_max_m",
    "camera_side_net_dist_mean_m",
    "camera_side_net_dist_std_m",
    "camera_side_net_dist_min_m",
    "camera_side_net_dist_max_m",
    "opposite_side_net_dist_mean_m",
    "opposite_side_net_dist_std_m",
    "opposite_side_net_dist_min_m",
    "opposite_side_net_dist_max_m",
    "camera_side_centroid_wx_m",
    "camera_side_centroid_wy_m",
    "opposite_side_centroid_wx_m",
    "opposite_side_centroid_wy_m",
    "inter_centroid_dist_m",
    "camera_side_mocon_mean_m",
    "camera_side_mocon_std_m",
    "camera_side_mocon_max_m",
    "opposite_side_mocon_mean_m",
    "opposite_side_mocon_std_m",
    "opposite_side_mocon_max_m",
]

FEATURE_COLUMNS_FLOW_MOTION: list[str] = [
    "flow_mag_mean",
    "flow_mag_std",
    "flow_mag_p90",
    "flow_mag_p95",
    "camera_side_centroid_speed_m",
    "opposite_side_centroid_speed_m",
    "inter_centroid_speed_m",
]

FEATURE_COLUMNS: list[str] = list(FEATURE_COLUMNS_BASE) + list(FEATURE_COLUMNS_CHUNK1_SPATIAL) + list(FEATURE_COLUMNS_FLOW_MOTION)

FEATURE_COLUMNS_COUNTS: list[str] = [
    "n_players_total",
    "n_front_row",
    "n_back_row",
    "n_camera_side",
    "n_opposite_side",
    "hands_above_head_count",
    "n_pose_instances_raw",
    "squat_count",
    "wrists_above_shoulder_count",
    "high_five_pair_count",
]

FEATURE_COLUMNS_PAIRWISE: list[str] = [
    "camera_side_pairwise_mean_m",
    "camera_side_pairwise_std_m",
    "camera_side_pairwise_min_m",
    "camera_side_pairwise_max_m",
    "opposite_side_pairwise_mean_m",
    "opposite_side_pairwise_std_m",
    "opposite_side_pairwise_min_m",
    "opposite_side_pairwise_max_m",
]

FEATURE_COLUMNS_NET_DIST: list[str] = [
    "camera_side_net_dist_mean_m",
    "camera_side_net_dist_std_m",
    "camera_side_net_dist_min_m",
    "camera_side_net_dist_max_m",
    "opposite_side_net_dist_mean_m",
    "opposite_side_net_dist_std_m",
    "opposite_side_net_dist_min_m",
    "opposite_side_net_dist_max_m",
]

FEATURE_COLUMNS_CENTROIDS: list[str] = [
    "camera_side_centroid_wx_m",
    "camera_side_centroid_wy_m",
    "opposite_side_centroid_wx_m",
    "opposite_side_centroid_wy_m",
    "inter_centroid_dist_m",
]

FEATURE_COLUMNS_MOCON: list[str] = [
    "camera_side_mocon_mean_m",
    "camera_side_mocon_std_m",
    "camera_side_mocon_max_m",
    "opposite_side_mocon_mean_m",
    "opposite_side_mocon_std_m",
    "opposite_side_mocon_max_m",
]

FEATURE_COLUMNS_POSE_ANGLES: list[str] = [
    "knee_angle_mean_deg",
    "knee_angle_min_deg",
]

FEATURE_COLUMNS_ACTIONS: list[str] = [
    "squat_count",
    "squat_ratio",
    "wrists_above_shoulder_count",
    "high_five_pair_count",
]

FEATURE_COLUMNS_TEMPORAL: list[str] = list(FEATURE_COLUMNS_FLOW_MOTION)

FEATURE_COLUMNS_SPATIAL: list[str] = list(FEATURE_COLUMNS_CHUNK1_SPATIAL)

INTEGER_E2E_COLUMNS: frozenset[str] = frozenset(
    {
        "n_players_total",
        "n_front_row",
        "n_back_row",
        "n_camera_side",
        "n_opposite_side",
        "hands_above_head_count",
        "n_pose_instances_raw",
        "squat_count",
        "wrists_above_shoulder_count",
        "high_five_pair_count",
    }
)


def active_feature_columns(feature_subset: str) -> list[str]:
    subset = feature_subset.strip().lower()
    if subset == "all":
        return list(FEATURE_COLUMNS)
    if subset == "base":
        return list(FEATURE_COLUMNS_BASE)
    if subset == "counts":
        return list(FEATURE_COLUMNS_COUNTS)
    if subset == "pairwise":
        return list(FEATURE_COLUMNS_PAIRWISE)
    if subset == "net_dist":
        return list(FEATURE_COLUMNS_NET_DIST)
    if subset == "centroids":
        return list(FEATURE_COLUMNS_CENTROIDS)
    if subset == "mocon":
        return list(FEATURE_COLUMNS_MOCON)
    if subset == "spatial":
        return list(FEATURE_COLUMNS_SPATIAL)
    if subset == "pose_angles":
        return list(FEATURE_COLUMNS_POSE_ANGLES)
    if subset == "actions":
        return list(FEATURE_COLUMNS_ACTIONS)
    if subset == "temporal":
        return list(FEATURE_COLUMNS_TEMPORAL)
    raise ValueError(
        "feature_subset must be one of 'all', 'base', 'counts', 'pairwise', "
        "'net_dist', 'centroids', 'mocon', 'spatial', 'pose_angles', "
        "'actions', 'temporal', "
        f"got {feature_subset!r}"
    )
