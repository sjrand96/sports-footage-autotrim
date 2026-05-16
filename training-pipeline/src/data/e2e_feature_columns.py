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

FEATURE_COLUMNS: list[str] = list(FEATURE_COLUMNS_BASE) + list(FEATURE_COLUMNS_CHUNK1_SPATIAL)

INTEGER_E2E_COLUMNS: frozenset[str] = frozenset(
    {
        "n_players_total",
        "n_front_row",
        "n_back_row",
        "n_camera_side",
        "n_opposite_side",
        "hands_above_head_count",
        "n_pose_instances_raw",
    }
)


def active_feature_columns(feature_subset: str) -> list[str]:
    subset = feature_subset.strip().lower()
    if subset == "all":
        return list(FEATURE_COLUMNS)
    if subset == "base":
        return list(FEATURE_COLUMNS_BASE)
    raise ValueError(f"feature_subset must be 'all' or 'base', got {feature_subset!r}")
