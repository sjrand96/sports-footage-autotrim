"""Tabular feature column registry for feature-extraction parquets."""

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

_CHUNK1_SPATIAL: list[str] = [
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

_FLOW_AND_MOTION: list[str] = [
    "flow_mag_mean",
    "flow_mag_std",
    "flow_mag_p90",
    "flow_mag_p95",
    "camera_side_centroid_speed_m",
    "opposite_side_centroid_speed_m",
    "inter_centroid_speed_m",
]

NEW_FEATURE_COLUMNS: list[str] = [
    "knee_angle_mean_deg",
    "knee_angle_min_deg",
    "squat_count",
    "squat_ratio",
    "wrists_above_shoulder_count",
    "high_five_pair_count",
    "flow_mag_mean",
    "flow_mag_std",
    "flow_mag_p90",
    "flow_mag_p95",
    "camera_side_centroid_speed_m",
    "opposite_side_centroid_speed_m",
    "inter_centroid_speed_m",
]

FEATURE_COLUMNS_CHUNK1_SPATIAL: list[str] = list(_CHUNK1_SPATIAL)
FEATURE_COLUMNS_FLOW_MOTION: list[str] = list(_FLOW_AND_MOTION)

FEATURE_COLUMNS: list[str] = list(FEATURE_COLUMNS_BASE) + list(_CHUNK1_SPATIAL) + list(_FLOW_AND_MOTION)

PROVENANCE_COLUMNS: list[str] = [
    "source_id",
    "clip_index",
    "clip_id",
    "clip_s3_uri",
    "clip_local_path",
    "frame_idx",
    "timestamp_sec",
    "is_playing",
]

PARQUET_COLUMNS: list[str] = list(PROVENANCE_COLUMNS) + list(FEATURE_COLUMNS)

INTEGER_FEATURE_COLUMNS: frozenset[str] = frozenset(
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
    """``all`` = base + Chunk 1 spatial; ``base`` = original seven."""
    s = feature_subset.strip().lower()
    if s == "all":
        return list(FEATURE_COLUMNS)
    if s == "base":
        return list(FEATURE_COLUMNS_BASE)
    raise ValueError(f"feature_subset must be 'all' or 'base', got {feature_subset!r}")


def float_fillna_cols_for_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in INTEGER_FEATURE_COLUMNS]
