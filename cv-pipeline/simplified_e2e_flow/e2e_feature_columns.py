"""Single registry for pooled XGBoost feature columns (E2E parquet + trainer).

Chunk 1 spatial additions follow ``pose-feature-extraction-plan.md`` §7 Chunk 1 /
Appendix A #1 (raw count), #3–#6. ``wy < 0`` = camera side (near); ``wy >= 0`` =
opposite (far); net line ``wy = 0``; per-player net distance uses ``|wy|`` in metres.
"""

from __future__ import annotations

# Original seven (Tier-0)
FEATURE_COLUMNS_BASE: list[str] = [
    "n_players_total",
    "n_front_row",
    "n_back_row",
    "n_camera_side",
    "n_opposite_side",
    "median_nearest_neighbor_dist",
    "hands_above_head_count",
]

# Chunk 1 — Tier-1 spatial (pose-feature-extraction-plan.md §7 Chunk 1)
_CHUNK1_SPATIAL: list[str] = [
    "n_pose_instances_raw",
    # Pairwise foot distances within side (metres)
    "camera_side_pairwise_mean_m",
    "camera_side_pairwise_std_m",
    "camera_side_pairwise_min_m",
    "camera_side_pairwise_max_m",
    "opposite_side_pairwise_mean_m",
    "opposite_side_pairwise_std_m",
    "opposite_side_pairwise_min_m",
    "opposite_side_pairwise_max_m",
    # |wy| to net per side
    "camera_side_net_dist_mean_m",
    "camera_side_net_dist_std_m",
    "camera_side_net_dist_min_m",
    "camera_side_net_dist_max_m",
    "opposite_side_net_dist_mean_m",
    "opposite_side_net_dist_std_m",
    "opposite_side_net_dist_min_m",
    "opposite_side_net_dist_max_m",
    # Centroids (metres); near = camera side
    "camera_side_centroid_wx_m",
    "camera_side_centroid_wy_m",
    "opposite_side_centroid_wx_m",
    "opposite_side_centroid_wy_m",
    "inter_centroid_dist_m",
    # MOCON: distance to own-side centroid
    "camera_side_mocon_mean_m",
    "camera_side_mocon_std_m",
    "camera_side_mocon_max_m",
    "opposite_side_mocon_mean_m",
    "opposite_side_mocon_std_m",
    "opposite_side_mocon_max_m",
]

FEATURE_COLUMNS_CHUNK1_SPATIAL: list[str] = list(_CHUNK1_SPATIAL)

FEATURE_COLUMNS: list[str] = list(FEATURE_COLUMNS_BASE) + list(_CHUNK1_SPATIAL)

# Integer-like columns (no NaN fill for XGBoost input path)
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

# All non-integer feature columns get NaN → -1.0 before training (matches median_nn)
FEATURE_FLOAT_FILLNA_COLS: list[str] = [c for c in FEATURE_COLUMNS if c not in INTEGER_E2E_COLUMNS]


def active_feature_columns(feature_subset: str) -> list[str]:
    """``all`` = base + Chunk 1 spatial; ``base`` = original seven (fair A/B on same cache + seed)."""
    s = feature_subset.strip().lower()
    if s == "all":
        return list(FEATURE_COLUMNS)
    if s == "base":
        return list(FEATURE_COLUMNS_BASE)
    raise ValueError(f"feature_subset must be 'all' or 'base', got {feature_subset!r}")


def float_fillna_cols_for_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in INTEGER_E2E_COLUMNS]
