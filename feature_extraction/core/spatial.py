"""Chunk 1 spatial feature helpers (metres, court coords)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pairwise_stats_m(world_xy: np.ndarray) -> tuple[float, float, float, float]:
    n = int(world_xy.shape[0])
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    dvec: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dvec.append(float(np.linalg.norm(world_xy[i] - world_xy[j])))
    arr = np.asarray(dvec, dtype=np.float64)
    return (float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max()))


def _net_distance_stats_m(world_xy: np.ndarray) -> tuple[float, float, float, float]:
    n = int(world_xy.shape[0])
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    d = np.abs(world_xy[:, 1].astype(np.float64))
    return (float(d.mean()), float(d.std(ddof=0)), float(d.min()), float(d.max()))


def _centroid_m(world_xy: np.ndarray) -> tuple[float, float]:
    if world_xy.shape[0] == 0:
        return (float("nan"), float("nan"))
    c = world_xy.mean(axis=0)
    return (float(c[0]), float(c[1]))


def _inter_centroid_m(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def _mocon_mean_std_max_m(world_xy: np.ndarray) -> tuple[float, float, float]:
    n = int(world_xy.shape[0])
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    c = world_xy.mean(axis=0)
    dists = np.linalg.norm(world_xy - c, axis=1)
    return (float(dists.mean()), float(dists.std(ddof=0)), float(dists.max()))


def chunk1_spatial_dict(
    *,
    n_pose_instances_raw: int,
    camera_world_xy: np.ndarray,
    opposite_world_xy: np.ndarray,
) -> dict[str, Any]:
    cam = np.asarray(camera_world_xy, dtype=np.float64).reshape(-1, 2)
    opp = np.asarray(opposite_world_xy, dtype=np.float64).reshape(-1, 2)

    pmn, psd, pmi, pmx = _pairwise_stats_m(cam)
    omn, osd, omi, omx = _pairwise_stats_m(opp)

    cn_mn, cn_sd, cn_mi, cn_mx = _net_distance_stats_m(cam)
    on_mn, on_sd, on_mi, on_mx = _net_distance_stats_m(opp)

    ccx, ccy = _centroid_m(cam)
    ocx, ocy = _centroid_m(opp)
    inter = _inter_centroid_m(cam, opp)

    mcm, mcs, mcx = _mocon_mean_std_max_m(cam)
    mom, mos, mox = _mocon_mean_std_max_m(opp)

    return {
        "n_pose_instances_raw": int(n_pose_instances_raw),
        "camera_side_pairwise_mean_m": pmn,
        "camera_side_pairwise_std_m": psd,
        "camera_side_pairwise_min_m": pmi,
        "camera_side_pairwise_max_m": pmx,
        "opposite_side_pairwise_mean_m": omn,
        "opposite_side_pairwise_std_m": osd,
        "opposite_side_pairwise_min_m": omi,
        "opposite_side_pairwise_max_m": omx,
        "camera_side_net_dist_mean_m": cn_mn,
        "camera_side_net_dist_std_m": cn_sd,
        "camera_side_net_dist_min_m": cn_mi,
        "camera_side_net_dist_max_m": cn_mx,
        "opposite_side_net_dist_mean_m": on_mn,
        "opposite_side_net_dist_std_m": on_sd,
        "opposite_side_net_dist_min_m": on_mi,
        "opposite_side_net_dist_max_m": on_mx,
        "camera_side_centroid_wx_m": ccx,
        "camera_side_centroid_wy_m": ccy,
        "opposite_side_centroid_wx_m": ocx,
        "opposite_side_centroid_wy_m": ocy,
        "inter_centroid_dist_m": inter,
        "camera_side_mocon_mean_m": mcm,
        "camera_side_mocon_std_m": mcs,
        "camera_side_mocon_max_m": mcx,
        "opposite_side_mocon_mean_m": mom,
        "opposite_side_mocon_std_m": mos,
        "opposite_side_mocon_max_m": mox,
    }
