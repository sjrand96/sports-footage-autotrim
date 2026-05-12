#!/usr/bin/env python3
"""Feature research: per-frame visualization + pooled XGBoost explainability.

Modes
-----
frame-viz
    One video frame (by Supabase ``clips.id`` + source ``frame_idx``) with YOLO
    pose overlay, homography **top-down court** (warp + FIVB overlay + feet +
    side **centroid crosses**), base + Chunk~1 spatial columns from
    ``e2e_feature_columns.FEATURE_COLUMNS``.

pooled-explain
    Same clip-level train/test split and ``XGBClassifier`` as
    ``train_pooled_xgboost_from_cache.py``, then SHAP (TreeExplainer) on held-out
    test rows and optional sklearn permutation importance for global drop in
    accuracy when each column is shuffled.

Examples
--------
  python cv-pipeline/pose-based-feature-extraction/feature_lab.py frame-viz \\
    --clip-id 42 --frame-idx 1200 --out /tmp/frame_lab.png

  python cv-pipeline/pose-based-feature-extraction/feature_lab.py pooled-explain \\
    --out-dir /tmp/pooled_lab
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_PATH = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "simple_e2e_pipeline.py"
POOL_PATH = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "train_pooled_xgboost_from_cache.py"
_DEFAULT_CACHE_DIR = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "cache"
_LOCAL_CLIP_ROOT = REPO_ROOT / "cv-pipeline" / "pose-detection" / "media" / "clips"


def _ensure_sys_path_for_e2e() -> None:
    for p in (REPO_ROOT, REPO_ROOT / "cv-pipeline" / "calibration", REPO_ROOT / "cv-pipeline" / "pose-detection"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_e2e_module() -> Any:
    _ensure_sys_path_for_e2e()
    spec = importlib.util.spec_from_file_location("simple_e2e_pipeline_lab", E2E_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {E2E_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pool_module() -> Any:
    spec = importlib.util.spec_from_file_location("train_pooled_xgboost_lab", POOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {POOL_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _local_clip_path(source_id: str, clip_index: int) -> Path:
    return _LOCAL_CLIP_ROOT / source_id / f"{source_id}_{clip_index:03d}.mp4"


def cmd_frame_viz(args: argparse.Namespace) -> int:
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from dotenv import load_dotenv
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("frame-viz needs opencv-python, matplotlib, ultralytics, python-dotenv.") from exc

    load_dotenv(REPO_ROOT / ".env")
    sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers  # noqa: E402

    e2e = load_e2e_module()
    _flow = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow"
    if str(_flow) not in sys.path:
        sys.path.insert(0, str(_flow))
    from e2e_feature_columns import FEATURE_COLUMNS_BASE, FEATURE_COLUMNS_CHUNK1_SPATIAL  # noqa: E402

    client = db_helpers.get_supabase_client()
    clip_row = db_helpers.get_clip_by_id(client, int(args.clip_id))
    if clip_row is None:
        print(f"no clips row for id={args.clip_id}", file=sys.stderr)
        return 1

    source_id = str(clip_row["source_id"])
    clip_index = int(clip_row["clip_index"])
    local_clip = _local_clip_path(source_id, clip_index)
    if not local_clip.is_file():
        print(
            f"local clip missing: {local_clip}\n"
            "Run simplified_e2e_flow/simple_e2e_pipeline.py for this clip (or copy the MP4 here).",
            file=sys.stderr,
        )
        return 1

    cal = db_helpers.get_court_calibration(client, source_id)
    if cal is None:
        print(f"no court_calibrations for source_id={source_id!r}", file=sys.stderr)
        return 1

    from homography_io import homography_arrays_from_court_calibration_row  # noqa: E402

    H, wx_min, wx_max, wy_min, wy_max, out_w, out_h = homography_arrays_from_court_calibration_row(cal)

    from court_homography import draw_world_rect_overlay, warp_topdown  # noqa: E402
    from pose_side_by_side_video import (  # noqa: E402
        _foot_uv_from_coco17,
        _image_uv_to_world_m,
        _world_to_canvas_px,
    )

    cap = cv2.VideoCapture(str(local_clip))
    if not cap.isOpened():
        print(f"could not open video: {local_clip}", file=sys.stderr)
        return 1
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(args.frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            print(f"could not read frame_idx={args.frame_idx}", file=sys.stderr)
            return 1
    finally:
        cap.release()

    model = YOLO(args.weights)
    result = model(frame_bgr, imgsz=int(args.imgsz), conf=float(args.det_conf), verbose=False)[0]

    feats = e2e.compute_e2e_feature_row_from_yolo_result(
        result,
        H=H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        ankle_conf=float(args.ankle_conf),
        kp_conf_thresh=float(args.kp_conf_thresh),
    )

    plot_bgr = result.plot()
    if hasattr(plot_bgr, "cpu"):
        plot_bgr = plot_bgr.cpu().numpy()
    rgb = plot_bgr[:, :, ::-1].copy()

    topdown = warp_topdown(
        frame_bgr,
        H,
        wx_min=wx_min,
        wx_max=wx_max,
        wy_min=wy_min,
        wy_max=wy_max,
        out_w=out_w,
        out_h=out_h,
    ).copy()
    draw_world_rect_overlay(topdown, wx_min=wx_min, wx_max=wx_max, wy_min=wy_min, wy_max=wy_max)

    kp_td = result.keypoints
    if kp_td is not None and kp_td.xy is not None and kp_td.xy.shape[0] > 0:
        xy_td = kp_td.xy.cpu().numpy()
        if kp_td.conf is not None:
            kconf_td = kp_td.conf.cpu().numpy()
        else:
            kconf_td = np.ones((xy_td.shape[0], xy_td.shape[1]), dtype=np.float32)
        n_td = xy_td.shape[0]
        ankle_c = float(args.ankle_conf)
        for i in range(n_td):
            foot = _foot_uv_from_coco17(xy_td[i], kconf_td[i], ankle_conf=ankle_c)
            if foot is None:
                continue
            wx_f, wy_f = _image_uv_to_world_m(H, foot[0], foot[1])
            cx, cy = _world_to_canvas_px(
                wx_f,
                wy_f,
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                out_w=out_w,
                out_h=out_h,
            )
            hue = int(180 * i / max(n_td, 1)) % 180
            col = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
            col = (int(col[0]), int(col[1]), int(col[2]))
            cv2.circle(topdown, (cx, cy), 12, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(topdown, (cx, cy), 13, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    import math

    def _finite_num(x: Any) -> bool:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return False
        return v == v and math.isfinite(v)

    for wx_k, wy_k, col_bgr in (
        ("camera_side_centroid_wx_m", "camera_side_centroid_wy_m", (0, 255, 255)),
        ("opposite_side_centroid_wx_m", "opposite_side_centroid_wy_m", (255, 128, 0)),
    ):
        if _finite_num(feats[wx_k]) and _finite_num(feats[wy_k]):
            cx_c, cy_c = _world_to_canvas_px(
                float(feats[wx_k]),
                float(feats[wy_k]),
                wx_min=wx_min,
                wx_max=wx_max,
                wy_min=wy_min,
                wy_max=wy_max,
                out_w=out_w,
                out_h=out_h,
            )
            cv2.drawMarker(
                topdown,
                (cx_c, cy_c),
                col_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

    top_rgb = topdown[:, :, ::-1].copy()

    is_playing: str | None = None
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    stem = f"{source_id}_{clip_index:03d}"
    pred_path = cache_dir / f"{stem}_predictions.parquet"
    if pred_path.is_file():
        df_p = pd.read_parquet(pred_path)
        m = df_p[df_p["frame_idx"] == int(args.frame_idx)]
        if not m.empty and "is_playing" in m.columns:
            is_playing = str(bool(m.iloc[0]["is_playing"]))

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.3, 1.35], width_ratios=[1, 1], hspace=0.18, wspace=0.1)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(rgb)
    ax_img.set_title(f"{stem}  frame_idx={int(args.frame_idx)}  (camera + pose)")
    ax_img.axis("off")

    ax_td = fig.add_subplot(gs[0, 1])
    ax_td.imshow(top_rgb)
    ax_td.set_title("Top-down (warp + feet; cross = side centroid)")
    ax_td.axis("off")

    ax_tbl_base = fig.add_subplot(gs[1, 0])
    ax_tbl_base.axis("off")
    lines_base = [f"{k}:  {feats[k]}" for k in FEATURE_COLUMNS_BASE]
    if is_playing is not None:
        lines_base.append("")
        lines_base.append(f"is_playing (cached parquet):  {is_playing}")
    else:
        lines_base.append("")
        lines_base.append("(no cached is_playing for this frame_idx)")
    ax_tbl_base.text(
        0.02,
        0.98,
        "\n".join(lines_base),
        transform=ax_tbl_base.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    ax_tbl_base.set_title("Base features", loc="left", fontsize=10)

    ax_tbl_sp = fig.add_subplot(gs[1, 1])
    ax_tbl_sp.axis("off")
    lines_sp = [f"{k}:  {feats[k]}" for k in FEATURE_COLUMNS_CHUNK1_SPATIAL]
    ax_tbl_sp.text(
        0.02,
        0.98,
        "\n".join(lines_sp),
        transform=ax_tbl_sp.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )
    ax_tbl_sp.set_title("Chunk 1 spatial (pairwise | net | centroid | MOCON)", loc="left", fontsize=10)

    fig.suptitle("E2E features (registry: simplified_e2e_flow/e2e_feature_columns.py)", fontsize=12)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


def cmd_pooled_explain(args: argparse.Namespace) -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("pooled-explain needs matplotlib.") from exc

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("pooled-explain needs shap. Install with `pip install shap`.") from exc

    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    pool = load_pool_module()
    _flow = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow"
    if str(_flow) not in sys.path:
        sys.path.insert(0, str(_flow))
    from e2e_feature_columns import active_feature_columns, float_fillna_cols_for_features  # noqa: E402

    cache_dir = Path(args.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() / cache_dir).resolve()
    else:
        cache_dir = cache_dir.resolve()

    train_args = argparse.Namespace(
        test_size=float(args.test_size),
        random_seed=int(args.random_seed),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        min_clips=int(args.min_clips),
        feature_subset=str(args.feature_subset),
    )
    train_args.feature_columns = active_feature_columns(train_args.feature_subset)

    df = pool.load_paired_clip_rows(cache_dir, train_args.feature_columns)
    report, _test_preds, model = pool.train_and_evaluate(df, train_args)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    clip_keys = np.array(sorted(df["clip_key"].unique().tolist()))
    _train_keys, test_keys = train_test_split(
        clip_keys,
        test_size=train_args.test_size,
        random_state=train_args.random_seed,
        shuffle=True,
    )
    test_df = df[df["clip_key"].isin(test_keys)].copy()
    feat_cols = list(train_args.feature_columns)
    fill_cols = float_fillna_cols_for_features(feat_cols)
    X_test = test_df[feat_cols].copy()
    for _col in fill_cols:
        X_test[_col] = X_test[_col].fillna(-1.0)
    y_test = test_df["is_playing"].astype(int).to_numpy()
    X_np = X_test.to_numpy(dtype=np.float32)

    max_n = int(args.shap_max_samples)
    if max_n > 0 and X_np.shape[0] > max_n:
        rng = np.random.default_rng(int(args.random_seed))
        idx = rng.choice(X_np.shape[0], size=max_n, replace=False)
        X_sub = X_np[idx]
        X_sub_df = X_test.iloc[idx]
    else:
        X_sub = X_np
        X_sub_df = X_test

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_sub)
    if isinstance(shap_raw, list):
        shap_vals = np.asarray(shap_raw[1], dtype=np.float64)
    else:
        shap_vals = np.asarray(shap_raw, dtype=np.float64)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_json = {name: float(v) for name, v in sorted(zip(feat_cols, mean_abs), key=lambda x: -x[1])}
    (out_dir / "shap_mean_abs.json").write_text(json.dumps(shap_json, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals,
        X_sub_df,
        feature_names=feat_cols,
        plot_type="bar",
        show=False,
        max_display=len(feat_cols) + 2,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 9))
    shap.summary_plot(
        shap_vals,
        X_sub_df,
        feature_names=feat_cols,
        show=False,
        max_display=len(feat_cols) + 2,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    perm = permutation_importance(
        model,
        X_np,
        y_test,
        n_repeats=int(args.permutation_repeats),
        random_state=int(args.random_seed),
        n_jobs=-1,
    )
    perm_dict = {
        name: {"mean": float(m), "std": float(s)}
        for name, m, s in zip(feat_cols, perm.importances_mean, perm.importances_std)
    }
    perm_sorted = dict(sorted(perm_dict.items(), key=lambda kv: -kv[1]["mean"]))
    (out_dir / "permutation_importance_accuracy.json").write_text(json.dumps(perm_sorted, indent=2), encoding="utf-8")

    print("=== pooled-explain ===")
    print(f"feature_subset: {train_args.feature_subset}  ({len(feat_cols)} columns)")
    print(f"cache_dir: {cache_dir}")
    print(f"out_dir:   {out_dir}")
    print(f"SHAP rows: {X_sub.shape[0]} (of {X_np.shape[0]} test rows)")
    print("metrics (test clips):")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v:.4f}")
    print("mean |SHAP| (higher = more influence on this sample set):")
    for name in feat_cols:
        print(f"  {name}: {shap_json[name]:.6f}")
    print("permutation Δ accuracy (mean over repeats; higher = feature used by model):")
    for name in feat_cols:
        print(f"  {name}: {perm_dict[name]['mean']:.6f} ± {perm_dict[name]['std']:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature lab: frame viz + pooled SHAP / permutation importance.")
    sub = p.add_subparsers(dest="command", required=True)

    p_frame = sub.add_parser("frame-viz", help="Visualize one frame and its E2E feature vector.")
    p_frame.add_argument("--clip-id", type=int, required=True, help="Supabase clips.id")
    p_frame.add_argument("--frame-idx", type=int, required=True, help="Zero-based source frame index (same as parquet).")
    p_frame.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p_frame.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Where *_predictions.parquet live (optional is_playing overlay).",
    )
    p_frame.add_argument(
        "--weights",
        type=str,
        default="yolov8s-pose.pt",
        help="YOLO pose weights (name or path; same as E2E pipeline).",
    )
    p_frame.add_argument("--imgsz", type=int, default=1280)
    p_frame.add_argument("--det-conf", type=float, default=0.15)
    p_frame.add_argument("--ankle-conf", type=float, default=0.25)
    p_frame.add_argument("--kp-conf-thresh", type=float, default=0.25)

    p_pool = sub.add_parser("pooled-explain", help="Train pooled XGBoost like cache trainer + SHAP + permutation importance.")
    p_pool.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    p_pool.add_argument("--out-dir", type=Path, required=True)
    p_pool.add_argument("--test-size", type=float, default=0.2)
    p_pool.add_argument("--random-seed", type=int, default=42)
    p_pool.add_argument("--n-estimators", type=int, default=400)
    p_pool.add_argument("--max-depth", type=int, default=5)
    p_pool.add_argument("--learning-rate", type=float, default=0.05)
    p_pool.add_argument("--subsample", type=float, default=0.9)
    p_pool.add_argument("--colsample-bytree", type=float, default=0.9)
    p_pool.add_argument("--min-clips", type=int, default=2)
    p_pool.add_argument("--shap-max-samples", type=int, default=4000, help="Cap test rows for SHAP plots (0 = all).")
    p_pool.add_argument("--permutation-repeats", type=int, default=8)
    p_pool.add_argument(
        "--feature-subset",
        choices=("all", "base"),
        default="all",
        help="Match train_pooled_xgboost_from_cache: 'base' = 7 legacy columns only for SHAP/train.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "frame-viz":
        return cmd_frame_viz(args)
    if args.command == "pooled-explain":
        return cmd_pooled_explain(args)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
