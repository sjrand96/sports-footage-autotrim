#!/usr/bin/env python3
"""Plot actual vs predicted ``is_playing`` for one clip (intuition timeline).

Looks up ``clips.id`` in Supabase, then reads
``{cache_dir}/{source_id}_{clip_index:03d}_predictions.parquet`` for labels and
timestamps.

**Default (no ``--pooled-model``):** uses ``pred_playing`` from that parquet. In
E2E, that column is from an XGBoost model **fit and evaluated on the same clip
rows** (wiring placeholder), so agreement is often **near-perfect** and does not
measure generalization.

**With ``--pooled-model``:** loads ``*_features.parquet``, merges labels from
predictions, runs the **saved pooled** XGBoost model (same feature columns as
training). That is usually the comparison you want for “how wrong are we on
this clip?” (the clip may or may not have been in the pooled model’s train set).

Example::

    python cv-pipeline/pose-based-feature-extraction/clip_pred_timeline.py \\
      --clip-id 69 --out outputs/clip69_timeline.png

    python cv-pipeline/pose-based-feature-extraction/clip_pred_timeline.py \\
      --clip-id 69 --pooled-model path/to/model.json \\
      --out outputs/clip69_pooled_timeline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow" / "cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Timeline PNG: is_playing vs pred_playing for one clip.")
    p.add_argument("--clip-id", type=int, required=True, help="Supabase clips.id")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Directory containing *_predictions.parquet (default: E2E cache).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    p.add_argument(
        "--predictions-parquet",
        type=Path,
        default=None,
        help="Override parquet path (default: derive stem from clip id + cache dir).",
    )
    p.add_argument(
        "--pooled-model",
        type=Path,
        default=None,
        help="Saved pooled XGBoost model (JSON/UBJ). Re-predict from *_features.parquet + labels.",
    )
    p.add_argument(
        "--feature-subset",
        choices=("all", "base"),
        default="all",
        help="Feature columns when using --pooled-model (must match how the model was trained).",
    )
    return p.parse_args()


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Need matplotlib and python-dotenv.") from exc

    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    sys.path.insert(0, str(REPO_ROOT))
    from src import db as db_helpers  # noqa: E402

    cache_dir = Path(args.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (Path.cwd() / cache_dir).resolve()
    else:
        cache_dir = cache_dir.resolve()

    client = db_helpers.get_supabase_client()
    clip_row = db_helpers.get_clip_by_id(client, int(args.clip_id))
    if clip_row is None:
        print(f"no clips row for id={args.clip_id}", file=sys.stderr)
        return 1

    source_id = str(clip_row["source_id"])
    clip_index = int(clip_row["clip_index"])
    stem = f"{source_id}_{clip_index:03d}"

    if args.predictions_parquet is not None:
        pred_path = Path(args.predictions_parquet).expanduser().resolve()
    else:
        pred_path = cache_dir / f"{stem}_predictions.parquet"

    if not pred_path.is_file():
        print(
            f"missing predictions parquet: {pred_path}\n"
            "Run: python cv-pipeline/simplified_e2e_flow/simple_e2e_pipeline.py --clip-id <same id> …",
            file=sys.stderr,
        )
        return 1

    df = pd.read_parquet(pred_path)
    required_lab = {"timestamp_sec", "is_playing", "frame_idx"}
    missing_lab = required_lab - set(df.columns)
    if missing_lab:
        print(f"{pred_path.name} missing columns: {sorted(missing_lab)}", file=sys.stderr)
        return 1

    note_lines: list[str] = []

    if args.pooled_model is not None:
        feat_path = cache_dir / f"{stem}_features.parquet"
        if not feat_path.is_file():
            print(f"missing features parquet for pooled mode: {feat_path}", file=sys.stderr)
            return 1
        _flow = REPO_ROOT / "cv-pipeline" / "simplified_e2e_flow"
        if str(_flow) not in sys.path:
            sys.path.insert(0, str(_flow))
        from e2e_feature_columns import active_feature_columns, float_fillna_cols_for_features  # noqa: E402

        feat_cols = active_feature_columns(str(args.feature_subset))
        df_feat = pd.read_parquet(feat_path)
        miss_f = [c for c in feat_cols if c not in df_feat.columns]
        if miss_f:
            print(f"{feat_path.name} missing feature columns (showing first 12): {miss_f[:12]}", file=sys.stderr)
            return 1

        lab = df[["frame_idx", "timestamp_sec", "is_playing"]].copy()
        feat_only = df_feat[["frame_idx"] + feat_cols].copy()
        merged = lab.merge(feat_only, on="frame_idx", how="inner", validate="one_to_one")
        if merged.empty:
            print("merge(features, labels on frame_idx) returned 0 rows", file=sys.stderr)
            return 1

        df = merged.sort_values("timestamp_sec", kind="mergesort").reset_index(drop=True)
        Xs = df[feat_cols].copy()
        for c in float_fillna_cols_for_features(feat_cols):
            Xs[c] = Xs[c].fillna(-1.0)

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("pip install xgboost for --pooled-model") from exc

        model = XGBClassifier()
        model.load_model(str(Path(args.pooled_model).expanduser().resolve()))
        y_pred = model.predict(Xs.to_numpy(dtype=np.float32)).astype(bool)
        y_true = df["is_playing"].astype(bool).to_numpy()
        ts = df["timestamp_sec"].to_numpy(dtype=np.float64)
        pred_label = f"pooled XGB ({args.feature_subset})"
        note_lines.append(
            "Predictions from saved pooled model on this clip’s feature rows (clip may have been in or out of that model’s train set)."
        )
    else:
        if "pred_playing" not in df.columns:
            print(f"{pred_path.name} missing column: pred_playing", file=sys.stderr)
            return 1
        df = df.sort_values("timestamp_sec", kind="mergesort").reset_index(drop=True)
        ts = df["timestamp_sec"].to_numpy(dtype=np.float64)
        y_true = df["is_playing"].astype(bool).to_numpy()
        y_pred = df["pred_playing"].astype(bool).to_numpy()
        pred_label = "E2E in-sample XGB"
        print(
            "NOTE: E2E parquet pred_playing is in-sample (XGB fit + predict on the same clip rows). "
            "Near-perfect overlap is expected. Use --pooled-model for a stricter single-clip view.",
            file=sys.stderr,
        )
        note_lines.append(
            "E2E path: pred_playing = in-sample XGB (train and predict on these same rows)—not a generalization test."
        )
    mismatch = y_true != y_pred
    n = int(len(df))
    n_err = int(np.sum(mismatch))
    frac_err = float(n_err / max(n, 1))

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(14, 4.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    y0 = y_true.astype(np.float64)
    y1 = y_pred.astype(np.float64)

    ax0.fill_between(ts, 0.0, y0, step="post", alpha=0.35, color="tab:green", linewidth=0)
    ax0.step(ts, y0, where="post", color="tab:green", linewidth=1.8, label="actual (label)")
    ax0.set_ylabel("playing")
    ax0.set_yticks([0.0, 1.0])
    ax0.set_yticklabels(["no", "yes"])
    ax0.set_ylim(-0.08, 1.08)
    ax0.grid(True, axis="x", alpha=0.3)
    ax0.legend(loc="upper right", fontsize=9)

    ax1.fill_between(ts, 0.0, y1, step="post", alpha=0.35, color="tab:blue", linewidth=0)
    ax1.step(ts, y1, where="post", color="tab:blue", linewidth=1.8, label=f"predicted ({pred_label})")
    ax1.set_ylabel("playing")
    ax1.set_yticks([0.0, 1.0])
    ax1.set_yticklabels(["no", "yes"])
    ax1.set_ylim(-0.08, 1.08)
    ax1.set_xlabel("time (seconds)")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    if np.any(mismatch):
        for t in ts[mismatch]:
            ax0.axvline(t, color="red", alpha=0.12, linewidth=0.8, zorder=0)
            ax1.axvline(t, color="red", alpha=0.12, linewidth=0.8, zorder=0)

    fig.suptitle(
        f"{stem}  (clips.id={int(args.clip_id)})  —  {n} sampled rows, "
        f"{n_err} frame disagreements ({100.0 * frac_err:.1f}%)  |  red tint: mismatch time"
        + (f"\n{note_lines[0]}" if note_lines else ""),
        fontsize=10,
        y=1.05,
    )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
