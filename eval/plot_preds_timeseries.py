#!/usr/bin/env python3
"""Plot ground-truth vs predicted playing labels over time (test-set CSV)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

# Timeline strips: filled = playing (1), light gray = downtime (0).
CMAP_GT = ListedColormap(["#f0f0f0", "#2ca02c"])
CMAP_PRED = ListedColormap(["#f0f0f0", "#d62728"])

REQUIRED_COLS = (
    "clip_key",
    "frame_idx",
    "is_playing",
    "pred_playing",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-series plot of is_playing vs predictions.")
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="xgb_test_preds.csv from train.py --save-test-csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: next to CSV, name derived from mode).",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--clip",
        type=str,
        default=None,
        help="Single clip_key (e.g. 1rXZJyVXUHU_003).",
    )
    mode.add_argument(
        "--source",
        type=str,
        default=None,
        help="All clips for one source_id / video, concatenated on one panel.",
    )
    mode.add_argument(
        "--all-sources",
        action="store_true",
        help="One subplot per source_id (video), clips concatenated.",
    )
    mode.add_argument(
        "--all-clips",
        action="store_true",
        help="One subplot per clip_key (can be tall; use --max-panels).",
    )
    p.add_argument(
        "--max-panels",
        type=int,
        default=None,
        help="Cap number of subplots for --all-sources / --all-clips.",
    )
    p.add_argument(
        "--style",
        choices=("strips", "lines"),
        default="strips",
        help="strips = two timeline bands (default); lines = overlaid step plots.",
    )
    p.add_argument(
        "--show-prob",
        action="store_true",
        help="Also plot prob_playing (only with --style lines).",
    )
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window (in addition to saving).",
    )
    return p.parse_args()


def load_preds_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns {missing}: {path}")
    if "prob_playing" not in df.columns:
        df["prob_playing"] = float("nan")
    return df


def _parse_clip_key(clip_key: str) -> tuple[str, int]:
    base, idx = clip_key.rsplit("_", 1)
    return base, int(idx)


def _clip_sort_key(clip_key: str) -> tuple[str, int]:
    return _parse_clip_key(clip_key)


def _concat_clips(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate clips on a single timeline (cumulative frame offset)."""
    parts: list[pd.DataFrame] = []
    boundaries: list[int] = []
    offset = 0
    for chunk in frames:
        part = chunk.sort_values("frame_idx", kind="stable").copy()
        if offset > 0:
            boundaries.append(offset)
        part["plot_x"] = part["frame_idx"].astype(int) + offset
        offset = int(part["plot_x"].max()) + 1
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    combined.attrs["clip_boundaries"] = boundaries
    return combined


def _x_extent(df: pd.DataFrame, x_col: str) -> tuple[float, float]:
    x = df[x_col].to_numpy(dtype=float)
    if len(x) == 0:
        return 0.0, 1.0
    if len(x) == 1:
        return float(x[0]) - 0.5, float(x[0]) + 0.5
    dx = float(np.median(np.diff(x)))
    if dx <= 0:
        dx = 1.0
    return float(x[0]) - dx / 2, float(x[-1]) + dx / 2


def _plot_panel_strips(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    x_col: str = "frame_idx",
    mark_clip_boundaries: list[int] | None = None,
) -> None:
    """Two stacked timeline bands — compare vertical alignment frame-by-frame."""
    gt = df["is_playing"].to_numpy(dtype=float)
    pred = df["pred_playing"].to_numpy(dtype=float)
    xmin, xmax = _x_extent(df, x_col)

    ax.imshow(
        gt[np.newaxis, :],
        aspect="auto",
        extent=[xmin, xmax, 1.0, 2.0],
        cmap=CMAP_GT,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    ax.imshow(
        pred[np.newaxis, :],
        aspect="auto",
        extent=[xmin, xmax, 0.0, 1.0],
        cmap=CMAP_PRED,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        origin="lower",
    )
    if mark_clip_boundaries:
        for boundary in mark_clip_boundaries:
            ax.axvline(boundary, color="#444444", linewidth=0.7, linestyle=":", ymin=0, ymax=1)

    ax.set_ylim(0, 2)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["prediction", "ground truth"], fontsize=8)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(False)


def _plot_panel_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    x_col: str = "frame_idx",
    show_prob: bool = False,
    mark_clip_boundaries: list[int] | None = None,
) -> None:
    x = df[x_col].to_numpy()
    # Prediction first (thin); ground truth on top (thick green).
    if show_prob and df["prob_playing"].notna().any():
        ax.plot(
            x,
            df["prob_playing"],
            label="P(playing)",
            color="#1f77b4",
            linewidth=0.7,
            alpha=0.4,
            zorder=1,
        )
    ax.plot(
        x,
        df["pred_playing"],
        label="prediction",
        color="#d62728",
        linewidth=0.6,
        drawstyle="steps-post",
        zorder=2,
    )
    ax.plot(
        x,
        df["is_playing"],
        label="ground truth",
        color="#2ca02c",
        linewidth=3.5,
        drawstyle="steps-post",
        zorder=3,
    )
    if mark_clip_boundaries:
        for boundary in mark_clip_boundaries:
            ax.axvline(boundary, color="#888888", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["downtime", "playing"])
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    style: str,
    x_col: str = "frame_idx",
    show_prob: bool = False,
    mark_clip_boundaries: list[int] | None = None,
) -> None:
    if style == "strips":
        _plot_panel_strips(ax, df, title=title, x_col=x_col, mark_clip_boundaries=mark_clip_boundaries)
    else:
        _plot_panel_lines(
            ax,
            df,
            title=title,
            x_col=x_col,
            show_prob=show_prob,
            mark_clip_boundaries=mark_clip_boundaries,
        )


def plot_single_clip(
    df: pd.DataFrame,
    clip_key: str,
    *,
    style: str,
    show_prob: bool,
) -> plt.Figure:
    clip_df = df[df["clip_key"] == clip_key].sort_values("frame_idx", kind="stable")
    if clip_df.empty:
        known = ", ".join(sorted(df["clip_key"].unique()[:8]))
        raise SystemExit(f"clip_key not found: {clip_key!r}. Examples: {known}…")

    fig_h = 2.0 if style == "strips" else 3.0
    fig, ax = plt.subplots(figsize=(12, fig_h))
    uri = clip_df["clip_s3_uri"].iloc[0] if "clip_s3_uri" in clip_df.columns else ""
    title = f"{clip_key}" + (f"\n{uri}" if isinstance(uri, str) and uri else "")
    _plot_panel(ax, clip_df, title=title, style=style, show_prob=show_prob)
    ax.set_xlabel("frame_idx")
    fig.tight_layout()
    return fig


def _panels_for_clips(df: pd.DataFrame, clip_keys: list[str]) -> list[tuple[str, pd.DataFrame]]:
    return [(key, df[df["clip_key"] == key].sort_values("frame_idx", kind="stable")) for key in clip_keys]


def _panels_for_sources(df: pd.DataFrame, source_ids: list[str]) -> list[tuple[str, pd.DataFrame]]:
    panels: list[tuple[str, pd.DataFrame]] = []
    for source_id in source_ids:
        keys = sorted(
            [k for k in df["clip_key"].unique() if k.startswith(f"{source_id}_")],
            key=_clip_sort_key,
        )
        if not keys:
            keys = sorted(
                [k for k in df["clip_key"].unique() if _parse_clip_key(k)[0] == source_id],
                key=_clip_sort_key,
            )
        frames = [df[df["clip_key"] == k] for k in keys]
        panels.append((source_id, _concat_clips(frames)))
    return panels


def plot_stacked_panels(
    panels: list[tuple[str, pd.DataFrame]],
    *,
    style: str,
    show_prob: bool,
    suptitle: str,
) -> plt.Figure:
    n = len(panels)
    if n == 0:
        raise SystemExit("no panels to plot")

    row_h = 1.15 if style == "strips" else 2.2
    fig_h = max(row_h * n, 2.5 if style == "strips" else 3.0)
    fig, axes = plt.subplots(n, 1, figsize=(12, fig_h), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (title, panel_df) in zip(axes, panels, strict=True):
        x_col = "plot_x" if "plot_x" in panel_df.columns else "frame_idx"
        boundaries = panel_df.attrs.get("clip_boundaries", [])
        _plot_panel(
            ax,
            panel_df,
            title=title,
            style=style,
            x_col=x_col,
            show_prob=show_prob,
            mark_clip_boundaries=boundaries,
        )
        if style == "lines":
            ax.set_ylabel("")

    axes[-1].set_xlabel("frame_idx (concatenated across clips when grouped by source)")
    fig.suptitle(suptitle, fontsize=11, y=1.002)
    fig.tight_layout()
    return fig


def list_clip_keys(df: pd.DataFrame) -> list[str]:
    return sorted(df["clip_key"].unique(), key=_clip_sort_key)


def list_source_ids(df: pd.DataFrame) -> list[str]:
    return sorted({_parse_clip_key(k)[0] for k in df["clip_key"].unique()})


def default_out_path(csv_path: Path, args: argparse.Namespace) -> Path:
    stem = csv_path.stem
    if args.clip:
        safe = re.sub(r"[^\w.-]+", "_", args.clip)
        return csv_path.with_name(f"{stem}_clip_{safe}.png")
    if args.source:
        safe = re.sub(r"[^\w.-]+", "_", args.source)
        return csv_path.with_name(f"{stem}_source_{safe}.png")
    if args.all_sources:
        return csv_path.with_name(f"{stem}_all_sources.png")
    return csv_path.with_name(f"{stem}_all_clips.png")


def main() -> int:
    args = parse_args()
    df = load_preds_csv(args.csv.expanduser().resolve())

    if args.show_prob and args.style != "lines":
        raise SystemExit("--show-prob only applies with --style lines")

    if args.clip:
        fig = plot_single_clip(df, args.clip, style=args.style, show_prob=args.show_prob)
    elif args.source:
        panels = _panels_for_sources(df, [args.source])
        if not panels or panels[0][1].empty:
            raise SystemExit(f"no rows for source {args.source!r}")
        fig = plot_stacked_panels(
            panels,
            style=args.style,
            show_prob=args.show_prob,
            suptitle=f"source_id={args.source}",
        )
    elif args.all_sources:
        sources = list_source_ids(df)
        if args.max_panels is not None:
            sources = sources[: args.max_panels]
        panels = _panels_for_sources(df, sources)
        fig = plot_stacked_panels(
            panels,
            style=args.style,
            show_prob=args.show_prob,
            suptitle="Test set — one row per video (source_id)",
        )
    else:
        clips = list_clip_keys(df)
        if args.max_panels is not None:
            clips = clips[: args.max_panels]
        panels = _panels_for_clips(df, clips)
        fig = plot_stacked_panels(
            panels,
            style=args.style,
            show_prob=args.show_prob,
            suptitle="Test set — one row per clip",
        )

    out_path = (args.out or default_out_path(args.csv, args)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())