#!/usr/bin/env python3
"""Grouped per-fold F2 bar chart: within-video (clip CV) vs novel-video (source CV)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO_ROOT
    / "models/tabular_xgb/cv/all_clips_features_v2-xgb-5fold/plots/per_fold_f2_within_vs_novel.jpg"
)

# 11-source eval set; novel from cv_summary.json, within from clip-grouped 5-fold (seed=42).
FOLDS = np.arange(1, 6)
WITHIN_F2 = np.array([0.846, 0.848, 0.853, 0.852, 0.846])
NOVEL_F2 = np.array([0.832, 0.848, 0.845, 0.774, 0.825])

COLOR_WITHIN = "#4A7FD4"  # blue
COLOR_NOVEL = "#E8B923"  # gold / yellow


def plot_per_fold_f2(*, out_path: Path, dpi: int = 150) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    x = np.arange(len(FOLDS), dtype=float)
    width = 0.36

    ax.bar(
        x - width / 2,
        WITHIN_F2,
        width,
        label="Within-video",
        color=COLOR_WITHIN,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        NOVEL_F2,
        width,
        label="Novel-video",
        color=COLOR_NOVEL,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_title("PER-FOLD F2", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("EXAMPLE FOLDS", fontsize=11, labelpad=8)
    ax.set_ylabel("F2", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(f)) for f in FOLDS])
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#CCCCCC", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        columnspacing=2.0,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        format=out_path.suffix.lstrip(".") or "png",
    )
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    path = plot_per_fold_f2(out_path=args.out.resolve(), dpi=args.dpi)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
