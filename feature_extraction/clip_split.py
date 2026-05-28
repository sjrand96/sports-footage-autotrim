"""*** TRAIN / TEST ASSIGNMENT — edit here when split logic changes ***

Replace ``assign_train_test`` when moving from the random placeholder to hand-curated
clip lists or grouped splits (e.g. by ``source_id``). ``job.py`` must call only this module.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from feature_extraction.core.clip_selection import ClipSpec

SplitName = Literal["train", "test"]
SPLIT_METHOD_PLACEHOLDER = "random_placeholder"
SPLIT_METHOD_SOURCE_MANIFEST = "source_level_manifest_v1"


def assign_train_test(
    clips: list[ClipSpec],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[ClipSpec], list[ClipSpec], dict[str, Any]]:
    """Partition clips into train and test lists (v1: random shuffle by clip)."""
    if not clips:
        return [], [], _split_metadata([], [], test_fraction=test_fraction, seed=seed)

    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    shuffled = list(clips)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    if len(shuffled) == 1:
        n_test = 0
    n_test = min(n_test, len(shuffled) - 1) if len(shuffled) > 1 else 0

    test_clips = shuffled[:n_test]
    train_clips = shuffled[n_test:]
    meta = _split_metadata(train_clips, test_clips, test_fraction=test_fraction, seed=seed)
    return train_clips, test_clips, meta


def assign_train_test_from_source_manifest(
    clips: list[ClipSpec],
    *,
    manifest_path: Path,
    eval_group: str = "test",
) -> tuple[list[ClipSpec], list[ClipSpec], dict[str, Any]]:
    """Partition clips by source_id using an explicit eval split manifest."""
    manifest = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
    train_sources, eval_sources = _source_groups_from_manifest(manifest, eval_group=eval_group)

    train_clips = sorted(
        [clip for clip in clips if clip.source_id in train_sources],
        key=lambda c: (c.source_id, c.clip_index),
    )
    test_clips = sorted(
        [clip for clip in clips if clip.source_id in eval_sources],
        key=lambda c: (c.source_id, c.clip_index),
    )

    if not train_clips:
        raise ValueError("source split manifest produced no train clips")
    if not test_clips:
        raise ValueError(f"source split manifest produced no {eval_group} clips")

    used_sources = {clip.source_id for clip in clips}
    missing_train_sources = sorted(train_sources - used_sources)
    missing_eval_sources = sorted(eval_sources - used_sources)
    meta = _source_split_metadata(
        manifest,
        manifest_path=manifest_path,
        eval_group=eval_group,
        train_clips=train_clips,
        test_clips=test_clips,
        missing_train_sources=missing_train_sources,
        missing_eval_sources=missing_eval_sources,
    )
    return train_clips, test_clips, meta


def _split_metadata(
    train_clips: list[ClipSpec],
    test_clips: list[ClipSpec],
    *,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "split_method": SPLIT_METHOD_PLACEHOLDER,
        "split_seed": int(seed),
        "test_fraction": float(test_fraction),
        "train_clip_ids": [c.clip_id for c in train_clips],
        "test_clip_ids": [c.clip_id for c in test_clips],
    }


def _source_groups_from_manifest(
    manifest: dict[str, Any],
    *,
    eval_group: str,
) -> tuple[set[str], set[str]]:
    train_sources = set(map(str, manifest.get("train_source_ids") or []))
    if eval_group == "test":
        eval_sources = set(map(str, manifest.get("test_source_ids") or []))
    elif eval_group in {"shift", "distribution_shift"}:
        eval_sources = set(map(str, manifest.get("distribution_shift_source_ids") or []))
    else:
        raise ValueError(f"eval_group must be 'test' or 'shift', got {eval_group!r}")

    if not train_sources:
        raise ValueError("split manifest has no train_source_ids")
    if not eval_sources:
        raise ValueError(f"split manifest has no sources for eval_group={eval_group!r}")

    optional_groups = [
        ("test_source_ids", set(map(str, manifest.get("test_source_ids") or []))),
        (
            "distribution_shift_source_ids",
            set(map(str, manifest.get("distribution_shift_source_ids") or [])),
        ),
        ("unlabeled_source_ids", set(map(str, manifest.get("unlabeled_source_ids") or []))),
    ]
    overlaps: list[str] = []
    for name, group in optional_groups:
        overlap = train_sources & group
        if overlap:
            overlaps.append(f"train_source_ids/{name}: " + ", ".join(sorted(overlap)))
    test_shift_overlap = optional_groups[0][1] & optional_groups[1][1]
    if test_shift_overlap:
        overlaps.append(
            "test_source_ids/distribution_shift_source_ids: "
            + ", ".join(sorted(test_shift_overlap))
        )
    if overlaps:
        raise ValueError("source split groups overlap: " + "; ".join(overlaps))
    return train_sources, eval_sources


def _source_split_metadata(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    eval_group: str,
    train_clips: list[ClipSpec],
    test_clips: list[ClipSpec],
    missing_train_sources: list[str],
    missing_eval_sources: list[str],
) -> dict[str, Any]:
    train_sources_used = sorted({c.source_id for c in train_clips})
    test_sources_used = sorted({c.source_id for c in test_clips})
    return {
        "split_method": str(manifest.get("split_method") or SPLIT_METHOD_SOURCE_MANIFEST),
        "split_manifest_path": str(manifest_path.expanduser().resolve()),
        "split_eval_group": eval_group,
        "split_seed": manifest.get("split_seed"),
        "test_fraction": manifest.get("test_fraction"),
        "train_source_ids": train_sources_used,
        "test_source_ids": test_sources_used,
        "distribution_shift_source_ids": list(
            map(str, manifest.get("distribution_shift_source_ids") or [])
        ),
        "unlabeled_source_ids": list(map(str, manifest.get("unlabeled_source_ids") or [])),
        "missing_train_source_ids": missing_train_sources,
        "missing_eval_source_ids": missing_eval_sources,
        "train_clip_ids": [c.clip_id for c in train_clips],
        "test_clip_ids": [c.clip_id for c in test_clips],
    }
