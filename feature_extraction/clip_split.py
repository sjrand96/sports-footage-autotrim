"""*** TRAIN / TEST ASSIGNMENT — edit here when split logic changes ***

Replace ``assign_train_test`` when moving from the random placeholder to hand-curated
clip lists or grouped splits (e.g. by ``source_id``). ``job.py`` must call only this module.
"""

from __future__ import annotations

import random
from typing import Any, Literal

from feature_extraction.core.clip_selection import ClipSpec

SplitName = Literal["train", "test"]
SPLIT_METHOD_PLACEHOLDER = "random_placeholder"


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
