"""Build deterministic per-video train/test clip lists from Supabase.

Writes ``data/train_clips.csv``, ``data/test_clips.csv``, and ``data/train-test-split-meta.json``.
Only clips with timeline labels in Supabase are included; unlabeled clips are skipped with a warning.
Edit ``data/source_ids.py`` for which sources to include. Re-runs are reproducible (seed 42).

Run::

    python data/train_test_split.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train_clips.csv"
TEST_CSV = DATA_DIR / "test_clips.csv"
META_JSON = DATA_DIR / "train_test_split_meta.json"

TEST_FRACTION = 0.2
SPLIT_SEED = 42


def assign_train_test_by_source(
    clips: list[dict[str, Any]],
    *,
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Random 80/20 split within each ``source_id`` (deterministic per video)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_source[str(clip["source_id"])].append(clip)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    per_source: dict[str, dict[str, int]] = {}

    for source_id in sorted(by_source):
        group = sorted(by_source[source_id], key=lambda c: int(c["clip_index"]))
        shuffled = list(group)
        subseed = int.from_bytes(
            hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:8],
            "big",
        )
        random.Random(subseed).shuffle(shuffled)

        n = len(shuffled)
        if n == 1:
            n_test = 0
        else:
            n_test = max(1, round(n * test_fraction))
            n_test = min(n_test, n - 1)

        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
        per_source[source_id] = {"total": n, "train": n - n_test, "test": n_test}

    key = lambda c: (str(c["source_id"]), int(c["clip_index"]))
    train.sort(key=key)
    test.sort(key=key)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "split_method": "per_source_id_random",
        "split_seed": seed,
        "test_fraction": test_fraction,
        "source_ids": sorted(by_source),
        "per_source": per_source,
        "train_clips": len(train),
        "test_clips": len(test),
        "train_clip_ids": [c["clip_id"] for c in train],
        "test_clip_ids": [c["clip_id"] for c in test],
    }
    return train, test, meta


def train_test_split() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from data.fetch_data import (
        _get_db_helpers,
        ensure_env_loaded,
        fetch_latest_annotation,
        list_clips_for_source,
    )
    from data.source_ids import SOURCE_IDS

    if not SOURCE_IDS:
        raise RuntimeError("SOURCE_IDS is empty; edit data/source_ids.py")

    ensure_env_loaded()
    client = _get_db_helpers().get_supabase_client()

    clips: list[dict[str, Any]] = []
    for source_id in SOURCE_IDS:
        rows = list_clips_for_source(client, source_id)
        if not rows:
            raise RuntimeError(f"no clips in Supabase for source_id={source_id!r}")
        for row in rows:
            sid = str(row["source_id"])
            idx = int(row["clip_index"])
            clip_id = f"{sid}_{idx:03d}"
            ann = fetch_latest_annotation(client, clip_id=int(row["id"]))
            if ann is None or not isinstance(ann.get("payload"), dict):
                print(f"WARN: no labels in Supabase for {clip_id}, skipping")
                continue
            clips.append(
                {
                    "clip_id": clip_id,
                    "source_id": sid,
                    "clip_index": idx,
                }
            )

    if not clips:
        raise RuntimeError("no clips with labels in Supabase for any source_id")

    train, test, meta = assign_train_test_by_source(clips)

    for path, rows in ((TRAIN_CSV, train), (TEST_CSV, test)):
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows([[c["clip_id"]] for c in rows])

    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train clips: {len(train)} -> {TRAIN_CSV.relative_to(REPO_ROOT)}")
    print(f"test clips:  {len(test)} -> {TEST_CSV.relative_to(REPO_ROOT)}")
    print(f"meta:        {META_JSON.relative_to(REPO_ROOT)}")
    for source_id, counts in sorted(meta["per_source"].items()):
        print(
            f"  {source_id}: {counts['train']} train / {counts['test']} test "
            f"(total {counts['total']})"
        )


if __name__ == "__main__":
    try:
        train_test_split()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
