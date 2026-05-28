#!/usr/bin/env python3
"""Create and validate source-level evaluation split manifests.

The segment evaluator needs complete videos in the held-out set. This helper
keeps that contract explicit: every source_id belongs to at most one group.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPLIT_METHOD = "source_level_manifest_v1"
DEFAULT_SPLIT_PATH = REPO_ROOT / "eval" / "source_video_split.json"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = raw.strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_default_source_ids() -> list[str]:
    from data.source_ids import SOURCE_IDS

    return list(SOURCE_IDS)


def build_source_split_manifest(
    source_ids: list[str],
    *,
    test_source_ids: list[str] | None = None,
    distribution_shift_source_ids: list[str] | None = None,
    unlabeled_source_ids: list[str] | None = None,
    test_fraction: float = 0.2,
    seed: int = 42,
    name: str = "source_video_split",
    notes: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable source-level split manifest."""
    all_sources = sorted(_dedupe_preserve_order(source_ids))
    if not all_sources:
        raise ValueError("source_ids is empty")

    shift = sorted(_dedupe_preserve_order(distribution_shift_source_ids or []))
    unlabeled = sorted(_dedupe_preserve_order(unlabeled_source_ids or []))
    reserved = set(shift) | set(unlabeled)

    unknown_reserved = sorted(reserved - set(all_sources))
    if unknown_reserved:
        raise ValueError(
            "reserved source IDs are not in source_ids: " + ", ".join(unknown_reserved)
        )

    candidate_sources = [sid for sid in all_sources if sid not in reserved]
    if test_source_ids is None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        shuffled = list(candidate_sources)
        random.Random(seed).shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_fraction))
        if len(shuffled) == 1:
            n_test = 0
        n_test = min(n_test, len(shuffled) - 1) if len(shuffled) > 1 else 0
        test = sorted(shuffled[:n_test])
    else:
        test = sorted(_dedupe_preserve_order(test_source_ids))

    unknown_test = sorted(set(test) - set(candidate_sources))
    if unknown_test:
        raise ValueError(
            "test source IDs are unknown or reserved: " + ", ".join(unknown_test)
        )

    train = sorted(sid for sid in candidate_sources if sid not in set(test))
    manifest: dict[str, Any] = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_method": SPLIT_METHOD,
        "split_seed": int(seed),
        "test_fraction": float(test_fraction),
        "source_ids": all_sources,
        "train_source_ids": train,
        "test_source_ids": test,
        "distribution_shift_source_ids": shift,
        "unlabeled_source_ids": unlabeled,
    }
    if notes:
        manifest["notes"] = notes
    validate_source_split_manifest(manifest)
    return manifest


def load_source_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.expanduser().read_text(encoding="utf-8"))
    validate_source_split_manifest(manifest)
    return manifest


def validate_source_split_manifest(manifest: dict[str, Any]) -> None:
    """Raise ValueError when split groups overlap or are malformed."""
    required = ("train_source_ids", "test_source_ids")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("split manifest missing keys: " + ", ".join(missing))

    groups = {
        "train_source_ids": set(map(str, manifest.get("train_source_ids") or [])),
        "test_source_ids": set(map(str, manifest.get("test_source_ids") or [])),
        "distribution_shift_source_ids": set(
            map(str, manifest.get("distribution_shift_source_ids") or [])
        ),
        "unlabeled_source_ids": set(map(str, manifest.get("unlabeled_source_ids") or [])),
    }
    for name, values in groups.items():
        if len(values) != len(manifest.get(name) or []):
            raise ValueError(f"{name} contains duplicate source IDs")

    names = list(groups)
    overlaps: list[str] = []
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = groups[left_name] & groups[right_name]
            if overlap:
                overlaps.append(
                    f"{left_name}/{right_name}: " + ", ".join(sorted(overlap))
                )
    if overlaps:
        raise ValueError("source split groups overlap: " + "; ".join(overlaps))

    if not groups["train_source_ids"]:
        raise ValueError("train_source_ids must not be empty")
    if not groups["test_source_ids"]:
        raise ValueError("test_source_ids must not be empty")


def source_groups_for_feature_eval(
    manifest: dict[str, Any],
    *,
    eval_group: str = "test",
) -> tuple[set[str], set[str]]:
    """Return ``(train_sources, eval_sources)`` for feature extraction."""
    validate_source_split_manifest(manifest)
    train = set(map(str, manifest["train_source_ids"]))
    if eval_group == "test":
        eval_sources = set(map(str, manifest["test_source_ids"]))
    elif eval_group in {"shift", "distribution_shift"}:
        eval_sources = set(map(str, manifest.get("distribution_shift_source_ids") or []))
    else:
        raise ValueError(f"eval_group must be 'test' or 'shift', got {eval_group!r}")
    if not eval_sources:
        raise ValueError(f"source split manifest has no sources for eval_group={eval_group!r}")
    return train, eval_sources


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_SPLIT_PATH)
    p.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="Source ID to include. Defaults to data.source_ids.SOURCE_IDS.",
    )
    p.add_argument("--test-source-id", action="append", default=None)
    p.add_argument("--shift-source-id", action="append", default=None)
    p.add_argument("--unlabeled-source-id", action="append", default=None)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="source_video_split")
    p.add_argument("--notes", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_ids = args.source_id if args.source_id is not None else _load_default_source_ids()
    manifest = build_source_split_manifest(
        source_ids,
        test_source_ids=args.test_source_id,
        distribution_shift_source_ids=args.shift_source_id,
        unlabeled_source_ids=args.unlabeled_source_id,
        test_fraction=args.test_fraction,
        seed=args.seed,
        name=args.name,
        notes=args.notes,
    )
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(
        "sources: "
        f"train={len(manifest['train_source_ids'])} "
        f"test={len(manifest['test_source_ids'])} "
        f"shift={len(manifest['distribution_shift_source_ids'])} "
        f"unlabeled={len(manifest['unlabeled_source_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
