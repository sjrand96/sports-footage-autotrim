#!/usr/bin/env python3
"""Split a window manifest into train/val/test by group key."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _group_split(
    rows: List[Dict[str, Any]],
    field: str,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get(field) or row.get("match_id") or row.get("source_id") or "unknown"
        groups.setdefault(str(key), []).append(row)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for key, group_rows in groups.items():
        if key in train_keys:
            train_rows.extend(group_rows)
        elif key in val_keys:
            val_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)

    return train_rows, val_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split window manifest by group key.")
    parser.add_argument("--input", required=True, help="Window manifest JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for split manifests.")
    parser.add_argument("--split-field", default="source_id", help="Field to split on.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    args = parser.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    rows = _load_jsonl(args.input)
    train_rows, val_rows, test_rows = _group_split(rows, args.split_field, ratios, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    _write_jsonl(train_rows, train_path)
    _write_jsonl(val_rows, val_path)
    _write_jsonl(test_rows, test_path)

    print(f"Train={len(train_rows)} Val={len(val_rows)} Test={len(test_rows)}")


if __name__ == "__main__":
    main()
