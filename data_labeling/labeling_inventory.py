"""Print labeling coverage from Supabase (sources, clips, homography, pipeline eligibility).

Usage:
    python data_labeling/labeling_inventory.py
    python data_labeling/labeling_inventory.py --json
    python data_labeling/labeling_inventory.py --test-fraction 0.2 --split-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LABEL_FPS = 30.0


def _fetch_all(client: Any, table: str, select_cols: str, *, page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        res = client.table(table).select(select_cols).range(offset, offset + page_size - 1).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _playing_range_count(payload: dict[str, Any], label_fps: float) -> int:
    """Count Playing timeline ranges in a push_timeline_annotation-style payload."""
    n = 0
    ann = payload.get("label_studio_annotation")
    if not isinstance(ann, dict):
        return 0
    result = ann.get("result")
    if not isinstance(result, list):
        return 0
    for item in result:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        if value.get("timelinelabels") != ["Playing"]:
            continue
        item_ranges = value.get("ranges")
        if not isinstance(item_ranges, list):
            continue
        for r in item_ranges:
            if isinstance(r, dict) and r.get("start") is not None and r.get("end") is not None:
                n += 1
    return n


def gather_inventory(
    *,
    test_fraction: float,
    split_seed: int,
    label_fps: float,
) -> dict[str, Any]:
    from src import db
    from feature_extraction.clip_split import assign_train_test
    from feature_extraction.core.clip_selection import fetch_annotated_clip_keys, list_eligible_clips

    client = db.get_supabase_client()

    sources = _fetch_all(client, "source_videos", "id,display_name,duration_sec")
    clips = _fetch_all(client, "clips", "id,source_id,clip_index")
    anns = _fetch_all(client, "annotations", "clip_id,payload,exported_at")
    calibs = _fetch_all(client, "court_calibrations", "source_id")

    annotated_keys = fetch_annotated_clip_keys(client)
    calibrated_ids = db.list_court_calibration_source_ids(client)
    eligible = list_eligible_clips(db, client)
    train_clips, test_clips, split_meta = assign_train_test(
        eligible, test_fraction=test_fraction, seed=split_seed
    )

    ann_clip_ids = {int(r["clip_id"]) for r in anns if r.get("clip_id") is not None}
    clip_by_id = {int(c["id"]): c for c in clips}
    annotated_sources = sorted(
        {str(clip_by_id[cid]["source_id"]) for cid in ann_clip_ids if cid in clip_by_id}
    )

    latest_ann: dict[int, dict[str, Any]] = {}
    for r in anns:
        cid = r.get("clip_id")
        if cid is None:
            continue
        cid = int(cid)
        if cid not in latest_ann or str(r.get("exported_at") or "") > str(
            latest_ann[cid].get("exported_at") or ""
        ):
            latest_ann[cid] = r

    with_playing = 0
    without_playing = 0
    for row in latest_ann.values():
        payload = row.get("payload")
        if not isinstance(payload, dict):
            without_playing += 1
            continue
        if _playing_range_count(payload, label_fps) > 0:
            with_playing += 1
        else:
            without_playing += 1

    clips_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in clips:
        clips_by_source[str(c["source_id"])].append(c)

    annotated_by_source: dict[str, set[int]] = defaultdict(set)
    for sid, idx in annotated_keys:
        annotated_by_source[sid].add(idx)

    name_by_id = {str(s["id"]): s.get("display_name") or str(s["id"]) for s in sources}

    per_source: list[dict[str, Any]] = []
    for sid in sorted(clips_by_source):
        total = len(clips_by_source[sid])
        ann_n = len(annotated_by_source[sid])
        has_cal = sid in calibrated_ids
        eligible_n = ann_n if has_cal else 0
        per_source.append(
            {
                "source_id": sid,
                "display_name": name_by_id.get(sid, sid),
                "clips_total": total,
                "clips_annotated": ann_n,
                "has_homography": has_cal,
                "clips_pipeline_eligible": eligible_n,
            }
        )

    missing_calib_sources = sorted(
        {sid for sid, _ in annotated_keys if sid not in calibrated_ids}
    )
    calibrated_no_ann = sorted(calibrated_ids - set(annotated_sources))

    return {
        "totals": {
            "sources_ingested": len(sources),
            "clips_total": len(clips),
            "sources_with_homography": len(calibrated_ids),
            "annotation_rows": len(anns),
            "clips_with_any_annotation": len(ann_clip_ids),
            "clips_annotated_distinct": len(annotated_keys),
            "sources_with_any_annotated_clip": len(annotated_sources),
            "clips_pipeline_eligible": len(eligible),
            "clips_latest_export_with_playing_ranges": with_playing,
            "clips_latest_export_without_playing_ranges": without_playing,
        },
        "train_test_placeholder": {
            "test_fraction": test_fraction,
            "split_seed": split_seed,
            "split_method": split_meta.get("split_method"),
            "train_clips": len(train_clips),
            "test_clips": len(test_clips),
        },
        "gaps": {
            "sources_ingested_without_homography": len(
                set(clips_by_source) - calibrated_ids
            ),
            "sources_with_annotated_clips_missing_homography": len(missing_calib_sources),
            "source_ids_with_annotated_clips_missing_homography": missing_calib_sources,
            "sources_with_homography_zero_annotated_clips": len(calibrated_no_ann),
            "source_ids_with_homography_zero_annotated_clips": calibrated_no_ann,
        },
        "per_source": per_source,
    }


def _print_report(data: dict[str, Any]) -> None:
    t = data["totals"]
    tt = data["train_test_placeholder"]
    g = data["gaps"]

    print("=== Labeling inventory (Supabase) ===\n")
    print("Totals")
    print(f"  Sources ingested:              {t['sources_ingested']}")
    print(f"  Clips in DB:                   {t['clips_total']}")
    print(f"  Sources with homography:       {t['sources_with_homography']}")
    print(f"  Annotation rows (append log):  {t['annotation_rows']}")
    print(f"  Clips with ≥1 annotation:      {t['clips_with_any_annotation']}")
    print(f"  Sources with ≥1 annotated clip:{t['sources_with_any_annotated_clip']}")
    print()
    print("Pipeline-ready (timeline + homography per source)")
    print(f"  Eligible clips:                {t['clips_pipeline_eligible']}")
    print(
        f"  Placeholder train/test:        {tt['train_clips']} train / {tt['test_clips']} test "
        f"(fraction={tt['test_fraction']}, seed={tt['split_seed']})"
    )
    print()
    print("Timeline quality (latest export per clip)")
    print(f"  With Playing ranges:           {t['clips_latest_export_with_playing_ranges']}")
    print(f"  Without Playing ranges:        {t['clips_latest_export_without_playing_ranges']}")
    print()
    print("Gaps")
    print(f"  Ingested sources, no homography: {g['sources_ingested_without_homography']}")
    print(
        f"  Annotated sources, no homography: {g['sources_with_annotated_clips_missing_homography']}"
    )
    if g["source_ids_with_annotated_clips_missing_homography"]:
        ids = ", ".join(g["source_ids_with_annotated_clips_missing_homography"])
        print(f"    → {ids}")
    if g["sources_with_homography_zero_annotated_clips"]:
        print(
            f"  Homography but zero annotated clips: {g['sources_with_homography_zero_annotated_clips']}"
        )

    print("\nPer source")
    print(f"  {'source_id':<14} {'cal':^3} {'clips':>5} {'ann':>5} {'eligible':>8}  display_name")
    print(f"  {'-' * 14} {'-' * 3} {'-' * 5} {'-' * 5} {'-' * 8}  {'-' * 24}")
    for row in data["per_source"]:
        cal = "yes" if row["has_homography"] else "no"
        name = (row["display_name"] or "")[:40]
        print(
            f"  {row['source_id']:<14} {cal:^3} {row['clips_total']:5d} "
            f"{row['clips_annotated']:5d} {row['clips_pipeline_eligible']:8d}  {name}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize labeling coverage in Supabase.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test fraction for placeholder train/test counts (default: 0.2)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for placeholder train/test (default: 42)",
    )
    parser.add_argument(
        "--label-fps",
        type=float,
        default=DEFAULT_LABEL_FPS,
        help="Label Studio timeline FPS for Playing-range check (default: 30)",
    )
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        print("ERROR: --test-fraction must be in (0, 1)", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[misc, assignment]

    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    missing = [k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY") if not __import__("os").environ.get(k)]
    if missing:
        print(f"ERROR: missing env: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        data = gather_inventory(
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            label_fps=args.label_fps,
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_report(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
