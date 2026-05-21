"""Resolve eligible clips from Supabase (annotated + court calibration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClipSpec:
    clip_id: int
    source_id: str
    clip_index: int
    s3_bucket: str
    s3_key: str

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_key}"


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def _fetch_all_rows(client: Any, table: str, select_cols: str, *, page_size: int = 1000) -> list[dict[str, Any]]:
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


def fetch_annotated_clip_keys(db_client: Any) -> set[tuple[str, int]]:
    ann_rows = _fetch_all_rows(db_client, "annotations", "clip_id")
    clip_ids = sorted({int(r["clip_id"]) for r in ann_rows if r.get("clip_id") is not None})
    if not clip_ids:
        return set()

    keys: set[tuple[str, int]] = set()
    for chunk in _chunked(clip_ids, chunk_size=500):
        res = db_client.table("clips").select("id,source_id,clip_index").in_("id", chunk).execute()
        for row in res.data or []:
            sid = row.get("source_id")
            cidx = row.get("clip_index")
            if sid is None or cidx is None:
                continue
            keys.add((str(sid), int(cidx)))
    return keys


def list_eligible_clips(db_helpers: Any, db_client: Any) -> list[ClipSpec]:
    """Clips with timeline annotation and ``court_calibrations`` for their ``source_id``."""
    annotated = fetch_annotated_clip_keys(db_client)
    calibrated = db_helpers.list_court_calibration_source_ids(db_client)
    eligible_keys = {k for k in annotated if k[0] in calibrated}
    if not eligible_keys:
        return []

    clip_ids = _clip_ids_from_keys(db_client, eligible_keys)
    specs: list[ClipSpec] = []
    for chunk in _chunked(clip_ids, chunk_size=500):
        res = (
            db_client.table("clips")
            .select("id,source_id,clip_index,s3_bucket,s3_key")
            .in_("id", chunk)
            .execute()
        )
        for row in res.data or []:
            sid = str(row["source_id"])
            cidx = int(row["clip_index"])
            if (sid, cidx) not in eligible_keys:
                continue
            specs.append(
                ClipSpec(
                    clip_id=int(row["id"]),
                    source_id=sid,
                    clip_index=cidx,
                    s3_bucket=str(row["s3_bucket"]),
                    s3_key=str(row["s3_key"]),
                )
            )
    specs.sort(key=lambda c: (c.source_id, c.clip_index))
    return specs


def _clip_ids_from_keys(db_client: Any, keys: set[tuple[str, int]]) -> list[int]:
    ids: list[int] = []
    source_ids = sorted({k[0] for k in keys})
    for sid in source_ids:
        res = db_client.table("clips").select("id,source_id,clip_index").eq("source_id", sid).execute()
        for row in res.data or []:
            if (str(row["source_id"]), int(row["clip_index"])) in keys:
                ids.append(int(row["id"]))
    return ids


def clip_spec_by_id(db_helpers: Any, db_client: Any, clip_id: int) -> ClipSpec | None:
    row = db_helpers.get_clip_by_id(db_client, clip_id)
    if row is None:
        return None
    spec = ClipSpec(
        clip_id=int(row["id"]),
        source_id=str(row["source_id"]),
        clip_index=int(row["clip_index"]),
        s3_bucket=str(row["s3_bucket"]),
        s3_key=str(row["s3_key"]),
    )
    annotated = fetch_annotated_clip_keys(db_client)
    calibrated = db_helpers.list_court_calibration_source_ids(db_client)
    if (spec.source_id, spec.clip_index) not in annotated:
        return None
    if spec.source_id not in calibrated:
        return None
    return spec


def filter_eligible(specs: list[ClipSpec], db_helpers: Any, db_client: Any) -> list[ClipSpec]:
    annotated = fetch_annotated_clip_keys(db_client)
    calibrated = db_helpers.list_court_calibration_source_ids(db_client)
    return [
        s
        for s in specs
        if (s.source_id, s.clip_index) in annotated and s.source_id in calibrated
    ]
