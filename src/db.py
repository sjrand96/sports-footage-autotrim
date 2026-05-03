"""Supabase client wrapper used by prep_videos.py and push_annotations.py."""

from __future__ import annotations

import os
from typing import Any

from supabase import Client, create_client


def get_supabase_client() -> Client:
    """Return a Supabase client using SUPABASE_URL and SUPABASE_SERVICE_KEY from env."""
    url = os.environ["SUPABASE_URL"].strip()
    key = os.environ["SUPABASE_SERVICE_KEY"].strip()
    return create_client(url, key)


def get_source_video(client: Client, source_id: str) -> dict[str, Any] | None:
    """Return the source_videos row for the given YouTube ID, or None."""
    res = client.table("source_videos").select("*").eq("id", source_id).execute()
    return res.data[0] if res.data else None


def upsert_source_video(
    client: Client,
    *,
    source_id: str,
    url: str,
    display_name: str | None,
    duration_sec: float | None,
    fps_original: float | None,
    downloaded_by: str | None,
) -> dict[str, Any]:
    """Insert or update a source_videos row. Returns the resulting row."""
    payload = {
        "id": source_id,
        "url": url,
        "display_name": display_name,
        "duration_sec": duration_sec,
        "fps_original": fps_original,
        "downloaded_by": downloaded_by,
    }
    res = client.table("source_videos").upsert(payload).execute()
    return res.data[0]


def get_clip(client: Client, source_id: str, clip_index: int) -> dict[str, Any] | None:
    """Return a clip row for (source_id, clip_index), or None."""
    res = (
        client.table("clips")
        .select("*")
        .eq("source_id", source_id)
        .eq("clip_index", clip_index)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_clip(
    client: Client,
    *,
    source_id: str,
    clip_index: int,
    filename: str,
    s3_bucket: str,
    s3_key: str,
    thumbnail_s3_key: str | None,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
) -> dict[str, Any]:
    """Insert or update a clips row keyed on (source_id, clip_index). Returns the row."""
    payload = {
        "source_id": source_id,
        "clip_index": clip_index,
        "filename": filename,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "thumbnail_s3_key": thumbnail_s3_key,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
    }
    res = (
        client.table("clips")
        .upsert(payload, on_conflict="source_id,clip_index")
        .execute()
    )
    return res.data[0]


def insert_annotation(
    client: Client,
    *,
    clip_id: int,
    label_studio_task_id: int | None,
    label_studio_project_id: int | None,
    annotator: str,
    lead_time_sec: float | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Insert an annotation row. Always inserts a new row (append-only)."""
    row = {
        "clip_id": clip_id,
        "label_studio_task_id": label_studio_task_id,
        "label_studio_project_id": label_studio_project_id,
        "annotator": annotator,
        "lead_time_sec": lead_time_sec,
        "payload": payload,
    }
    res = client.table("annotations").insert(row).execute()
    return res.data[0]


def annotation_exists_for_task(
    client: Client,
    *,
    clip_id: int,
    label_studio_task_id: int,
    annotator: str,
) -> bool:
    """True if an annotation row already exists for this clip, LS task, and annotator."""
    res = (
        client.table("annotations")
        .select("id")
        .eq("clip_id", clip_id)
        .eq("label_studio_task_id", label_studio_task_id)
        .eq("annotator", annotator)
        .limit(1)
        .execute()
    )
    return bool(res.data)
