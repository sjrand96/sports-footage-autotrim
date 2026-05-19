"""Feature extraction core library."""

from feature_extraction.core.clip_selection import ClipSpec, clip_spec_by_id, list_eligible_clips
from feature_extraction.core.feature_columns import FEATURE_COLUMNS, PARQUET_COLUMNS
from feature_extraction.core.version import EXTRACTOR_VERSION, FEATURE_SCHEMA_VERSION

__all__ = [
    "ClipSpec",
    "EXTRACTOR_VERSION",
    "FEATURE_COLUMNS",
    "FEATURE_SCHEMA_VERSION",
    "PARQUET_COLUMNS",
    "clip_spec_by_id",
    "list_eligible_clips",
]
