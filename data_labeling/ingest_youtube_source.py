"""Download a YouTube source video, segment into 1-minute 30fps clips, generate
thumbnails, upload to S3, and upsert `source_videos` + `clips` in Supabase.

Usage:
    python data_labeling/ingest_youtube_source.py <youtube_url> [--display-name "Game name"]
                                                   [--force]
                                                   [--software-encode]

Environment variables required (from .env):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION
    S3_BUCKET
    SUPABASE_URL
    SUPABASE_SERVICE_KEY

Idempotent: skips work that's already been done (download present locally,
clip already on S3 with matching size, etc.). Use --force to redo all steps.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WORKDIR_ROOT = Path("./workdir")
CLIP_DURATION_SEC = 60
TARGET_FPS = 30
THUMBNAIL_OFFSET_FRACTION = 0.5  # middle of clip
YT_DLP_FORMAT = "299+140"  # 1080p60 H.264 + m4a audio (strict)


@dataclass
class ClipInfo:
    index: int
    local_path: Path
    thumbnail_path: Path
    start_sec: float
    end_sec: float
    duration_sec: float

    @property
    def filename(self) -> str:
        return self.local_path.name

    @property
    def thumbnail_filename(self) -> str:
        return self.thumbnail_path.name


def log(msg: str) -> None:
    print(msg, flush=True)


def step(label: str) -> None:
    log(f"\n=== {label} ===")


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    return subprocess.run(cmd, check=True)


def extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract a YouTube video ID from: {url}")


def has_videotoolbox() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "h264_videotoolbox" in result.stdout


def ffprobe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture=True,
    )
    return float(result.stdout.strip())


def ffprobe_fps(path: Path) -> float | None:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture=True,
    )
    raw = result.stdout.strip()
    if "/" in raw:
        num, denom = raw.split("/")
        try:
            return float(num) / float(denom) if float(denom) else None
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def write_middle_frame_thumbnail(clip_path: Path, duration: float) -> Path:
    thumb_path = clip_path.with_suffix(".jpg")
    thumb_offset = duration * THUMBNAIL_OFFSET_FRACTION
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{thumb_offset:.2f}",
            "-i",
            str(clip_path),
            "-vframes",
            "1",
            "-q:v",
            "3",
            str(thumb_path),
        ],
        capture=True,
    )
    return thumb_path


def download_source(url: str, source_id: str, dest: Path, force: bool) -> Path:
    output_path = dest / f"{source_id}.mp4"
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log(f"Source already downloaded ({size_mb:.1f} MB), skipping. (use --force to redo)")
        return output_path

    log(f"Downloading from {url}")
    log(f"Format: {YT_DLP_FORMAT} (1080p60 H.264 + AAC audio, strict)")

    cmd = [
        "yt-dlp",
        "-f",
        YT_DLP_FORMAT,
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ]
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"yt-dlp failed (likely missing format {YT_DLP_FORMAT} for this video).\n"
            f"To see available formats, run: yt-dlp -F '{url}'\n"
            "This video may not have a 1080p60 H.264 stream — try a different one."
        ) from e

    return output_path


def segment_and_thumbnail(
    source_path: Path,
    source_id: str,
    clips_dir: Path,
    *,
    use_videotoolbox: bool,
    force: bool,
) -> list[ClipInfo]:
    existing = sorted(clips_dir.glob(f"{source_id}_*.mp4"))
    if existing and not force:
        log(f"Found {len(existing)} existing clips locally, reusing. (use --force to redo)")
        clips = []
        for clip_path in existing:
            idx = int(clip_path.stem.rsplit("_", 1)[1])
            duration = ffprobe_duration(clip_path)
            thumb = clip_path.with_suffix(".jpg")
            if not thumb.exists():
                log(f"  Missing thumbnail for {clip_path.name}, regenerating")
                thumb = write_middle_frame_thumbnail(clip_path, duration)
            clips.append(
                ClipInfo(
                    index=idx,
                    local_path=clip_path,
                    thumbnail_path=thumb,
                    start_sec=(idx - 1) * CLIP_DURATION_SEC,
                    end_sec=(idx - 1) * CLIP_DURATION_SEC + duration,
                    duration_sec=duration,
                )
            )
        return clips

    if clips_dir.exists():
        for f in clips_dir.glob(f"{source_id}_*"):
            f.unlink()
    clips_dir.mkdir(parents=True, exist_ok=True)

    if use_videotoolbox:
        log("Using hardware-accelerated encoder: h264_videotoolbox")
        video_codec_args = [
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "5M",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        log("Using software encoder: libx264")
        video_codec_args = [
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.0",
            "-pix_fmt",
            "yuv420p",
        ]

    output_pattern = str(clips_dir / f"{source_id}_%03d.mp4")
    log(f"Segmenting into {CLIP_DURATION_SEC}s clips at {TARGET_FPS}fps")

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            *video_codec_args,
            "-r",
            str(TARGET_FPS),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "segment",
            "-segment_time",
            str(CLIP_DURATION_SEC),
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            "1",
            output_pattern,
        ]
    )

    clip_paths = sorted(clips_dir.glob(f"{source_id}_*.mp4"))
    log(f"Generated {len(clip_paths)} clips")

    log("Generating thumbnails (middle frame per clip)")
    clips: list[ClipInfo] = []
    for clip_path in clip_paths:
        idx = int(clip_path.stem.rsplit("_", 1)[1])
        duration = ffprobe_duration(clip_path)
        thumb_path = write_middle_frame_thumbnail(clip_path, duration)
        clips.append(
            ClipInfo(
                index=idx,
                local_path=clip_path,
                thumbnail_path=thumb_path,
                start_sec=(idx - 1) * CLIP_DURATION_SEC,
                end_sec=(idx - 1) * CLIP_DURATION_SEC + duration,
                duration_sec=duration,
            )
        )
    return clips


def s3_object_size(s3, bucket: str, key: str) -> int | None:
    from botocore.exceptions import ClientError

    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def upload_if_needed(s3, bucket: str, local_path: Path, s3_key: str, force: bool) -> bool:
    local_size = local_path.stat().st_size
    if not force:
        remote_size = s3_object_size(s3, bucket, s3_key)
        if remote_size == local_size:
            return False

    extra: dict[str, str] = {}
    if local_path.suffix.lower() == ".mp4":
        extra["ContentType"] = "video/mp4"
    elif local_path.suffix.lower() == ".jpg":
        extra["ContentType"] = "image/jpeg"

    s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra)
    return True


def upload_source(s3, bucket: str, source_path: Path, source_id: str, force: bool) -> str:
    key = f"sources/{source_id}.mp4"
    uploaded = upload_if_needed(s3, bucket, source_path, key, force)
    log(f"  source: {'uploaded' if uploaded else 'already present'} ({key})")
    return key


def upload_clips(
    s3, bucket: str, clips: list[ClipInfo], source_id: str, force: bool
) -> list[tuple[ClipInfo, str, str]]:
    out = []
    for c in clips:
        clip_key = f"clips/{source_id}/{c.filename}"
        thumb_key = f"clips/{source_id}/{c.thumbnail_filename}"

        clip_uploaded = upload_if_needed(s3, bucket, c.local_path, clip_key, force)
        thumb_uploaded = upload_if_needed(s3, bucket, c.thumbnail_path, thumb_key, force)
        log(
            f"  clip {c.index:03d}: "
            f"video {'uploaded' if clip_uploaded else 'present'}, "
            f"thumb {'uploaded' if thumb_uploaded else 'present'}"
        )
        out.append((c, clip_key, thumb_key))
    return out


def write_db_rows(
    *,
    supabase,
    source_id: str,
    url: str,
    display_name: str | None,
    source_duration: float,
    source_fps: float | None,
    downloaded_by: str | None,
    clips_with_keys: list[tuple[ClipInfo, str, str]],
    s3_bucket: str,
) -> None:
    from src import db

    db.upsert_source_video(
        supabase,
        source_id=source_id,
        url=url,
        display_name=display_name,
        duration_sec=source_duration,
        fps_original=source_fps,
        downloaded_by=downloaded_by,
    )
    log(f"  source_videos row upserted: {source_id}")

    for c, clip_key, thumb_key in clips_with_keys:
        db.upsert_clip(
            supabase,
            source_id=source_id,
            clip_index=c.index,
            filename=c.filename,
            s3_bucket=s3_bucket,
            s3_key=clip_key,
            thumbnail_s3_key=thumb_key,
            start_sec=c.start_sec,
            end_sec=c.end_sec,
            duration_sec=c.duration_sec,
        )
    log(f"  clips rows upserted: {len(clips_with_keys)}")


def main() -> int:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        description="Ingest a YouTube source: download, segment to 60s/30fps clips, upload to S3, upsert source_videos and clips in Supabase."
    )
    parser.add_argument("url", help="YouTube URL or 11-char video ID")
    parser.add_argument("--display-name", default=None, help="Human-readable name (e.g. 'USCG vs Stanford 4/15')")
    parser.add_argument("--force", action="store_true", help="Redo all steps even if outputs already exist")
    parser.add_argument(
        "--software-encode",
        action="store_true",
        help="Force libx264 instead of hardware-accelerated h264_videotoolbox",
    )
    args = parser.parse_args()

    load_dotenv()
    required_env = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
    ]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        log(f"ERROR: missing env vars: {', '.join(missing)}")
        return 1

    s3_bucket = os.environ["S3_BUCKET"]
    downloaded_by = os.environ.get("ANNOTATOR_NAME") or os.environ.get("USER")

    source_id = extract_youtube_id(args.url)
    canonical_url = f"https://www.youtube.com/watch?v={source_id}"
    log(f"Source ID: {source_id}")
    log(f"URL: {canonical_url}")
    if args.display_name:
        log(f"Display name: {args.display_name}")

    use_videotoolbox = not args.software_encode and has_videotoolbox()
    if args.software_encode:
        log("Encoder: libx264 (forced via --software-encode)")
    elif not use_videotoolbox:
        log("Encoder: libx264 (h264_videotoolbox not available)")

    from src import db

    supabase = db.get_supabase_client()

    import boto3

    s3 = boto3.client(
        "s3",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    existing_source = db.get_source_video(supabase, source_id)
    if existing_source and not args.force:
        log(
            f"\nSource {source_id} already exists in Supabase "
            f"(downloaded {existing_source['downloaded_at']} by "
            f"{existing_source.get('downloaded_by') or 'unknown'})."
        )
        log("Idempotent re-run will skip already-completed steps. Use --force to redo all work.")

    workdir = WORKDIR_ROOT / source_id
    workdir.mkdir(parents=True, exist_ok=True)

    step("[1/4] Download source video")
    source_path = download_source(canonical_url, source_id, workdir, args.force)
    source_duration = ffprobe_duration(source_path)
    source_fps = ffprobe_fps(source_path)
    fps_str = f"{source_fps:.2f}" if source_fps is not None else "unknown"
    log(f"Source: {source_duration:.1f}s, {fps_str} fps")

    step("[2/4] Segment into 60s/30fps clips and generate thumbnails")
    clips = segment_and_thumbnail(
        source_path,
        source_id,
        workdir / "clips",
        use_videotoolbox=use_videotoolbox,
        force=args.force,
    )
    log(f"Total clips: {len(clips)}")

    step(f"[3/4] Upload to s3://{s3_bucket}/")
    upload_source(s3, s3_bucket, source_path, source_id, args.force)
    clips_with_keys = upload_clips(s3, s3_bucket, clips, source_id, args.force)

    step("[4/4] Write metadata to Supabase")
    write_db_rows(
        supabase=supabase,
        source_id=source_id,
        url=canonical_url,
        display_name=args.display_name,
        source_duration=source_duration,
        source_fps=source_fps,
        downloaded_by=downloaded_by,
        clips_with_keys=clips_with_keys,
        s3_bucket=s3_bucket,
    )

    step("Cleanup")
    shutil.rmtree(workdir)
    log(f"Removed {workdir}")

    log("\n✓ Done.")
    log("\nNext steps:")
    log("  1. Add a row to the coordination sheet:")
    log(f"     Source ID: {source_id}")
    log(f"     Display name: {args.display_name or '(none)'}")
    log("     Status: Available")
    log("  2. In your local Label Studio:")
    log(f"     Project Settings → Cloud Storage → set Bucket Prefix to clips/{source_id}/")
    log("     Click Sync to pull tasks for this source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

