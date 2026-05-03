# Volleyball Clip Prep Workflow

A repeatable pipeline for turning a YouTube URL into a folder of 1-minute, 30fps clips ready for Label Studio annotation.

**Automated path (team default):** from the repo root, run `python scripts/prep_videos.py` with AWS + Supabase env vars set (see `docs/annotation_schema_and_systems.md` — Credentials). That script performs the same download + ffmpeg segmentation described below, uploads to S3, writes `source_videos` / `clips` (including thumbnails) to Supabase, then deletes local `./workdir/{video_id}/`. The steps on this page are the **manual** equivalent if you need to debug formats or run without the pipeline.

## Prerequisites

Install once:

```
brew install yt-dlp ffmpeg
```

## Naming convention

Given a base name like `USCG1`:

- Downloaded full video: `USCG1.mp4`
- Output folder: `USCG1_clips/`
- Clips inside the folder: `USCG1_001.mp4`, `USCG1_002.mp4`, `USCG1_003.mp4`, ...

Zero-padded 3-digit suffixes keep clips in correct sort order even if a video runs over 99 minutes.

## Step 1: Download the video at 1080p60

Replace `<URL>` with the YouTube link and `<NAME>` with your chosen base name (e.g. `USCG1`).

```
yt-dlp -f "299+140" --merge-output-format mp4 -o "<NAME>.mp4" "<URL>"
```

What this does:

- `-f "299+140"` — format 299 (1080p60 H.264 video) plus format 140 (m4a audio)
- `--merge-output-format mp4` — merge into mp4 container
- `-o` — set output filename

If you're unsure whether a specific video has format 299 available, list formats first:

```
yt-dlp -F "<URL>"
```

Look for `299` (1080p60 mp4) and `140` (m4a audio). If the video was uploaded at 30fps, you'll see `137` (1080p30 mp4) instead — use `-f "137+140"` in that case.

If you want a more resilient selector that handles any 1080p video automatically:

```
yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" --merge-output-format mp4 -o "<NAME>.mp4" "<URL>"
```

## Step 2: Cut into 1-minute, 30fps clips

This is a single ffmpeg command that re-encodes to 30fps and segments into 60-second clips in one pass:

```
mkdir -p <NAME>_clips && \
ffmpeg -i <NAME>.mp4 \
  -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 128k \
  -f segment -segment_time 60 -reset_timestamps 1 \
  -segment_start_number 1 \
  "<NAME>_clips/<NAME>_%03d.mp4"
```

What each flag does:

- `mkdir -p <NAME>_clips` — creates the output folder if it doesn't exist
- `-c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p` — H.264 video with broad-compatibility settings
- `-r 30` — force constant 30fps (drops every other frame from 60fps source)
- `-c:a aac -b:a 128k` — AAC audio at 128 kbps
- `-f segment` — use ffmpeg's segment muxer
- `-segment_time 60` — each segment is 60 seconds
- `-reset_timestamps 1` — each clip's timestamps start at zero (important for Label Studio)
- `-segment_start_number 1` — start numbering at 001 instead of 000
- `%03d` — zero-padded 3-digit segment index

A 17-minute video produces `<NAME>_001.mp4` through `<NAME>_017.mp4`, each exactly 60 seconds (the last one may be shorter if the source isn't a whole number of minutes).

This step takes a few minutes for a long video — you're re-encoding the entire thing, not just copying.

## Step 3: Verify the output

Quick sanity check on one of the clips:

```
ffprobe -v error -show_format -show_streams -print_format json <NAME>_clips/<NAME>_001.mp4
```

Confirm:

- `r_frame_rate` is `30/1`
- `codec_name` is `h264`
- `pix_fmt` is `yuv420p`
- `duration` is around `60.0`

You can also list all clips with their durations:

```
for f in <NAME>_clips/*.mp4; do
  echo -n "$f: "
  ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$f"
done
```

## Step 4: Import into Label Studio

In your Label Studio project, use the Data Manager → Import to drag the entire `<NAME>_clips/` folder in. Each clip becomes its own task. Or, for many videos at scale, set up local file storage pointing at the parent directory containing all your `*_clips/` folders.

## Full example

For a 17-minute video at `https://youtube.com/watch?v=ABC123` named `USCG1`:

```
# Download
yt-dlp -f "299+140" --merge-output-format mp4 -o "USCG1.mp4" "https://youtube.com/watch?v=ABC123"

# Cut and re-encode in one pass
mkdir -p USCG1_clips && \
ffmpeg -i USCG1.mp4 \
  -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 128k \
  -f segment -segment_time 60 -reset_timestamps 1 \
  -segment_start_number 1 \
  "USCG1_clips/USCG1_%03d.mp4"

# Verify one clip
ffprobe -v error -show_format -show_streams -print_format json USCG1_clips/USCG1_001.mp4
```

Result: `USCG1_clips/` folder containing `USCG1_001.mp4` through `USCG1_017.mp4`.

## Optional: shell script wrapper

If you'll do this often, save the following as `prep_video.sh` and make it executable with `chmod +x prep_video.sh`:

```bash
#!/bin/bash
set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <youtube_url> <base_name>"
  echo "Example: $0 'https://youtube.com/watch?v=ABC123' USCG1"
  exit 1
fi

URL="$1"
NAME="$2"

echo "Downloading $NAME..."
yt-dlp -f "299+140" --merge-output-format mp4 -o "${NAME}.mp4" "$URL"

echo "Cutting into 1-minute 30fps clips..."
mkdir -p "${NAME}_clips"
ffmpeg -i "${NAME}.mp4" \
  -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p -r 30 \
  -c:a aac -b:a 128k \
  -f segment -segment_time 60 -reset_timestamps 1 \
  -segment_start_number 1 \
  "${NAME}_clips/${NAME}_%03d.mp4"

echo "Done. Clips are in ${NAME}_clips/"
ls "${NAME}_clips/"
```

Run with: `./prep_video.sh "https://youtube.com/watch?v=ABC123" USCG1`

## After labeling (S3-backed projects)

If you use the team pipeline (clips on S3 + rows in Supabase): submit tasks in Label Studio, then **Export → JSON**, and run:

`python scripts/push_annotations.py /path/to/export.json`

See **`docs/annotation_schema_and_systems.md`** (section W3) for env vars, idempotency, and payload shape.

## Notes and edge cases

- **Last clip may be shorter than 60 seconds.** If your source is 17 minutes 23 seconds, the final clip will be 23 seconds. This is fine for Label Studio.
- **Disk space.** Re-encoded 1080p30 H.264 runs roughly 100–200 MB per minute depending on motion complexity. A 17-minute source could yield 2–3 GB of clips. The original download is also kept; delete it after if you're tight on space.
- **Re-encoding time.** On an M-series MacBook Air, expect roughly real-time speed (a 17-minute video takes ~17 minutes to process). Apple Silicon's hardware encoder via `-c:v h264_videotoolbox` is faster, but produces slightly larger files at the same visual quality. For batch processing, the speed/size tradeoff usually favors `libx264`.
- **Frame alignment across clips.** Each clip's frame 0 is a fresh keyframe due to `-reset_timestamps 1`. If you ever need to map a clip frame back to the original video timeline, `original_frame = (clip_index - 1) * 1800 + clip_frame` (since 60s × 30fps = 1800 frames per clip).