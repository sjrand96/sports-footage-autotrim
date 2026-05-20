/**
 * Import "Playing" timeline segments from Label Studio JSON exports into
 * second-based intervals for the video editor (30 fps CFR clips — see
 * docs/annotation_process/annotation_schema_and_systems.md).
 */

/** Matches prep / Label Studio `frameRate="30"`. */
export const LABEL_STUDIO_FPS = 30

const MIN_SEGMENT_SEC = 0.05 * 2

export function clipBasenameFromLsVideoField(video) {
  if (!video || typeof video !== 'string') return null
  let s = video.trim()
  try {
    s = decodeURIComponent(s)
  } catch {
    /* ignore malformed escape */
  }
  const seg = s.split('?')[0].split('/').pop() || ''
  if (!/\.mp4$/i.test(seg)) return null
  return seg
}

export function localFileBasename(fileLabel) {
  if (!fileLabel || typeof fileLabel !== 'string') return null
  const parts = fileLabel.replace(/\\/g, '/').split('/')
  return parts.pop()?.trim() || null
}

function pickLatestAnnotation(annotations) {
  const candidates = (annotations || []).filter(
    (a) => a && typeof a === 'object' && !a.was_cancelled,
  )
  if (candidates.length === 0) return null
  return candidates.reduce((best, cur) => {
    const bt = `${best.updated_at ?? best.created_at ?? ''}`
    const ct = `${cur.updated_at ?? cur.created_at ?? ''}`
    return ct >= bt ? cur : best
  })
}

/** 1-based inclusive timeline frames → [start,end) seconds in CFR video. */
function framesToSecondsRange(startFrame, endFrame, fps) {
  const start = Math.max(0, (startFrame - 1) / fps)
  const end = Math.max(start, endFrame / fps)
  return { start, end }
}

function collectPlayingFramePairs(valueLike) {
  const labels = valueLike?.timelinelabels
  if (!Array.isArray(labels) || !labels.includes('Playing')) return []
  const ranges = valueLike?.ranges
  if (!Array.isArray(ranges)) return []
  const out = []
  for (const r of ranges) {
    const sf = Number(r?.start)
    const ef = Number(r?.end)
    if (!Number.isFinite(sf) || !Number.isFinite(ef) || ef < sf) continue
    out.push({ start: sf, end: ef })
  }
  return out
}

function playingFramePairsFromAnnotation(annotation) {
  const result = annotation?.result
  if (!Array.isArray(result)) return []
  const pairs = []
  for (const item of result) {
    if (item?.type !== 'timelinelabels' || !item?.value) continue
    pairs.push(...collectPlayingFramePairs(item.value))
  }
  return pairs
}

function playingFramePairsLegacy(task) {
  const vls = task?.videoLabels
  if (!Array.isArray(vls)) return []
  const pairs = []
  for (const vl of vls) pairs.push(...collectPlayingFramePairs(vl))
  return pairs
}

function sortByStart(a, b) {
  return a.start - b.start || a.end - b.end
}

export function mergeAdjacentOrOverlapping(intervals) {
  const sorted = [...intervals].sort(sortByStart)
  const out = []
  for (const iv of sorted) {
    const last = out[out.length - 1]
    if (last && iv.start <= last.end + 1e-9) last.end = Math.max(last.end, iv.end)
    else out.push({ start: iv.start, end: iv.end })
  }
  return out
}

export function clampToDuration(intervals, durationSec) {
  const d = durationSec
  const out = []
  for (const iv of intervals) {
    let { start, end } = iv
    start = Math.min(Math.max(0, start), d)
    end = Math.min(Math.max(0, end), d)
    if (end <= start || end - start < MIN_SEGMENT_SEC) continue
    out.push({ start, end })
  }
  return out
}

/**
 * Normalize Label Studio tasks from multiple export shapes:
 * - Bulk LS export: `[{ data: { video }, annotations, ... }, ...]`
 * - Per-clip DB / repo label file: `{ payload: { label_studio_task }, source_id, clip_index }`
 * - Single task object at root
 */
export function normalizeLabelStudioTasks(data) {
  if (Array.isArray(data)) return data

  if (data && typeof data === 'object') {
    const fromPayload = data.payload?.label_studio_task ?? data.label_studio_task
    if (fromPayload && typeof fromPayload === 'object') return [fromPayload]

    if (data.data?.video ?? data.video) return [data]
  }

  throw new Error(
    'Unrecognized label JSON. Use a Label Studio export array or a per-clip label file from data/labels/.',
  )
}

export function parseLabelStudioTasksJson(raw) {
  const data = typeof raw === 'string' ? JSON.parse(raw) : raw
  return normalizeLabelStudioTasks(data)
}

/** `{source_id}_{NNN}.mp4` from a per-clip label wrapper, if present. */
export function clipBasenameFromLabelRecord(record) {
  if (!record || typeof record !== 'object') return null
  const sourceId = record.source_id
  const clipIndex = Number(record.clip_index)
  if (typeof sourceId === 'string' && Number.isFinite(clipIndex) && clipIndex > 0) {
    return `${sourceId}_${String(clipIndex).padStart(3, '0')}.mp4`
  }
  return null
}

/**
 * @returns {{ intervals?: { start: number, end: number }[], error?: string }}
 */
export function playingIntervalsSecondsForExport(
  tasks,
  fileLabel,
  durationSec,
  fps = LABEL_STUDIO_FPS,
  { labelRecord = null } = {},
) {
  const basename = localFileBasename(fileLabel)?.toLowerCase()
  if (!basename) return { error: 'Open a video file first.' }
  if (!Number.isFinite(durationSec) || durationSec <= 0)
    return { error: 'Wait for the video to finish loading.' }
  if (!Array.isArray(tasks)) return { error: 'Invalid export format (not an array).' }

  const recordBasename = clipBasenameFromLabelRecord(labelRecord)?.toLowerCase()
  let task = null
  if (tasks.length === 1) {
    const only = tasks[0]
    const v = only?.data?.video ?? only?.video
    const b = clipBasenameFromLsVideoField(v)?.toLowerCase()
    if (!b || b === basename || recordBasename === basename) task = only
  }
  if (!task) {
    task = tasks.find((t) => {
      const v = t?.data?.video ?? t?.video
      const b = clipBasenameFromLsVideoField(v)
      return b && b.toLowerCase() === basename
    })
  }
  if (!task && recordBasename === basename && tasks.length > 0) task = tasks[0]
  if (!task) {
    return {
      error: `No task matches this file (${basename}). Export URLs should point at that clip.`,
    }
  }

  let framePairs = []
  const annotations = task.annotations
  if (Array.isArray(annotations) && annotations.length > 0) {
    const ann = pickLatestAnnotation(annotations)
    if (ann) framePairs = playingFramePairsFromAnnotation(ann)
  }
  if (framePairs.length === 0 && Array.isArray(task.videoLabels)) {
    framePairs = playingFramePairsLegacy(task)
  }

  if (framePairs.length === 0)
    return { error: 'No Playing segments found for this video in the export.' }

  let intervals = framePairs.map(({ start: sf, end: ef }) =>
    framesToSecondsRange(sf, ef, fps),
  )
  intervals = mergeAdjacentOrOverlapping(intervals)
  intervals = clampToDuration(intervals, durationSec)
  if (intervals.length === 0)
    return { error: 'All imported segments lie outside this video\'s duration.' }

  return { intervals }
}

/**
 * Parse label JSON (any supported shape) and return playing intervals for the open video.
 * @returns {{ intervals?: { start: number, end: number }[], error?: string }}
 */
export function playingIntervalsSecondsFromLabelJson(raw, fileLabel, durationSec, fps) {
  let data
  try {
    data = typeof raw === 'string' ? JSON.parse(raw) : raw
  } catch {
    return { error: 'Invalid JSON file.' }
  }
  let tasks
  try {
    tasks = normalizeLabelStudioTasks(data)
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Unrecognized label JSON format.' }
  }
  const labelRecord =
    data && typeof data === 'object' && !Array.isArray(data) ? data : null
  return playingIntervalsSecondsForExport(
    tasks,
    fileLabel,
    durationSec,
    fps,
    { labelRecord },
  )
}
