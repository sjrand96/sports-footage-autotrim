/**
 * Import per-frame model predictions (CSV) into second-based "Playing" intervals.
 * CSV columns: clip_id or clip_key, frame_idx, prob_playing, pred_playing;
 * optional is_playing (ground truth) or label_playing (legacy).
 * Frame indices are 0-based; clips are 30 fps CFR (see labelStudioImport.js).
 */

import {
  LABEL_STUDIO_FPS,
  clampToDuration,
  localFileBasename,
  mergeAdjacentOrOverlapping,
} from './labelStudioImport.js'

const REQUIRED_COLUMNS = ['frame_idx', 'pred_playing']

function parseBoolCell(value) {
  const s = String(value ?? '')
    .trim()
    .toLowerCase()
  if (s === 'true' || s === '1') return true
  if (s === 'false' || s === '0') return false
  return null
}

function clipIdFromFileLabel(fileLabel) {
  const basename = localFileBasename(fileLabel)
  if (!basename) return null
  return basename.replace(/\.mp4$/i, '')
}

/** 0-based inclusive frame indices → [start, end) seconds. */
export function zeroBasedFramesToSecondsRange(startIdx, endIdxInclusive, fps) {
  const start = Math.max(0, startIdx / fps)
  const end = Math.max(start, (endIdxInclusive + 1) / fps)
  return { start, end }
}

/**
 * @returns {{
 *   rows: { clip_id: string, frame_idx: number, prob_playing: number, pred_playing: boolean, is_playing: boolean | null, label_playing: boolean | null }[],
 *   hasIsPlayingColumn: boolean
 * }}
 */
export function parseFramePredictionsCsv(raw) {
  const text = typeof raw === 'string' ? raw.trim() : ''
  if (!text) throw new Error('CSV file is empty')

  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length < 2) throw new Error('CSV has no data rows')

  const header = lines[0].split(',').map((h) => h.trim())
  const clipCol = header.includes('clip_key')
    ? 'clip_key'
    : header.includes('clip_id')
      ? 'clip_id'
      : null
  if (!clipCol) {
    throw new Error('CSV missing required column: clip_id or clip_key')
  }
  for (const col of REQUIRED_COLUMNS) {
    if (!header.includes(col)) {
      throw new Error(`CSV missing required column: ${col}`)
    }
  }
  const col = Object.fromEntries(header.map((name, i) => [name, i]))
  const hasIsPlayingColumn = header.includes('is_playing')

  const rows = []
  for (let li = 1; li < lines.length; li++) {
    const cells = lines[li].split(',')
    const clip_id = cells[col[clipCol]]?.trim()
    const frame_idx = Number(cells[col.frame_idx])
    const pred_playing = parseBoolCell(cells[col.pred_playing])
    if (!clip_id || !Number.isFinite(frame_idx) || frame_idx < 0 || pred_playing == null) {
      continue
    }
    const prob_playing = Number(cells[col.prob_playing])
    const is_playing =
      col.is_playing != null ? parseBoolCell(cells[col.is_playing]) : null
    const label_playing =
      col.label_playing != null ? parseBoolCell(cells[col.label_playing]) : null
    rows.push({
      clip_id,
      frame_idx,
      prob_playing: Number.isFinite(prob_playing) ? prob_playing : 0,
      pred_playing,
      is_playing,
      label_playing,
    })
  }

  if (rows.length === 0) throw new Error('No valid prediction rows found in CSV')
  return { rows, hasIsPlayingColumn }
}

function playingFrameRuns(rows, playingField) {
  const frames = rows
    .filter((r) => r[playingField] === true)
    .map((r) => r.frame_idx)
    .sort((a, b) => a - b)
  if (frames.length === 0) return []

  const runs = []
  let runStart = frames[0]
  let runEnd = frames[0]
  for (let i = 1; i < frames.length; i++) {
    if (frames[i] === runEnd + 1) runEnd = frames[i]
    else {
      runs.push({ start: runStart, end: runEnd })
      runStart = frames[i]
      runEnd = frames[i]
    }
  }
  runs.push({ start: runStart, end: runEnd })
  return runs
}

function runsToIntervals(runs, durationSec, fps) {
  if (runs.length === 0) return []
  let intervals = runs.map(({ start, end }) =>
    zeroBasedFramesToSecondsRange(start, end, fps),
  )
  intervals = mergeAdjacentOrOverlapping(intervals)
  return clampToDuration(intervals, durationSec)
}

/**
 * @returns {{
 *   intervals?: { start: number, end: number }[],
 *   groundTruthIntervals?: { start: number, end: number }[],
 *   hasGroundTruthColumn?: boolean,
 *   error?: string
 * }}
 */
export function playingIntervalsSecondsFromFramePredictionsCsv(
  rawCsv,
  fileLabel,
  durationSec,
  { playingField = 'pred_playing', fps = LABEL_STUDIO_FPS } = {},
) {
  const clipId = clipIdFromFileLabel(fileLabel)?.toLowerCase()
  if (!clipId) return { error: 'Open a video file first.' }
  if (!Number.isFinite(durationSec) || durationSec <= 0)
    return { error: 'Wait for the video to finish loading.' }

  let rows
  let hasGroundTruthColumn
  try {
    const parsed = parseFramePredictionsCsv(rawCsv)
    rows = parsed.rows
    hasGroundTruthColumn = parsed.hasIsPlayingColumn
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Invalid CSV format.' }
  }

  const matching = rows.filter((r) => r.clip_id.toLowerCase() === clipId)
  if (matching.length === 0) {
    return {
      error: `No CSV rows match this clip (${clipId}). Expected clip_id/clip_key to match the open .mp4 basename.`,
    }
  }

  const predRuns = playingFrameRuns(matching, playingField)
  if (predRuns.length === 0) {
    return { error: 'No predicted Playing frames found for this clip in the CSV.' }
  }

  const intervals = runsToIntervals(predRuns, durationSec, fps)
  if (intervals.length === 0) {
    return { error: "All imported segments lie outside this video's duration." }
  }

  const result = { intervals, hasGroundTruthColumn }
  if (hasGroundTruthColumn) {
    const gtRuns = playingFrameRuns(matching, 'is_playing')
    result.groundTruthIntervals = runsToIntervals(gtRuns, durationSec, fps)
  }

  return result
}
