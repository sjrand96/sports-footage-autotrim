/**
 * Import per-frame model predictions (CSV) into second-based "Playing" intervals.
 * CSV columns: clip_id, frame_idx, prob_playing, pred_playing, label_playing
 * Frame indices are 0-based; clips are 30 fps CFR (see labelStudioImport.js).
 */

import {
  LABEL_STUDIO_FPS,
  clampToDuration,
  localFileBasename,
  mergeAdjacentOrOverlapping,
} from './labelStudioImport.js'

const REQUIRED_COLUMNS = ['clip_id', 'frame_idx', 'pred_playing']

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
 * @returns {{ clip_id: string, frame_idx: number, prob_playing: number, pred_playing: boolean, label_playing: boolean | null }[]}
 */
export function parseFramePredictionsCsv(raw) {
  const text = typeof raw === 'string' ? raw.trim() : ''
  if (!text) throw new Error('CSV file is empty')

  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length < 2) throw new Error('CSV has no data rows')

  const header = lines[0].split(',').map((h) => h.trim())
  for (const col of REQUIRED_COLUMNS) {
    if (!header.includes(col)) {
      throw new Error(`CSV missing required column: ${col}`)
    }
  }
  const col = Object.fromEntries(header.map((name, i) => [name, i]))

  const rows = []
  for (let li = 1; li < lines.length; li++) {
    const cells = lines[li].split(',')
    const clip_id = cells[col.clip_id]?.trim()
    const frame_idx = Number(cells[col.frame_idx])
    const pred_playing = parseBoolCell(cells[col.pred_playing])
    if (!clip_id || !Number.isFinite(frame_idx) || frame_idx < 0 || pred_playing == null) {
      continue
    }
    const prob_playing = Number(cells[col.prob_playing])
    const label_playing =
      col.label_playing != null ? parseBoolCell(cells[col.label_playing]) : null
    rows.push({
      clip_id,
      frame_idx,
      prob_playing: Number.isFinite(prob_playing) ? prob_playing : 0,
      pred_playing,
      label_playing,
    })
  }

  if (rows.length === 0) throw new Error('No valid prediction rows found in CSV')
  return rows
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

/**
 * @returns {{ intervals?: { start: number, end: number }[], error?: string }}
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
  try {
    rows = parseFramePredictionsCsv(rawCsv)
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Invalid CSV format.' }
  }

  const matching = rows.filter((r) => r.clip_id.toLowerCase() === clipId)
  if (matching.length === 0) {
    return {
      error: `No CSV rows match this clip (${clipId}). Expected clip_id to match the open .mp4 basename.`,
    }
  }

  const runs = playingFrameRuns(matching, playingField)
  if (runs.length === 0) {
    return { error: 'No predicted Playing frames found for this clip in the CSV.' }
  }

  let intervals = runs.map(({ start, end }) =>
    zeroBasedFramesToSecondsRange(start, end, fps),
  )
  intervals = mergeAdjacentOrOverlapping(intervals)
  intervals = clampToDuration(intervals, durationSec)
  if (intervals.length === 0)
    return { error: 'All imported segments lie outside this video\'s duration.' }

  return { intervals }
}
