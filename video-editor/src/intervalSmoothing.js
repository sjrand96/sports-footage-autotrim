/** Morphological-style smoothing on playing (positive) intervals. */

import { mergeIntervals } from './timelineMetrics.js'

const EPS = 1e-9

/** @typedef {{ start: number, end: number }} Span */

/** Defaults aligned with training eval gap (0.5s) and ~10 frames at 30 fps. */
export const DEFAULT_SMOOTH_OPTS = {
  maxGapSec: 0.5,
  minPlaySec: 1,
}

/**
 * Pass 1: merge playing segments separated by at most maxGapSec of downtime.
 * @param {Span[]} spans sorted, non-overlapping
 */
export function mergeAcrossGaps(spans, maxGapSec) {
  if (spans.length === 0) return []
  if (!(maxGapSec > 0)) return spans.map((iv) => ({ ...iv }))

  const out = [{ start: spans[0].start, end: spans[0].end }]
  for (let i = 1; i < spans.length; i++) {
    const iv = spans[i]
    const last = out[out.length - 1]
    if (iv.start - last.end <= maxGapSec + EPS) {
      last.end = Math.max(last.end, iv.end)
    } else {
      out.push({ start: iv.start, end: iv.end })
    }
  }
  return out
}

/**
 * Pass 2: drop playing segments shorter than minPlaySec.
 * @param {Span[]} spans
 */
export function dropShortSpans(spans, minPlaySec) {
  if (!(minPlaySec > 0)) return spans.map((iv) => ({ ...iv }))
  return spans
    .filter((iv) => iv.end - iv.start >= minPlaySec - EPS)
    .map((iv) => ({ start: iv.start, end: iv.end }))
}

/**
 * Two-pass interval smoothing: merge across gaps, then remove short positives.
 * @param {Span[]} spans
 * @param {number} durationSec
 * @param {{ maxGapSec?: number, minPlaySec?: number }} [opts]
 */
export function smoothPlayingIntervals(spans, durationSec, opts = {}) {
  const { maxGapSec, minPlaySec } = { ...DEFAULT_SMOOTH_OPTS, ...opts }
  let merged = mergeIntervals(spans, durationSec)
  merged = mergeAcrossGaps(merged, maxGapSec)
  merged = dropShortSpans(merged, minPlaySec)
  return merged
}
