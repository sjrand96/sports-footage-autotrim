/** Binary timeline metrics: Playing (positive) vs downtime (negative). */

const EPS = 1e-9

/** @typedef {{ start: number, end: number }} Span */

/** @param {Span[]} spans */
export function mergeIntervals(spans, durationCap) {
  const d =
    durationCap != null && Number.isFinite(durationCap)
      ? Math.max(0, durationCap)
      : Infinity
  const trimmed = spans
    .map((iv) => {
      const lo = Math.max(0, Math.min(iv.start, d))
      const hi = Math.max(lo, Math.min(iv.end, d))
      return { start: lo, end: hi }
    })
    .filter((iv) => iv.end - iv.start > EPS)

  if (trimmed.length === 0) return []

  trimmed.sort((a, b) => a.start - b.start || a.end - b.end)
  const out = []
  for (const iv of trimmed) {
    const last = out[out.length - 1]
    if (!last || iv.start > last.end + EPS) out.push({ ...iv })
    else last.end = Math.max(last.end, iv.end)
  }
  return out
}

/** @param {Span[]} merged */
export function totalPlayingSeconds(merged) {
  let s = 0
  for (const iv of merged) s += iv.end - iv.start
  return s
}

/**
 * Half-open [start, end): aligned with scrubber segments over the clip.
 *
 * @param {number} t
 * @param {Span[]} merged
 */
export function isPlayingAt(t, merged) {
  return merged.some((iv) => t >= iv.start && t < iv.end)
}

/**
 * @typedef {{
 *   tpSec: number, fpSec: number, fnSec: number, tnSec: number,
 *   predictedCoveragePct: number, gtCoveragePct: number
 * }} TimelineMetricsResult
 */

/**
 * @param {Span[]} predicted
 * @param {Span[]} groundTruth
 * @param {number} durationSec
 * @returns {TimelineMetricsResult}
 */
export function computeTimelineMetrics(predicted, groundTruth, durationSec) {
  const d = durationSec
  if (!Number.isFinite(d) || d <= EPS) {
    return {
      tpSec: 0,
      fpSec: 0,
      fnSec: 0,
      tnSec: 0,
      predictedCoveragePct: 0,
      gtCoveragePct: 0,
    }
  }

  const predM = mergeIntervals(predicted ?? [], d)
  const gtM = mergeIntervals(groundTruth ?? [], d)

  const breakpoints = new Set([0, d])
  for (const iv of predM) {
    breakpoints.add(iv.start)
    breakpoints.add(iv.end)
  }
  for (const iv of gtM) {
    breakpoints.add(iv.start)
    breakpoints.add(iv.end)
  }

  const bps = [...breakpoints].filter((x) => x >= 0 && x <= d).sort((a, b) => a - b)

  let tp = 0
  let fp = 0
  let fn = 0
  let tn = 0

  for (let i = 0; i < bps.length - 1; i++) {
    const a = bps[i]
    const b = bps[i + 1]
    const slice = b - a
    if (slice <= EPS) continue

    const t = (a + b) / 2
    const predOn = isPlayingAt(t, predM)
    const gtOn = isPlayingAt(t, gtM)

    if (predOn && gtOn) tp += slice
    else if (predOn && !gtOn) fp += slice
    else if (!predOn && gtOn) fn += slice
    else tn += slice
  }

  const predictedCoveragePct = Math.min(
    100,
    (totalPlayingSeconds(predM) / d) * 100,
  )
  const gtCoveragePct = Math.min(
    100,
    (totalPlayingSeconds(gtM) / d) * 100,
  )

  return {
    tpSec: tp,
    fpSec: fp,
    fnSec: fn,
    tnSec: tn,
    predictedCoveragePct,
    gtCoveragePct,
  }
}
