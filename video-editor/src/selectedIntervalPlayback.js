export const SELECTED_PLAY_EPS = 1e-3

export function sortIntervals(intervals) {
  return [...intervals]
    .filter((iv) => iv.end > iv.start + SELECTED_PLAY_EPS)
    .sort((a, b) => a.start - b.start)
}

export function isTimeInSelectedIntervals(t, intervals) {
  const sorted = sortIntervals(intervals)
  return sorted.some(
    (iv) =>
      t >= iv.start - SELECTED_PLAY_EPS && t < iv.end - SELECTED_PLAY_EPS,
  )
}

/**
 * While playing with "selected only" mode: stay inside highlighted segments
 * by seeking across gaps to the next segment; after the last segment, loop to
 * the first segment start.
 */
export function gatePlaySelectedOnly(video, intervals) {
  const sorted = sortIntervals(intervals)
  if (sorted.length === 0) return

  const t = video.currentTime
  const dur = Number.isFinite(video.duration) ? video.duration : Infinity
  const first = sorted[0]
  const last = sorted[sorted.length - 1]

  if (isTimeInSelectedIntervals(t, intervals)) return

  if (t < first.start - SELECTED_PLAY_EPS) {
    video.currentTime = Math.max(0, first.start)
    return
  }

  if (t >= last.end - SELECTED_PLAY_EPS) {
    video.currentTime = Math.min(Math.max(0, first.start), dur)
    return
  }

  const next = sorted.find((iv) => iv.start > t + SELECTED_PLAY_EPS)
  if (next) {
    video.currentTime = next.start
    return
  }

  video.currentTime = Math.min(Math.max(0, first.start), dur)
}

/** When starting playback in selected-only mode, jump from gaps / past end into a valid segment. */
export function snapTimeForSelectedPlayStart(t, intervals) {
  const sorted = sortIntervals(intervals)
  if (sorted.length === 0) return null

  if (isTimeInSelectedIntervals(t, intervals)) return t

  if (t < sorted[0].start - SELECTED_PLAY_EPS) return sorted[0].start

  if (t >= sorted[sorted.length - 1].end - SELECTED_PLAY_EPS) {
    return sorted[0].start
  }

  const next = sorted.find((iv) => iv.start > t + SELECTED_PLAY_EPS)
  if (next) return next.start

  return sorted[0].start
}
