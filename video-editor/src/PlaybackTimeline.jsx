import { useCallback, useMemo, useRef } from 'react'
import { mergeIntervals, totalPlayingSeconds } from './timelineMetrics.js'

export default function PlaybackTimeline({
  duration,
  currentTime,
  intervals,
  groundTruthIntervals = [],
  isPlaying,
  onTogglePlay,
  onSeek,
  onIntervalBoundaryChange,
}) {
  const masterTrackRef = useRef(null)
  const gtTrackRef = useRef(null)

  const timeFromClientX = useCallback(
    (clientX, el) => {
      if (!el || !duration || duration <= 0) return null
      const r = el.getBoundingClientRect()
      const x = Math.min(Math.max(0, clientX - r.left), r.width)
      return (x / r.width) * duration
    },
    [duration],
  )

  const seekFromPointer = useCallback(
    (clientX, el) => {
      const t = timeFromClientX(clientX, el)
      if (t != null) onSeek(t)
    },
    [timeFromClientX, onSeek],
  )

  const onTrackPointerDown = (e, ref) => {
    if (e.button !== 0) return
    const t = e.target
    if (t.closest?.('[data-timeline-handle]')) return
    if (t.closest?.('[data-playhead]')) return
    e.preventDefault()
    seekFromPointer(e.clientX, ref.current)

    const node = e.currentTarget
    node.setPointerCapture(e.pointerId)

    const move = (ev) => {
      seekFromPointer(ev.clientX, ref.current)
    }
    const up = () => {
      try {
        node.releasePointerCapture(e.pointerId)
      } catch {
        /* released */
      }
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const onPlayheadPointerDown = (e, ref) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const node = e.currentTarget
    node.setPointerCapture(e.pointerId)

    const move = (ev) => {
      seekFromPointer(ev.clientX, ref.current)
    }
    const up = () => {
      try {
        node.releasePointerCapture(e.pointerId)
      } catch {
        /* released */
      }
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const onHandlePointerDown = (e, intervalId, edge) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const iv = intervals.find((x) => x.id === intervalId)
    if (iv) {
      onSeek(edge === 'start' ? iv.start : iv.end)
    }

    const node = e.currentTarget
    node.setPointerCapture(e.pointerId)

    const move = (ev) => {
      const t = timeFromClientX(ev.clientX)
      if (t != null) onIntervalBoundaryChange(intervalId, edge, t)
    }
    const up = () => {
      try {
        node.releasePointerCapture(e.pointerId)
      } catch {
        /* released */
      }
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0
  const gtCoveragePct = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) return 0
    const merged = mergeIntervals(
      (groundTruthIntervals ?? []).map(({ start, end }) => ({ start, end })),
      duration,
    )
    return Math.min(100, (totalPlayingSeconds(merged) / duration) * 100)
  }, [groundTruthIntervals, duration])

  return (
    <>
      {/* Master timeline: no highlighted segments */}
      <div className="playback-row">
        <button
          type="button"
          className="playback-play playback-left-slot"
          onClick={onTogglePlay}
          disabled={!duration}
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            /* Pause Icon */
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            /* Play Icon */
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          )}
        </button>

        <div className="playback-body">
          <div className="playback-track-column">
            <div
              ref={masterTrackRef}
              className="playback-stack"
              onPointerDown={(e) => onTrackPointerDown(e, masterTrackRef)}
              role="presentation"
            >
              <div className="playback-track">
                <div className="playback-track-inactive" aria-hidden />

                <div className="playback-handles">
                  {intervals.flatMap((iv) => [
                    <button
                      key={`${iv.id}-start`}
                      type="button"
                      data-timeline-handle
                      className="playback-handle playback-handle--start"
                      style={{ left: `${(iv.start / duration) * 100}%` }}
                      aria-label="Adjust interval start"
                      onPointerDown={(e) =>
                        onHandlePointerDown(e, iv.id, 'start')
                      }
                    />,
                    <button
                      key={`${iv.id}-end`}
                      type="button"
                      data-timeline-handle
                      className="playback-handle playback-handle--end"
                      style={{ left: `${(iv.end / duration) * 100}%` }}
                      aria-label="Adjust interval end"
                      onPointerDown={(e) =>
                        onHandlePointerDown(e, iv.id, 'end')
                      }
                    />,
                  ])}
                </div>
              </div>

              <div
                className="playback-playhead"
                data-playhead
                style={{ left: `${pct}%` }}
                onPointerDown={(e) => onPlayheadPointerDown(e, masterTrackRef)}
              >
                <div className="playback-playhead-triangle" aria-hidden />
                <div className="playback-playhead-line" aria-hidden />
              </div>
            </div>
          </div>

          <div className="playback-time playback-right-slot">
            <span className="playback-time-current">{formatTime(currentTime)}</span>
            <span className="playback-time-sep"> / </span>
            <span className="playback-time-duration">{formatTime(duration)}</span>
          </div>
        </div>
      </div>

      {/* Ground truth timeline row */}
      {groundTruthIntervals && groundTruthIntervals.length > 0 ? (
        <div className="playback-row playback-row--secondary">
          <div
            className="playback-row-label playback-left-slot"
            aria-label="Ground truth row label"
          >
            <span
              className="playback-row-swatch playback-row-swatch--ground-truth"
              aria-hidden
            />
            <span className="playback-row-label-text">
              <span className="playback-row-label-title">Ground Truth</span>
              <span className="playback-row-label-subtitle">Actual Playing</span>
            </span>
          </div>

          <div className="playback-body">
            <div className="playback-track-column">
              <div
                ref={gtTrackRef}
                className="playback-stack"
                onPointerDown={(e) => onTrackPointerDown(e, gtTrackRef)}
                role="presentation"
              >
                <div className="playback-track">
                  <div className="playback-track-inactive" aria-hidden />
                  {groundTruthIntervals.map((iv) => {
                    const left = (iv.start / duration) * 100
                    const w = ((iv.end - iv.start) / duration) * 100
                    return (
                      <div
                        key={iv.id}
                        className="playback-interval playback-interval--ground-truth"
                        aria-hidden
                        title="Ground truth Playing"
                        style={{ left: `${left}%`, width: `${w}%` }}
                      />
                    )
                  })}
                </div>

                <div
                  className="playback-playhead"
                  data-playhead
                  style={{ left: `${pct}%` }}
                  onPointerDown={(e) => onPlayheadPointerDown(e, gtTrackRef)}
                >
                  <div className="playback-playhead-triangle" aria-hidden />
                  <div className="playback-playhead-line" aria-hidden />
                </div>
              </div>
            </div>

            <div
              className="playback-coverage playback-right-slot"
              title="Ground truth coverage"
            >
              {`${gtCoveragePct.toFixed(1)}%`}
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
