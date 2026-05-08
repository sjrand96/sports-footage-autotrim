import { useCallback, useRef } from 'react'

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
  const trackRef = useRef(null)

  const timeFromClientX = useCallback(
    (clientX) => {
      const el = trackRef.current
      if (!el || !duration || duration <= 0) return null
      const r = el.getBoundingClientRect()
      const x = Math.min(Math.max(0, clientX - r.left), r.width)
      return (x / r.width) * duration
    },
    [duration],
  )

  const seekFromPointer = useCallback(
    (clientX) => {
      const t = timeFromClientX(clientX)
      if (t != null) onSeek(t)
    },
    [timeFromClientX, onSeek],
  )

  const onTrackPointerDown = (e) => {
    if (e.button !== 0) return
    const t = e.target
    if (t.closest?.('[data-timeline-handle]')) return
    if (t.closest?.('[data-playhead]')) return
    e.preventDefault()
    seekFromPointer(e.clientX)

    const node = e.currentTarget
    node.setPointerCapture(e.pointerId)

    const move = (ev) => {
      seekFromPointer(ev.clientX)
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

  const onPlayheadPointerDown = (e) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const node = e.currentTarget
    node.setPointerCapture(e.pointerId)

    const move = (ev) => {
      seekFromPointer(ev.clientX)
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

  return (
    <div className="playback-row">
      <button
        type="button"
        className="playback-play"
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
            ref={trackRef}
            className="playback-stack"
            onPointerDown={onTrackPointerDown}
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
              {intervals.map((iv) => {
                const left = (iv.start / duration) * 100
                const w = ((iv.end - iv.start) / duration) * 100
                return (
                  <div
                    key={iv.id}
                    className="playback-interval playback-interval--predicted"
                    style={{ left: `${left}%`, width: `${w}%` }}
                  />
                )
              })}

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
                    onPointerDown={(e) => onHandlePointerDown(e, iv.id, 'end')}
                  />,
                ])}
              </div>
            </div>

            <div
              className="playback-playhead"
              data-playhead
              style={{ left: `${pct}%` }}
              onPointerDown={onPlayheadPointerDown}
            >
              <div className="playback-playhead-triangle" aria-hidden />
              <div className="playback-playhead-line" aria-hidden />
            </div>
          </div>
        </div>

        <div className="playback-time">
          <span className="playback-time-current">
            {formatTime(currentTime)}
          </span>
          <span className="playback-time-sep"> / </span>
          <span className="playback-time-duration">{formatTime(duration)}</span>
        </div>
      </div>
    </div>
  )
}

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
