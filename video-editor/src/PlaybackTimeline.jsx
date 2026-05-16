import { useCallback, useMemo, useRef } from 'react'
import { isPlayingAt, mergeIntervals, totalPlayingSeconds } from './timelineMetrics.js'

export default function PlaybackTimeline({
  mode = 'evaluation',
  duration,
  currentTime,
  intervals,
  groundTruthIntervals = [],
  isPlaying,
  onTogglePlay,
  onSeek,
  onIntervalBoundaryChange,
}) {
  const isEditor = mode === 'editor'
  const masterTrackRef = useRef(null)
  const editorTrackRef = useRef(null)
  const gtTrackRef = useRef(null)
  const predTrackRef = useRef(null)

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

  const onHandlePointerDown = (e, intervalId, edge, trackRef) => {
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
      const t = timeFromClientX(ev.clientX, trackRef.current)
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

  const predMerged = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) return []
    return mergeIntervals(
      (intervals ?? []).map(({ start, end }) => ({ start, end })),
      duration,
    )
  }, [intervals, duration])

  const gtMerged = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) return []
    return mergeIntervals(
      (groundTruthIntervals ?? []).map(({ start, end }) => ({ start, end })),
      duration,
    )
  }, [groundTruthIntervals, duration])

  const predictedCoveragePct = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) return 0
    return Math.min(100, (totalPlayingSeconds(predMerged) / duration) * 100)
  }, [predMerged, duration])
  const gtCoveragePct = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) return 0
    return Math.min(100, (totalPlayingSeconds(gtMerged) / duration) * 100)
  }, [gtMerged, duration])

  const confusion = useMemo(() => {
    const d = duration
    if (!Number.isFinite(d) || d <= 0) {
      return { tp: [], fp: [], fn: [], tn: [] }
    }
    if (predMerged.length === 0 || gtMerged.length === 0) {
      return { tp: [], fp: [], fn: [], tn: [] }
    }

    const breakpoints = new Set([0, d])
    for (const iv of predMerged) {
      breakpoints.add(iv.start)
      breakpoints.add(iv.end)
    }
    for (const iv of gtMerged) {
      breakpoints.add(iv.start)
      breakpoints.add(iv.end)
    }
    const bps = [...breakpoints]
      .filter((x) => x >= 0 && x <= d)
      .sort((a, b) => a - b)

    /** @type {{ start: number, end: number }[]} */
    const tp = []
    /** @type {{ start: number, end: number }[]} */
    const fp = []
    /** @type {{ start: number, end: number }[]} */
    const fn = []
    /** @type {{ start: number, end: number }[]} */
    const tn = []

    for (let i = 0; i < bps.length - 1; i++) {
      const a = bps[i]
      const b = bps[i + 1]
      const slice = b - a
      if (!(slice > 1e-9)) continue
      const t = (a + b) / 2
      const predOn = isPlayingAt(t, predMerged)
      const gtOn = isPlayingAt(t, gtMerged)

      if (predOn && gtOn) tp.push({ start: a, end: b })
      else if (predOn && !gtOn) fp.push({ start: a, end: b })
      else if (!predOn && gtOn) fn.push({ start: a, end: b })
      else tn.push({ start: a, end: b })
    }

    return {
      tp: mergeIntervals(tp, d),
      fp: mergeIntervals(fp, d),
      fn: mergeIntervals(fn, d),
      tn: mergeIntervals(tn, d),
    }
  }, [predMerged, gtMerged, duration])

  const confusionCoveragePct = useMemo(() => {
    if (!Number.isFinite(duration) || duration <= 0) {
      return { tp: 0, fp: 0, fn: 0, tn: 0 }
    }
    const d = duration
    return {
      tp: Math.min(100, (totalPlayingSeconds(confusion.tp) / d) * 100),
      fp: Math.min(100, (totalPlayingSeconds(confusion.fp) / d) * 100),
      fn: Math.min(100, (totalPlayingSeconds(confusion.fn) / d) * 100),
      tn: Math.min(100, (totalPlayingSeconds(confusion.tn) / d) * 100),
    }
  }, [confusion, duration])

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

      {/* Editor: single editable labels row with trim handles */}
      {isEditor ? (
        <div className="playback-row playback-row--secondary playback-row--editor">
          <div
            className="playback-row-label playback-left-slot"
            aria-label="Playing segments row label"
          >
            <span className="playback-row-swatch" aria-hidden />
            <span className="playback-row-label-text">
              <span className="playback-row-label-title">Playing</span>
              <span className="playback-row-label-subtitle">Drag handles to adjust</span>
            </span>
          </div>

          <div className="playback-body">
            <div className="playback-track-column">
              <div
                ref={editorTrackRef}
                className="playback-stack"
                onPointerDown={(e) => onTrackPointerDown(e, editorTrackRef)}
                role="presentation"
              >
                <div className="playback-track playback-track--editable">
                  <div className="playback-track-inactive" aria-hidden />
                  {intervals.map((iv) => {
                    const left = (iv.start / duration) * 100
                    const w = ((iv.end - iv.start) / duration) * 100
                    return (
                      <div
                        key={iv.id}
                        className="playback-interval playback-interval--accent"
                        aria-hidden
                        title="Playing segment"
                        style={{ left: `${left}%`, width: `${w}%` }}
                      />
                    )
                  })}
                  {intervals.length > 0 ? (
                    <div className="playback-handles" aria-hidden>
                      {intervals.map((iv) => {
                        const startPct = (iv.start / duration) * 100
                        const endPct = (iv.end / duration) * 100
                        return (
                          <span key={iv.id} className="playback-handle-pair">
                            <button
                              type="button"
                              className="playback-handle"
                              data-timeline-handle
                              style={{ left: `${startPct}%` }}
                              aria-label={`Start of segment at ${formatTime(iv.start)}`}
                              onPointerDown={(e) =>
                                onHandlePointerDown(e, iv.id, 'start', editorTrackRef)
                              }
                            />
                            <button
                              type="button"
                              className="playback-handle"
                              data-timeline-handle
                              style={{ left: `${endPct}%` }}
                              aria-label={`End of segment at ${formatTime(iv.end)}`}
                              onPointerDown={(e) =>
                                onHandlePointerDown(e, iv.id, 'end', editorTrackRef)
                              }
                            />
                          </span>
                        )
                      })}
                    </div>
                  ) : null}
                </div>

                <div
                  className="playback-playhead playback-playhead--no-triangle"
                  data-playhead
                  style={{ left: `${pct}%` }}
                  onPointerDown={(e) => onPlayheadPointerDown(e, editorTrackRef)}
                >
                  <div className="playback-playhead-triangle" aria-hidden />
                  <div className="playback-playhead-line" aria-hidden />
                </div>
              </div>
            </div>

            <div className="playback-coverage playback-right-slot" title="Playing coverage">
              {`${predictedCoveragePct.toFixed(1)}%`}
            </div>
          </div>
        </div>
      ) : null}

      {/* Evaluation: ground truth timeline row */}
      {!isEditor && groundTruthIntervals && groundTruthIntervals.length > 0 ? (
        <div className="playback-row playback-row--secondary playback-row--ground-truth">
          <div
            className="playback-row-label playback-left-slot"
            aria-label="Ground truth row label"
          >
            <span
              className="playback-row-swatch"
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
                        className="playback-interval playback-interval--accent"
                        aria-hidden
                        title="Ground truth Playing"
                        style={{ left: `${left}%`, width: `${w}%` }}
                      />
                    )
                  })}
                </div>

                <div
                  className="playback-playhead playback-playhead--no-triangle"
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

      {/* Evaluation: predicted timeline row */}
      {!isEditor && intervals && intervals.length > 0 ? (
        <div className="playback-row playback-row--secondary playback-row--predicted">
          <div
            className="playback-row-label playback-left-slot"
            aria-label="Predicted row label"
          >
            <span
              className="playback-row-swatch"
              aria-hidden
            />
            <span className="playback-row-label-text">
              <span className="playback-row-label-title">Predicted Labels</span>
              <span className="playback-row-label-subtitle">Predicted Playing</span>
            </span>
          </div>

          <div className="playback-body">
            <div className="playback-track-column">
              <div
                className="playback-stack"
                ref={predTrackRef}
                onPointerDown={(e) => onTrackPointerDown(e, predTrackRef)}
                role="presentation"
              >
                <div className="playback-track">
                  <div className="playback-track-inactive" aria-hidden />
                  {intervals.map((iv) => {
                    const left = (iv.start / duration) * 100
                    const w = ((iv.end - iv.start) / duration) * 100
                    return (
                      <div
                        key={iv.id}
                        className="playback-interval playback-interval--accent"
                        aria-hidden
                        title="Predicted Playing"
                        style={{ left: `${left}%`, width: `${w}%` }}
                      />
                    )
                  })}
                </div>

                <div
                  className="playback-playhead playback-playhead--no-triangle"
                  data-playhead
                  style={{ left: `${pct}%` }}
                  onPointerDown={(e) => onPlayheadPointerDown(e, predTrackRef)}
                >
                  <div className="playback-playhead-triangle" aria-hidden />
                  <div className="playback-playhead-line" aria-hidden />
                </div>
              </div>
            </div>

            <div className="playback-coverage playback-right-slot" title="Predicted coverage">
              {`${predictedCoveragePct.toFixed(1)}%`}
            </div>
          </div>
        </div>
      ) : null}

      {/* Evaluation: confusion matrix timelines */}
      {!isEditor && predMerged.length > 0 && gtMerged.length > 0 ? (
        <>
          <div className="playback-divider" role="separator" aria-hidden />

          <ConfusionRow
            duration={duration}
            pct={pct}
            labelTitle="True Positive"
            labelSubtitle="Pred Playing & Actual Playing"
            rowClassName="playback-row--tp"
            intervals={confusion.tp}
            coveragePct={confusionCoveragePct.tp}
            onTrackPointerDown={onTrackPointerDown}
            onPlayheadPointerDown={onPlayheadPointerDown}
            timeRef={masterTrackRef}
          />
          <ConfusionRow
            duration={duration}
            pct={pct}
            labelTitle="False Positive"
            labelSubtitle="Pred Playing & Actual Downtime"
            rowClassName="playback-row--fp"
            intervals={confusion.fp}
            coveragePct={confusionCoveragePct.fp}
            onTrackPointerDown={onTrackPointerDown}
            onPlayheadPointerDown={onPlayheadPointerDown}
            timeRef={masterTrackRef}
          />
          <ConfusionRow
            duration={duration}
            pct={pct}
            labelTitle="False Negative"
            labelSubtitle="Pred Downtime & Actual Playing"
            rowClassName="playback-row--fn"
            intervals={confusion.fn}
            coveragePct={confusionCoveragePct.fn}
            onTrackPointerDown={onTrackPointerDown}
            onPlayheadPointerDown={onPlayheadPointerDown}
            timeRef={masterTrackRef}
          />
          <ConfusionRow
            duration={duration}
            pct={pct}
            labelTitle="True Negative"
            labelSubtitle="Pred Downtime & Actual Downtime"
            rowClassName="playback-row--tn"
            intervals={confusion.tn}
            coveragePct={confusionCoveragePct.tn}
            onTrackPointerDown={onTrackPointerDown}
            onPlayheadPointerDown={onPlayheadPointerDown}
            timeRef={masterTrackRef}
          />
        </>
      ) : null}
    </>
  )
}

function ConfusionRow({
  duration,
  pct,
  labelTitle,
  labelSubtitle,
  rowClassName,
  intervals,
  coveragePct,
  onTrackPointerDown,
  onPlayheadPointerDown,
  timeRef,
}) {
  const trackRef = useRef(null)
  return (
    <div className={`playback-row playback-row--secondary ${rowClassName}`}>
      <div className="playback-row-label playback-left-slot" aria-label={labelTitle}>
        <span className="playback-row-swatch" aria-hidden />
        <span className="playback-row-label-text">
          <span className="playback-row-label-title">{labelTitle}</span>
          <span className="playback-row-label-subtitle">{labelSubtitle}</span>
        </span>
      </div>

      <div className="playback-body">
        <div className="playback-track-column">
          <div
            ref={trackRef}
            className="playback-stack"
            onPointerDown={(e) => onTrackPointerDown(e, trackRef)}
            role="presentation"
          >
            <div className="playback-track">
              <div className="playback-track-inactive" aria-hidden />
              {(intervals ?? []).map((iv, idx) => {
                const left = (iv.start / duration) * 100
                const w = ((iv.end - iv.start) / duration) * 100
                return (
                  <div
                    key={`${iv.start}-${iv.end}-${idx}`}
                    className="playback-interval playback-interval--accent"
                    aria-hidden
                    style={{ left: `${left}%`, width: `${w}%` }}
                  />
                )
              })}
            </div>

            <div
              className="playback-playhead playback-playhead--no-triangle"
              data-playhead
              style={{ left: `${pct}%` }}
              onPointerDown={(e) => onPlayheadPointerDown(e, trackRef)}
            >
              <div className="playback-playhead-triangle" aria-hidden />
              <div className="playback-playhead-line" aria-hidden />
            </div>
          </div>
        </div>

        <div className="playback-coverage playback-right-slot" title="Coverage">
          {`${Number.isFinite(coveragePct) ? coveragePct.toFixed(1) : '0.0'}%`}
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
