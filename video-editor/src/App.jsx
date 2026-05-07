import { useCallback, useEffect, useRef, useState } from 'react'
import PlaybackTimeline from './PlaybackTimeline.jsx'
import {
  gatePlaySelectedOnly,
  snapTimeForSelectedPlayStart,
  sortIntervals,
} from './selectedIntervalPlayback.js'
import {
  parseLabelStudioTasksJson,
  playingIntervalsSecondsForExport,
} from './labelStudioImport.js'
import './App.css'

const MIN_INTERVAL_SEC = 0.05

export default function App() {
  const videoRef = useRef(null)
  const intervalIdRef = useRef(0)
  const [sourceUrl, setSourceUrl] = useState(null)
  const [fileLabel, setFileLabel] = useState('')
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [intervals, setIntervals] = useState([])
  const [playSelectedOnly, setPlaySelectedOnly] = useState(false)
  const [labelsHint, setLabelsHint] = useState('')

  // clean up the old URL when the source URL changes
  const revokeUrl = useCallback((url) => {
    if (url && url.startsWith('blob:')) {
      URL.revokeObjectURL(url)
    }
  }, [])

  useEffect(() => {
    return () => revokeUrl(sourceUrl)
  }, [sourceUrl, revokeUrl])

  const nextIntervalId = useCallback(() => {
    intervalIdRef.current += 1
    return `iv-${intervalIdRef.current}`
  }, [])

  const onPickFile = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    revokeUrl(sourceUrl)
    const url = URL.createObjectURL(file)
    setSourceUrl(url)
    setFileLabel(file.name)
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)
    setIntervals([])
    setLabelsHint('')
  }

  const onPickGroundTruthLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file) return
      if (!Number.isFinite(duration) || duration <= 0) {
        setLabelsHint('Wait until the clip has loaded, then try again.')
        return
      }
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const tasks = parseLabelStudioTasksJson(reader.result)
          const { intervals: imported, error } = playingIntervalsSecondsForExport(
            tasks,
            fileLabel,
            duration,
          )
          if (error) {
            setLabelsHint(error)
            return
          }
          const withIds = imported.map((iv) => ({
            id: nextIntervalId(),
            start: iv.start,
            end: iv.end,
          }))
          setIntervals(withIds)
          setLabelsHint(
            `Imported ${withIds.length} Playing segment${withIds.length !== 1 ? 's' : ''} from the JSON.`,
          )
        } catch (err) {
          setLabelsHint(err?.message || 'Could not read that JSON export.')
        }
      }
      reader.onerror = () => setLabelsHint('Failed to read file.')
      reader.readAsText(file)
    },
    [duration, fileLabel, nextIntervalId],
  )

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (!v) return
    setCurrentTime(v.currentTime)
    if (playSelectedOnly && !v.paused) {
      gatePlaySelectedOnly(v, intervals)
    }
  }

  const onLoadedMetadata = () => {
    const v = videoRef.current
    if (!v || !Number.isFinite(v.duration) || v.duration <= 0) return
    const d = v.duration
    setDuration(d)

    const minGap = MIN_INTERVAL_SEC * 2
    const minSeg = MIN_INTERVAL_SEC * 2
    if (d <= minGap + minSeg * 2) {
      setIntervals([{ id: nextIntervalId(), start: 0, end: d }])
      return
    }

    const firstEnd = Math.max(minSeg, d * 0.05)
    const secondStart = Math.min(d - minSeg, d * 0.95)
    if (secondStart <= firstEnd + minGap) {
      const mid = d / 2
      setIntervals([
        { id: nextIntervalId(), start: 0, end: Math.max(minSeg, mid - minGap / 2) },
        {
          id: nextIntervalId(),
          start: Math.min(d - minSeg, mid + minGap / 2),
          end: d,
        },
      ])
      return
    }

    setIntervals([
      { id: nextIntervalId(), start: 0, end: firstEnd },
      { id: nextIntervalId(), start: secondStart, end: d },
    ])
  }

  const seek = useCallback((t) => {
    const v = videoRef.current
    if (!v || !Number.isFinite(t)) return
    const d = v.duration
    const clamped = Math.min(Math.max(0, t), Number.isFinite(d) && d > 0 ? d : t)
    v.currentTime = clamped
    setCurrentTime(clamped)
  }, [])

  const onIntervalBoundaryChange = useCallback(
    (id, edge, rawTime) => {
      if (!Number.isFinite(duration) || duration <= 0) return
      setIntervals((prev) => {
        const sorted = [...prev].sort((a, b) => a.start - b.start)
        const i = sorted.findIndex((x) => x.id === id)
        if (i < 0) return prev
        const cur = { ...sorted[i] }
        const before = sorted[i - 1]
        const after = sorted[i + 1]

        if (edge === 'start') {
          const minS = before ? before.end + MIN_INTERVAL_SEC : 0
          const maxS = cur.end - MIN_INTERVAL_SEC
          cur.start = Math.min(Math.max(rawTime, minS), maxS)
          seek(cur.start)
        } else {
          const minE = cur.start + MIN_INTERVAL_SEC
          const maxE = after ? after.start - MIN_INTERVAL_SEC : duration
          cur.end = Math.min(Math.max(rawTime, minE), maxE)
          seek(cur.end)
        }

        const out = [...sorted]
        out[i] = cur
        return out
      })
    },
    [duration, seek],
  )

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      if (playSelectedOnly) {
        if (sortIntervals(intervals).length === 0) return
        const snap = snapTimeForSelectedPlayStart(v.currentTime, intervals)
        if (snap != null) v.currentTime = snap
      }
      void v.play()
    } else {
      v.pause()
    }
  }

  useEffect(() => {
    if (!playSelectedOnly) return
    const v = videoRef.current
    if (!v || v.paused) return
    gatePlaySelectedOnly(v, intervals)
  }, [playSelectedOnly, intervals])

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="title">Volleyball Video Editor</h1>
        <label className="file-button">
          Open video
          <input type="file" accept="video/*" onChange={onPickFile} hidden />
        </label>
      </header>

      <main className="main">
        {!sourceUrl ? (
          <div className="empty-state">
            <p>Choose a video file to preview and scrub the timeline.</p>
            <label className="file-button large">
              Select video
              <input type="file" accept="video/*" onChange={onPickFile} hidden />
            </label>
          </div>
        ) : (
          <div className="viewer">
            <div className="video-wrap">
              <video
                ref={videoRef}
                className="video"
                src={sourceUrl}
                playsInline
                draggable={false}
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={onLoadedMetadata}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
            </div>

            <div className="controls">
              <div className="file-row">
                <span className="file-name" title={fileLabel}>
                  {fileLabel}
                </span>
                <label
                  className={`file-button file-button--secondary${!duration ? ' file-button--disabled' : ''}`}
                >
                  Import ground-truth labels
                  <input
                    type="file"
                    accept="application/json,.json"
                    onChange={onPickGroundTruthLabels}
                    disabled={!duration}
                    hidden
                  />
                </label>
              </div>
              {labelsHint ? (
                <p
                  className="labels-import-hint"
                  role="status"
                  aria-live="polite"
                >
                  {labelsHint}
                </p>
              ) : null}

              <div className="playback-block">
                <PlaybackTimeline
                  duration={duration}
                  currentTime={currentTime}
                  intervals={intervals}
                  isPlaying={isPlaying}
                  onTogglePlay={togglePlay}
                  onSeek={seek}
                  onIntervalBoundaryChange={onIntervalBoundaryChange}
                />
                <label className="playback-selected-toggle">
                  <input
                    type="checkbox"
                    checked={playSelectedOnly}
                    onChange={(e) => setPlaySelectedOnly(e.target.checked)}
                    disabled={!duration || sortIntervals(intervals).length === 0}
                  />
                  <span>Play only selected intervals</span>
                </label>
              </div>

              <p className="hint">
                Drag the playhead to scrub. Drag the handles to adjust active
                intervals (shown highlighted on the bar). With &quot;Play only
                selected intervals&quot;, playback skips gaps and loops from the
                end of the last highlight back to the first.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
