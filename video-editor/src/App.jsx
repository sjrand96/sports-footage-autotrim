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
  const groundTruthIntervalIdRef = useRef(0)
  const [sourceUrl, setSourceUrl] = useState(null)
  const [fileLabel, setFileLabel] = useState('')
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [intervals, setIntervals] = useState([])
  const [groundTruthIntervals, setGroundTruthIntervals] = useState([])
  const [playSelectedOnly, setPlaySelectedOnly] = useState(false)
  const [predictLabelsImportName, setPredictLabelsImportName] = useState('')
  const [groundTruthLabelsImportName, setGroundTruthLabelsImportName] =
    useState('')

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

  const nextGroundTruthIntervalId = useCallback(() => {
    groundTruthIntervalIdRef.current += 1
    return `gt-${groundTruthIntervalIdRef.current}`
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
    setGroundTruthIntervals([])
    setPredictLabelsImportName('')
    setGroundTruthLabelsImportName('')
  }

  const onPickPredictedLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file) return
      if (!Number.isFinite(duration) || duration <= 0) {
        setPredictLabelsImportName('')
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
            setPredictLabelsImportName('')
            return
          }
          const withIds = imported.map((iv) => ({
            id: nextIntervalId(),
            start: iv.start,
            end: iv.end,
          }))
          setIntervals(withIds)
          setPredictLabelsImportName(file.name)
        } catch {
          setPredictLabelsImportName('')
        }
      }
      reader.onerror = () => setPredictLabelsImportName('')
      reader.readAsText(file)
    },
    [duration, fileLabel, nextIntervalId],
  )

  const onPickGroundTruthLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file) return
      if (!Number.isFinite(duration) || duration <= 0) {
        setGroundTruthLabelsImportName('')
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
            setGroundTruthLabelsImportName('')
            return
          }
          const withIds = imported.map((iv) => ({
            id: nextGroundTruthIntervalId(),
            start: iv.start,
            end: iv.end,
          }))
          setGroundTruthIntervals(withIds)
          setGroundTruthLabelsImportName(file.name)
        } catch {
          setGroundTruthLabelsImportName('')
        }
      }
      reader.onerror = () => setGroundTruthLabelsImportName('')
      reader.readAsText(file)
    },
    [duration, fileLabel, nextGroundTruthIntervalId],
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
    setDuration(v.duration)
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
          <div className="viewer-layout">
            <div className="viewer-main">
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
                <div className="import-json-slot import-json-slot--predicted">
                  <label
                    className={`file-button file-button--secondary${!duration ? ' file-button--disabled' : ''}`}
                  >
                    Import predicted labels
                    <input
                      type="file"
                      accept="application/json,.json"
                      onChange={onPickPredictedLabels}
                      disabled={!duration}
                      hidden
                    />
                  </label>
                  {predictLabelsImportName ? (
                    <span
                      className="labels-json-filename"
                      title={predictLabelsImportName}
                    >
                      {predictLabelsImportName}
                    </span>
                  ) : null}
                </div>
                <div className="import-json-slot import-json-slot--truth">
                  <label
                    className={`file-button file-button--secondary${!duration ? ' file-button--disabled' : ''}`}
                  >
                    Import ground truth labels
                    <input
                      type="file"
                      accept="application/json,.json"
                      onChange={onPickGroundTruthLabels}
                      disabled={!duration}
                      hidden
                    />
                  </label>
                  {groundTruthLabelsImportName ? (
                    <span
                      className="labels-json-filename"
                      title={groundTruthLabelsImportName}
                    >
                      {groundTruthLabelsImportName}
                    </span>
                  ) : null}
                </div>
              </div>

              <div className="playback-block">
                <PlaybackTimeline
                  duration={duration}
                  currentTime={currentTime}
                  intervals={intervals}
                  groundTruthIntervals={groundTruthIntervals}
                  isPlaying={isPlaying}
                  onTogglePlay={togglePlay}
                  onSeek={seek}
                  onIntervalBoundaryChange={onIntervalBoundaryChange}
                />
              </div>
            </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
