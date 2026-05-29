import { useCallback, useEffect, useRef, useState } from 'react'
import PlaybackTimeline from './PlaybackTimeline.jsx'
import {
  gatePlaySelectedOnly,
  snapTimeForSelectedPlayStart,
  sortIntervals,
} from './selectedIntervalPlayback.js'
import { playingIntervalsSecondsFromFramePredictionsCsv } from './framePredictionsImport.js'
import { playingIntervalsSecondsFromLabelJson } from './labelStudioImport.js'
import {
  canExportCut,
  defaultCutOutputName,
  exportCutVideo,
  getLocalVideoPath,
} from './exportCutVideo.js'
import {
  canGeneratePredictions,
  generatePredictions,
  predictProgressLabel,
} from './generatePredictions.js'
import { smoothPlayingIntervals } from './intervalSmoothing.js'
import './App.css'

const MIN_INTERVAL_SEC = 0.05

function applyImportedIntervals(imported, nextId) {
  return imported.map((iv) => ({
    id: nextId(),
    start: iv.start,
    end: iv.end,
  }))
}

function smoothIntervalsWithNewIds(intervals, durationSec, nextId) {
  if (intervals.length === 0) return []
  if (!Number.isFinite(durationSec) || durationSec <= 0) return intervals
  const smoothed = smoothPlayingIntervals(intervals, durationSec)
  return smoothed.map((iv) => ({
    id: nextId(),
    start: iv.start,
    end: iv.end,
  }))
}

function cloneIntervals(intervals) {
  return intervals.map((iv) => ({ id: iv.id, start: iv.start, end: iv.end }))
}

function updateIntervalBoundary(prev, id, edge, rawTime, durationSec) {
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
  } else {
    const minE = cur.start + MIN_INTERVAL_SEC
    const maxE = after ? after.start - MIN_INTERVAL_SEC : durationSec
    cur.end = Math.min(Math.max(rawTime, minE), maxE)
  }

  const out = [...sorted]
  out[i] = cur
  return out
}

export default function App() {
  const videoRef = useRef(null)
  const sourceFileRef = useRef(null)
  const sourceFilePathRef = useRef(null)
  const intervalIdRef = useRef(0)
  const groundTruthIntervalIdRef = useRef(0)
  const [appMode, setAppMode] = useState('editor')
  const [sourceUrl, setSourceUrl] = useState(null)
  const [fileLabel, setFileLabel] = useState('')
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [editorIntervals, setEditorIntervals] = useState([])
  /** Unsmoothed playing segments; restored when smoothing is turned off. */
  const [rawEditorIntervals, setRawEditorIntervals] = useState([])
  const [predictedIntervals, setPredictedIntervals] = useState([])
  const [groundTruthIntervals, setGroundTruthIntervals] = useState([])
  const [showGroundTruth, setShowGroundTruth] = useState(false)
  const [playSelectedOnly, setPlaySelectedOnly] = useState(false)
  const [smoothIntervals, setSmoothIntervals] = useState(true)
  const [predictLabelsImportName, setPredictLabelsImportName] = useState('')
  const [editorLabelsImportName, setEditorLabelsImportName] = useState('')
  const [groundTruthLabelsImportName, setGroundTruthLabelsImportName] =
    useState('')
  const [labelsImportError, setLabelsImportError] = useState('')
  const [exportStatus, setExportStatus] = useState('')
  const [isExporting, setIsExporting] = useState(false)
  const [predictStatus, setPredictStatus] = useState('')
  const [isPredicting, setIsPredicting] = useState(false)

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
    sourceFileRef.current = file
    sourceFilePathRef.current = getLocalVideoPath(file)
    const url = URL.createObjectURL(file)
    setSourceUrl(url)
    setFileLabel(file.name)
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)
    setEditorIntervals([])
    setRawEditorIntervals([])
    setPredictedIntervals([])
    setGroundTruthIntervals([])
    setShowGroundTruth(false)
    setSmoothIntervals(true)
    setPlaySelectedOnly(false)
    setPredictLabelsImportName('')
    setEditorLabelsImportName('')
    setGroundTruthLabelsImportName('')
    setLabelsImportError('')
    setExportStatus('')
    setPredictStatus('')
  }

  const importLabelsFromFile = useCallback(
    (file, raw, { target }) => {
      const isCsv = /\.csv$/i.test(file.name)
      let imported
      let csvGroundTruth
      let hasGroundTruthColumn = false
      let error

      if (isCsv) {
        const csvResult = playingIntervalsSecondsFromFramePredictionsCsv(
          raw,
          fileLabel,
          duration,
        )
        imported = csvResult.intervals
        csvGroundTruth = csvResult.groundTruthIntervals
        hasGroundTruthColumn = csvResult.hasGroundTruthColumn === true
        error = csvResult.error
      } else {
        const jsonResult = playingIntervalsSecondsFromLabelJson(
          raw,
          fileLabel,
          duration,
        )
        imported = jsonResult.intervals
        error = jsonResult.error
      }

      if (error) {
        if (target === 'editor') setEditorLabelsImportName('')
        else if (target === 'predicted') setPredictLabelsImportName('')
        else setGroundTruthLabelsImportName('')
        setLabelsImportError(error)
        return false
      }

      const withIds = applyImportedIntervals(
        imported,
        target === 'groundTruth' ? nextGroundTruthIntervalId : nextIntervalId,
      )

      if (target === 'editor') {
        setRawEditorIntervals(withIds)
        setEditorIntervals(
          smoothIntervals
            ? smoothIntervalsWithNewIds(withIds, duration, nextIntervalId)
            : withIds,
        )
        setGroundTruthIntervals([])
        setShowGroundTruth(false)
        setEditorLabelsImportName(file.name)
        setPredictLabelsImportName('')
        setGroundTruthLabelsImportName('')
      } else if (target === 'predicted') {
        setPredictedIntervals(withIds)
        const editorRaw = applyImportedIntervals(imported, nextIntervalId)
        setRawEditorIntervals(editorRaw)
        setEditorIntervals(
          smoothIntervals
            ? smoothIntervalsWithNewIds(editorRaw, duration, nextIntervalId)
            : editorRaw,
        )
        setPredictLabelsImportName(file.name)
        if (hasGroundTruthColumn) {
          setGroundTruthIntervals(
            applyImportedIntervals(
              csvGroundTruth ?? [],
              nextGroundTruthIntervalId,
            ),
          )
          setShowGroundTruth(true)
          setGroundTruthLabelsImportName('')
        } else {
          setGroundTruthIntervals([])
          setShowGroundTruth(false)
          setGroundTruthLabelsImportName('')
        }
      } else {
        setGroundTruthIntervals(withIds)
        setShowGroundTruth(true)
        setGroundTruthLabelsImportName(file.name)
      }

      setLabelsImportError('')
      return true
    },
    [duration, fileLabel, nextGroundTruthIntervalId, nextIntervalId, smoothIntervals],
  )

  const onSmoothIntervalsChange = useCallback(
    (enabled) => {
      setSmoothIntervals(enabled)
      if (rawEditorIntervals.length === 0) return
      if (enabled) {
        setEditorIntervals(
          smoothIntervalsWithNewIds(rawEditorIntervals, duration, nextIntervalId),
        )
      } else {
        setEditorIntervals(cloneIntervals(rawEditorIntervals))
      }
    },
    [duration, nextIntervalId, rawEditorIntervals],
  )

  const onPickEditorLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file || !Number.isFinite(duration) || duration <= 0) {
        setEditorLabelsImportName('')
        return
      }
      const reader = new FileReader()
      reader.onload = () => importLabelsFromFile(file, reader.result, { target: 'editor' })
      reader.onerror = () => setEditorLabelsImportName('')
      reader.readAsText(file)
    },
    [duration, importLabelsFromFile],
  )

  const onPickPredictedLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file || !Number.isFinite(duration) || duration <= 0) {
        setPredictLabelsImportName('')
        return
      }
      const reader = new FileReader()
      reader.onload = () => importLabelsFromFile(file, reader.result, { target: 'predicted' })
      reader.onerror = () => setPredictLabelsImportName('')
      reader.readAsText(file)
    },
    [duration, importLabelsFromFile],
  )

  const onPickGroundTruthLabels = useCallback(
    (e) => {
      const file = e.target.files?.[0]
      e.target.value = ''
      if (!file || !Number.isFinite(duration) || duration <= 0) {
        setGroundTruthLabelsImportName('')
        return
      }
      const reader = new FileReader()
      reader.onload = () => importLabelsFromFile(file, reader.result, { target: 'groundTruth' })
      reader.onerror = () => setGroundTruthLabelsImportName('')
      reader.readAsText(file)
    },
    [duration, importLabelsFromFile],
  )

  const onExportCutVideo = useCallback(async () => {
    setExportStatus('')
    setLabelsImportError('')
    const file = sourceFileRef.current
    const inputPath =
      sourceFilePathRef.current ?? getLocalVideoPath(file)
    if (!canExportCut(editorIntervals)) {
      setExportStatus('Add at least one playing interval before exporting.')
      return
    }

    setIsExporting(true)
    try {
      const result = await exportCutVideo({
        inputPath,
        intervals: editorIntervals,
        suggestedName: defaultCutOutputName(fileLabel),
      })
      if (result.ok) {
        setExportStatus(`Saved to ${result.outputPath}`)
      } else if (!result.cancelled) {
        setExportStatus(result.error)
      }
    } finally {
      setIsExporting(false)
    }
  }, [fileLabel, editorIntervals])

  const onGeneratePredictions = useCallback(async () => {
    setPredictStatus('')
    setLabelsImportError('')
    const file = sourceFileRef.current
    const inputPath = sourceFilePathRef.current ?? getLocalVideoPath(file)
    if (!canGeneratePredictions(duration, inputPath)) {
      setLabelsImportError(
        inputPath
          ? 'Wait for the video to finish loading.'
          : 'Generate predictions requires the desktop app. Run: npm run electron:dev',
      )
      return
    }

    setIsPredicting(true)
    setPredictStatus('Generating predictions…')
    try {
      const result = await generatePredictions({
        inputPath,
        onProgress: (line) => setPredictStatus(predictProgressLabel(line)),
      })
      if (!result.ok) {
        setLabelsImportError(result.error)
        setPredictStatus('')
        return
      }

      const stem = fileLabel.replace(/\.mp4$/i, '')
      const syntheticFile = { name: `${stem}_predictions.csv` }
      const target = appMode === 'editor' ? 'editor' : 'predicted'
      const ok = importLabelsFromFile(syntheticFile, result.csvText, { target })
      if (ok) {
        setPredictStatus('Predictions loaded.')
      } else {
        setPredictStatus('')
      }
    } finally {
      setIsPredicting(false)
    }
  }, [appMode, duration, fileLabel, importLabelsFromFile])

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (!v) return
    setCurrentTime(v.currentTime)
    if (playSelectedOnly && !v.paused) {
      gatePlaySelectedOnly(v, editorIntervals)
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
      setEditorIntervals((prev) => {
        const out = updateIntervalBoundary(prev, id, edge, rawTime, duration)
        if (out === prev) return prev
        const iv = out.find((x) => x.id === id)
        if (iv) seek(edge === 'start' ? iv.start : iv.end)
        if (!smoothIntervals) setRawEditorIntervals(cloneIntervals(out))
        return out
      })
    },
    [duration, seek, smoothIntervals],
  )

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      if (playSelectedOnly) {
        if (sortIntervals(editorIntervals).length === 0) return
        const snap = snapTimeForSelectedPlayStart(v.currentTime, editorIntervals)
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
    gatePlaySelectedOnly(v, editorIntervals)
  }, [playSelectedOnly, editorIntervals])

  const isEditor = appMode === 'editor'
  const exportReady = isEditor && duration > 0 && canExportCut(editorIntervals)
  const predictReady =
    duration > 0 &&
    canGeneratePredictions(
      duration,
      sourceFilePathRef.current ?? getLocalVideoPath(sourceFileRef.current),
    )

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="title">Volleyball Video Editor</h1>
        <div className="app-header-actions">
          <div className="mode-toggle" role="group" aria-label="Application mode">
            <button
              type="button"
              className={`mode-toggle-btn${isEditor ? ' mode-toggle-btn--active' : ''}`}
              aria-pressed={isEditor}
              onClick={() => setAppMode('editor')}
            >
              Editor
            </button>
            <button
              type="button"
              className={`mode-toggle-btn${!isEditor ? ' mode-toggle-btn--active' : ''}`}
              aria-pressed={!isEditor}
              onClick={() => setAppMode('evaluation')}
            >
              Evaluation
            </button>
          </div>
          <label className="file-button">
            Open video
            <input type="file" accept="video/*" onChange={onPickFile} hidden />
          </label>
        </div>
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
                  controls
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

                  {isEditor ? (
                    <>
                      <div className="import-json-slot import-json-slot--editor">
                        <label
                          className={`file-button file-button--secondary${!duration ? ' file-button--disabled' : ''}`}
                        >
                          Import labels
                          <input
                            type="file"
                            accept="application/json,.json,text/csv,.csv"
                            onChange={onPickEditorLabels}
                            disabled={!duration}
                            hidden
                          />
                        </label>
                        {editorLabelsImportName ? (
                          <span
                            className="labels-json-filename"
                            title={editorLabelsImportName}
                          >
                            {editorLabelsImportName}
                          </span>
                        ) : null}
                      </div>
                      <button
                        type="button"
                        className={`file-button file-button--secondary${!predictReady || isPredicting ? ' file-button--disabled' : ''}`}
                        disabled={!predictReady || isPredicting}
                        onClick={onGeneratePredictions}
                      >
                        {isPredicting ? 'Generating…' : 'Generate predictions'}
                      </button>
                      <button
                        type="button"
                        className={`file-button file-button--secondary${!exportReady || isExporting ? ' file-button--disabled' : ''}`}
                        disabled={!exportReady || isExporting}
                        onClick={onExportCutVideo}
                      >
                        {isExporting ? 'Exporting…' : 'Export cut video'}
                      </button>
                    </>
                  ) : (
                    <>
                      <div className="import-json-slot import-json-slot--predicted">
                        <label
                          className={`file-button file-button--secondary${!duration ? ' file-button--disabled' : ''}`}
                        >
                          Import predicted labels
                          <input
                            type="file"
                            accept="application/json,.json,text/csv,.csv"
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
                      <button
                        type="button"
                        className={`file-button file-button--secondary${!predictReady || isPredicting ? ' file-button--disabled' : ''}`}
                        disabled={!predictReady || isPredicting}
                        onClick={onGeneratePredictions}
                      >
                        {isPredicting ? 'Generating…' : 'Generate predictions'}
                      </button>
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
                    </>
                  )}
                </div>

                {labelsImportError ? (
                  <p className="labels-import-error" role="alert">
                    {labelsImportError}
                  </p>
                ) : null}

                {exportStatus ? (
                  <p
                    className={`labels-import-status${exportStatus.startsWith('Saved') ? ' labels-import-status--ok' : ''}`}
                    role="status"
                  >
                    {exportStatus}
                  </p>
                ) : null}

                {predictStatus ? (
                  <p
                    className={`labels-import-status${predictStatus === 'Predictions loaded.' ? ' labels-import-status--ok' : ''}`}
                    role="status"
                  >
                    {predictStatus}
                  </p>
                ) : null}

                <div className="playback-block">
                  {isEditor ? (
                    <div className="playback-toggles">
                      <label className="playback-selected-toggle">
                        <input
                          type="checkbox"
                          checked={smoothIntervals}
                          onChange={(e) => onSmoothIntervalsChange(e.target.checked)}
                        />
                        Smooth segment boundaries
                      </label>
                      <label className="playback-selected-toggle">
                        <input
                          type="checkbox"
                          checked={playSelectedOnly}
                          onChange={(e) => setPlaySelectedOnly(e.target.checked)}
                          disabled={editorIntervals.length === 0}
                        />
                        Play selected segments only
                      </label>
                    </div>
                  ) : null}
                  <PlaybackTimeline
                    mode={appMode}
                    duration={duration}
                    currentTime={currentTime}
                    editorIntervals={editorIntervals}
                    predictedIntervals={predictedIntervals}
                    groundTruthIntervals={groundTruthIntervals}
                    showGroundTruth={showGroundTruth}
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
