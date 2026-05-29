/**
 * Run LSTM inference on a local video file (Electron desktop app only).
 * @returns {Promise<{ ok: true, csvText: string } | { ok: false, error: string }>}
 */
export async function generatePredictions({ inputPath, checkpointPath, onProgress }) {
  if (!inputPath) {
    return {
      ok: false,
      error: window.electronAPI?.getPathForFile
        ? 'Could not resolve the video file path. Re-open the video and try again.'
        : 'Generate predictions requires the desktop app. Run: npm run electron:dev',
    }
  }
  if (!window.electronAPI?.generatePredictions) {
    return {
      ok: false,
      error: 'Generate predictions requires the desktop app. Run: npm run electron:dev',
    }
  }

  const unsubscribe =
    onProgress && window.electronAPI.onPredictProgress
      ? window.electronAPI.onPredictProgress(onProgress)
      : null

  try {
    return await window.electronAPI.generatePredictions({
      clipPath: inputPath,
      checkpointPath,
    })
  } finally {
    if (unsubscribe) unsubscribe()
  }
}

export function canGeneratePredictions(duration, inputPath) {
  return Number.isFinite(duration) && duration > 0 && Boolean(inputPath)
}

export function predictProgressLabel(line) {
  if (!line) return 'Generating predictions…'
  if (line.includes('extracted_frames') || line.includes('extract_fps')) {
    return 'Extracting frame features…'
  }
  if (line.includes('infer_frames') || line.includes('model_fps')) {
    return 'Running temporal model…'
  }
  if (line.startsWith('wrote ')) return 'Loading predictions…'
  return line
}
