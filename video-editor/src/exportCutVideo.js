import { sortIntervals } from './selectedIntervalPlayback.js'

export function defaultCutOutputName(fileLabel) {
  const base = (fileLabel || 'clip').replace(/\.[^.]+$/, '')
  return `${base}_cut.mp4`
}

export function canExportCut(intervals) {
  return sortIntervals(intervals).length > 0
}

/** Resolve disk path for a user-picked video File (Electron desktop app only). */
export function getLocalVideoPath(file) {
  if (!file) return null
  if (window.electronAPI?.getPathForFile) {
    try {
      return window.electronAPI.getPathForFile(file)
    } catch {
      return null
    }
  }
  return file.path ?? null
}

/**
 * Export trimmed segments concatenated into one file (Electron desktop app).
 * @returns {Promise<{ ok: true, outputPath: string } | { ok: false, error: string, cancelled?: boolean }>}
 */
export async function exportCutVideo({ inputPath, intervals, suggestedName }) {
  if (!inputPath) {
    return {
      ok: false,
      error: window.electronAPI?.getPathForFile
        ? 'Could not resolve the video file path. Re-open the video and try again.'
        : 'Export cut video requires the desktop app. Run: npm run electron:dev',
    }
  }
  if (!canExportCut(intervals)) {
    return { ok: false, error: 'Add at least one playing interval before exporting.' }
  }
  if (!window.electronAPI?.exportCutVideo) {
    return {
      ok: false,
      error: 'Export cut video requires the desktop app. Run: npm run electron:dev',
    }
  }

  const sorted = sortIntervals(intervals).map(({ start, end }) => ({ start, end }))
  return window.electronAPI.exportCutVideo({
    inputPath,
    intervals: sorted,
    suggestedName: suggestedName || 'clip_cut.mp4',
  })
}
