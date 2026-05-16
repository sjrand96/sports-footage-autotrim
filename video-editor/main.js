const path = require('node:path')
const { spawn } = require('node:child_process')
const { app, BrowserWindow, ipcMain, dialog } = require('electron/main')

const DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://127.0.0.1:5173'
const isDev =
  process.env.ELECTRON_DEV === '1' ||
  (!app.isPackaged && process.env.NODE_ENV === 'development')

function buildFilterComplex(intervals) {
  const parts = []
  const concatInputs = []
  intervals.forEach((iv, i) => {
    const s = Number(iv.start).toFixed(3)
    const e = Number(iv.end).toFixed(3)
    parts.push(`[0:v]trim=start=${s}:end=${e},setpts=PTS-STARTPTS[v${i}]`)
    parts.push(`[0:a]atrim=start=${s}:end=${e},asetpts=PTS-STARTPTS[a${i}]`)
    concatInputs.push(`[v${i}][a${i}]`)
  })
  const n = intervals.length
  parts.push(`${concatInputs.join('')}concat=n=${n}:v=1:a=1[outv][outa]`)
  return parts.join(';')
}

function runFfmpeg(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'pipe'] })
    let stderr = ''
    proc.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })
    proc.on('error', (err) => {
      if (err.code === 'ENOENT') {
        reject(new Error('ffmpeg not found. Install ffmpeg (e.g. brew install ffmpeg).'))
      } else {
        reject(err)
      }
    })
    proc.on('close', (code) => {
      if (code === 0) resolve()
      else reject(new Error(stderr.trim() || `ffmpeg exited with code ${code}`))
    })
  })
}

ipcMain.handle('export-cut-video', async (_event, { inputPath, intervals, suggestedName }) => {
  if (!inputPath || !Array.isArray(intervals) || intervals.length === 0) {
    return { ok: false, error: 'Missing input path or intervals.' }
  }

  const { canceled, filePath } = await dialog.showSaveDialog({
    defaultPath: suggestedName || 'clip_cut.mp4',
    filters: [{ name: 'MP4 video', extensions: ['mp4'] }],
  })
  if (canceled || !filePath) {
    return { ok: false, cancelled: true, error: 'Export cancelled.' }
  }

  const filter = buildFilterComplex(intervals)
  const args = [
    '-y',
    '-i',
    inputPath,
    '-filter_complex',
    filter,
    '-map',
    '[outv]',
    '-map',
    '[outa]',
    '-c:v',
    'libx264',
    '-preset',
    'fast',
    '-crf',
    '20',
    '-c:a',
    'aac',
    '-b:a',
    '128k',
    '-movflags',
    '+faststart',
    filePath,
  ]

  try {
    await runFfmpeg(args)
    return { ok: true, outputPath: filePath }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
})

const createWindow = () => {
  const win = new BrowserWindow({
    width: 960,
    height: 720,
    minWidth: 640,
    minHeight: 480,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: false,
    },
  })

  if (isDev) {
    win.loadURL(DEV_URL)
  } else {
    win.loadFile(path.join(__dirname, 'dist', 'index.html'))
  }
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
