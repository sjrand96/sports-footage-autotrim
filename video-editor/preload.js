const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  /** Absolute path for a File from <input type="file"> (Electron 32+; File.path is not in the renderer). */
  getPathForFile: (file) => webUtils.getPathForFile(file),
  exportCutVideo: (options) => ipcRenderer.invoke('export-cut-video', options),
  generatePredictions: (options) => ipcRenderer.invoke('generate-predictions', options),
  onPredictProgress: (callback) => {
    const listener = (_event, line) => callback(line)
    ipcRenderer.on('predict-progress', listener)
    return () => ipcRenderer.removeListener('predict-progress', listener)
  },
})
