const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  /** Absolute path for a File from <input type="file"> (Electron 32+; File.path is not in the renderer). */
  getPathForFile: (file) => webUtils.getPathForFile(file),
  exportCutVideo: (options) => ipcRenderer.invoke('export-cut-video', options),
})
