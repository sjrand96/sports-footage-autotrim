# Volleyball Video Editor

Electron + React desktop app for reviewing volleyball clips, editing playing intervals on a timeline, and exporting trimmed cuts.

## Setup

From the repo root, install Python dependencies (inference uses the LSTM model in `models/lstm/`):

```bash
pip install torch torchvision tqdm pandas scikit-learn opencv-python-headless
```

Or install the full repo `requirements.txt`.

From this directory, install Node dependencies:

```bash
npm install
```

## Run

**Desktop app (required for export and model inference):**

```bash
npm run electron:dev
```

This starts Vite and opens Electron. Use **Open video** to load a clip, then **Generate predictions** to run the CNN-LSTM model on the open file.

**Browser-only dev (playback and label import only):**

```bash
npm run dev
```

Export cut video and generate predictions are unavailable in the browser; run `npm run electron:dev` instead.

## Generate predictions

1. Open an `.mp4` in the app.
2. Click **Generate predictions** (Editor or Evaluation mode).
3. The app runs [`models/lstm/predict_one.py`](../models/lstm/predict_one.py) against the default checkpoint (`models/lstm/checkpoints/2026-05-27-14:47-cnn-lstm-3sec-context/best.pt`).
4. Predicted playing intervals appear on the timeline (editable in Editor mode; compared against ground truth in Evaluation mode).

**Requirements:**

- Python 3 on `PATH`, or set `SPORTS_AUTOTRIM_PYTHON` to your venv interpreter (e.g. `/path/to/.venv/bin/python`).
- Repo checkout layout intact (`video-editor/` next to `models/lstm/`).
- First run may download EfficientNet ImageNet weights via torchvision.

Inference is CPU/GPU-bound; status text updates while frames are extracted and the temporal model runs.

## Export cut video

Requires [ffmpeg](https://ffmpeg.org/) on `PATH` (e.g. `brew install ffmpeg`).

In **Editor** mode, after defining playing intervals, use **Export cut video** to concatenate trimmed segments into one MP4.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Vite dev server only |
| `npm run electron:dev` | Vite + Electron (full features) |
| `npm run build` | Production frontend build |
| `npm run start` | Electron with built frontend |
| `npm run electron:preview` | Build then launch Electron |
