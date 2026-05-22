# LSTM playing / inactive classifier

Frame-level binary classifier: **playing (active)** vs **inactive** on volleyball clips. A frozen **EfficientNetV2-M** encodes each video frame once; a small **BiLSTM** reads a 1-second window of those features and predicts the label at the center frame.

Upstream labeling and ingest live under [`data/`](../../data/) and [`data_labeling/`](../../data_labeling/).

---

## Pipeline

```mermaid
flowchart LR
  preprocess[data/preprocess_labels.py]
  extract[extract_features.py]
  train[train.py]
  labelCsv[frame_labels.csv]
  featCache[preprocessed_features/]
  ckpt[checkpoints/best.pt]

  preprocess --> labelCsv
  labelCsv --> extract --> featCache
  labelCsv --> train
  featCache --> train --> ckpt
```

**Run order (repo root, venv active):**

```bash
python data/preprocess_labels.py
python data/train_test_split.py
python models/lstm/extract_features.py --device mps --batch-size 32
python models/lstm/train.py --device mps --epochs 10
```

Training prints pooled test metrics each epoch, then **per-clip test metrics** from `best.pt` when finished. Re-run evaluation only:

```bash
python models/lstm/train.py --eval-only --device mps
```

---

## Modules

| File | Purpose |
|------|---------|
| [`encoders.py`](encoders.py) | Frame encoders; default `efficientnet_v2_m` @ 480×480 |
| [`extract_features.py`](extract_features.py) | Cache CNN embeddings; `ensure_features_for_clips()` for on-demand extraction |
| [`dataset.py`](dataset.py) | 30-frame feature windows + center-frame labels |
| [`model.py`](model.py) | BiLSTM → center-frame logit |
| [`train.py`](train.py) | Train, checkpoint, per-clip test inference |

---

## Training

- **Split:** `data/train_clips.csv` and `data/test_clips.csv` (required; create with `python data/train_test_split.py`)
- **Loss:** Tversky (`1 - TI`) with FP/FN weights aligned to **F_beta** (default **β=2**: `tversky_alpha=1/5`, `tversky_beta=4/5`); thresholded **F2** replaces F1; cost metric still uses inverse-frequency **`pos_weight = n_neg / n_pos`** on FN
- **Checkpoint:** `best.pt` by test metric (default `--checkpoint-metric loss`; also `recall`, `cost`, `f_beta`)
- **Outputs:** `checkpoints/best.pt`, `last.pt`, `train_config.json`, `test_clip_metrics.json`

After the last epoch, `train.py` reloads `best.pt` and prints per-clip recall, precision, F_beta, cost, and confusion counts, plus a pooled summary over all test frames.

---

## Dependencies

```bash
pip install torch torchvision tqdm pandas scikit-learn opencv-python-headless
```
