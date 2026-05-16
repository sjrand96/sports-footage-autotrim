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

- **Split:** 70% train / 30% test by `clip_id` (`random_seed=42`)
- **Loss:** weighted BCE with fixed **`pos_weight = POS_WEIGHT_POSITIVE`** (default **2**) on playing; adjust in `train.py`
- **Checkpoint:** `best.pt` by highest test recall, then lowest cost
- **Outputs:** `checkpoints/best.pt`, `last.pt`, `train_config.json`, `test_clip_metrics.json`

After the last epoch, `train.py` reloads `best.pt` and prints per-clip recall, precision, F1, cost, and confusion counts, plus a pooled summary over all test frames.

---

## Dependencies

```bash
pip install torch torchvision tqdm pandas scikit-learn opencv-python-headless
```
