# sports-footage-autotrim

CS348K project: turn long, casually recorded volleyball videos into gameplay-only trims with a labeled timeline.

## Summary

We aim to build a system that takes a long, casually recorded video of a volleyball match and produces:

- A **labeled timeline** separating gameplay from downtime
- An **automatically trimmed gameplay-only video** users can review and export

Our approach combines pretrained visual feature extraction with temporal modeling over video segments. We start from simple baselines and progress toward a lightweight predictor that identifies when meaningful play is happening in noisy, real-world hobbyist footage.

**Constraints:** human labeling time, dataset quality, and moderate compute.

## Inputs and outputs

### Input

Long-form hobbyist volleyball videos from a tripod-mounted phone or similar. These often include large amounts of dead time: retrieving balls, chatting, rotating players, switching sides, etc.

### Output

- A timeline labeling segments as **gameplay** or **downtime**
- A trimmed video containing predicted gameplay segments
- Optionally, a simple editor so users can adjust segment boundaries before export

## Core problem

**Binary temporal segmentation:** given a recreational sports video, decide which parts are active play and which are not.

A good solution should:

- Preserve all gameplay
- Remove a large amount of dead time
- Produce cuts close enough to human judgment that either no edits or only small edits are needed

## Roadmap and task list

**Priority:** get an **end-to-end pipeline** working early—even if the first version is simple—so we can always run the system on a video, produce predictions, and evaluate them.

### Planned work

1. Investigate datasets, starter tools, and related work in sports video segmentation.
2. Build a working dataset from recreational volleyball footage (personal recordings and rights-cleared online video).
3. Manually annotate gameplay vs. downtime with an open-source annotation tool.
4. Implement **simple baselines** (e.g., motion/activity heuristics without a neural network).
5. Build a **feature extraction pipeline** (motion cues, pose estimation, or general visual features).
6. Train a **lightweight predictor** on extractor outputs; run it successfully on individual frames.
7. If time permits, extend from single-frame predictors to **temporal modeling**.
8. Build a **simple UI**: timeline, small edits, export of trimmed videos.

### First week goals

- Fixed dataset
- Small labeled subset
- At least one baseline running end-to-end
- Evaluation script in place

### Nice to haves

- Lightweight or faster variant (e.g., mobile-oriented)
- Extra labels beyond gameplay/downtime (e.g., rally or serve detection)

## Team

| Person   | Focus                                                       |
| -------- | ----------------------------------------------------------- |
| Spencer  | Dataset collection, annotation, baselines                   |
| Raina    | Feature extraction and model training                       |
| Kory     | Interface, export pipeline, evaluation tools                |

## Deliverables and evaluation

### Demo

Show a raw volleyball video becoming:

1. A labeled timeline  
2. An automatically trimmed output video  

Ideally, also a simple UI to inspect and edit proposed cuts.

### Technical evaluation

Compare predictions to manually labeled ground truth. Likely metrics:

- Precision, recall, and F1 on the **gameplay** class
- Frame-wise accuracy
- Possibly segment overlap or boundary-tolerance metrics

### Practical usefulness (trimming)

- How much downtime is removed  
- How much gameplay is preserved  
- How short the final video becomes  
- How many manual corrections are still needed  

Compare against:

- Manual cutting  
- Motion-threshold methods  
- Non-temporal classifiers  

Include qualitative **success** and **failure** examples.

### Success criteria

We consider the project successful if:

1. **Meaningful reduction** — Total length is cut by a non-trivial amount vs. raw footage.  
2. **Comparable to manual editing** — Cuts are close to what a person would produce when editing out downtime; trimmed output needs at most minor tweaks.  
3. **Better than baselines** — Outperforms simple heuristics (e.g., basic motion thresholding or non-temporal classifiers) at isolating gameplay.

## Risks and mitigation

| Risk | Mitigation |
| ---- | ---------- |
| Datasets not generalizable; labeling slower than expected | Small, focused dataset early; clear annotation rules |
| Pretrained extractors weak on noisy amateur footage | Start with robust signals (motion, general visual features) before specialized detectors |
| Overfitting on limited data | Lightweight model; strong simple baselines; test on videos from different sources |
| Compute limits | Keep main path on **offline desktop** processing if heavier models are too costly |

## What we need help with

- **References** on sports video temporal segmentation or highlight extraction from casual footage.  
- **Feedback** on whether the evaluation plan is strong enough—especially for *usefulness as an editing tool*, not only as a classifier.  
- **Guidance** on appropriate compute if a local GPU is insufficient.
