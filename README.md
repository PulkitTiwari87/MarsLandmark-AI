# MarsLandmark-AI

A reproducible, scientifically honest computer-vision project for
classifying Martian geological landmarks from real NASA/JPL imagery.

**Status: Phases 00–09, 11–12 done** (training, final evaluation,
class-imbalance experiment, explainability/calibration, API, frontend).
No metric below is fabricated or estimated — anything not yet measured is
labeled `NOT YET MEASURED`/`NOT MEASURED`.

## Overview

MarsLandmark-AI classifies HiRISE (Mars Reconnaissance Orbiter) landmark
image crops into geological classes. See `docs/PROJECT_OVERVIEW.md` for
current phase status.

## Motivation

Planetary science teams and rover-operations groups increasingly rely on
automated content classification to triage the volume of orbital and
surface imagery NASA collects. This project builds and honestly evaluates
such a classifier on real NASA data, documenting the full path from raw
imagery to a defensible benchmark.

## NASA dataset

Primary dataset: **Mars orbital image (HiRISE) labeled data set, version 3**
(DeepMars, Wagstaff et al., NASA/JPL, HiRISE camera on Mars
Reconnaissance Orbiter; DOI `10.5281/zenodo.2538136`). Acquired, extracted,
checksummed, and measured directly — 73,031 images, 8 classes, 227×227
single-channel, 0 corrupted, 0 exact duplicates, no official split shipped
in the archive. Full provenance and every measured figure is in
`docs/DATASET.md`; the leakage-prevention split strategy is in
`docs/DATA_SPLIT.md`; dataset comparison/selection rationale is in
`docs/DATASET_SELECTION.md`.

## Problem definition

Primary task: multi-class image classification, 8 classes (crater, bright
dune, dark dune, slope streak, impact ejecta, swiss cheese, spider,
other). See `docs/DATASET_SELECTION.md` §"Decision" for why this task and
dataset were chosen over alternatives (AI4Mars terrain segmentation,
DeepMars MSL surface classification).

## Architecture

SimpleCNN (from-scratch baseline) and ResNet18 (transfer learning, both
frozen-backbone and fully fine-tuned variants) — see
`docs/MODEL_ARCHITECTURE.md` for design rationale.

## Data pipeline

Implemented — grouped leakage-safe split, grayscale-to-3-channel Dataset
with ImageNet normalization. See `docs/DATA_PIPELINE.md`.

## Models

Trained and evaluated — see `docs/EXPERIMENTS.md` and `docs/RESULTS.md`.
Best model: ResNet18, fully fine-tuned (test accuracy 93.11%, macro F1
71.56%).

## Experimental methodology

Train/validation/test integrity rules, leakage-prevention policy, and the
85–95% target-performance policy (a target, not a manipulation rule) are
described in the project's master instructions and will be elaborated in
`docs/DATA_SPLIT.md` and `docs/TRAINING.md`.

## Benchmark results

| Experiment | Model | Image Size | Epochs | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Train Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp_baseline_simplecnn | SimpleCNN (scratch) | 227 | 10 | 0.7766 | 0.2978 | NOT MEASURED | NOT MEASURED | 22.0 min |
| exp_resnet18_frozen (6 runs, 2 sessions) | ResNet18 (frozen) | 227 | 5 | 0.845–0.855 | 0.520–0.576 | NOT MEASURED | NOT MEASURED | ~10.6 min |
| exp_resnet18_finetuned (session 1) | ResNet18 (fine-tuned) | 227 | 15 | 0.9013 | 0.7188 | **0.9311** | **0.7156** | 52.9 min |
| exp_resnet18_finetuned (session 2) | ResNet18 (fine-tuned) | 227 | 6 | 0.9013 | 0.6926 | 0.9363 | 0.7356 | 19.2 min |
| exp_resnet18_finetuned_classweighted | ResNet18 (fine-tuned, weighted loss) | 227 | 15 | 0.8442 | 0.7153 | NOT MEASURED | NOT MEASURED | 48.1 min |

Full detail: `docs/EXPERIMENTS.md`, `docs/RESULTS.md`, `experiments/experiments.csv`.
Two independent full sessions produced two different final test results
(93.11%/71.56% and 93.63%/73.56%) from separately-trained models — real,
measured non-determinism, not an error; see `docs/REPRODUCIBILITY.md`.
**Class-weighted loss (Phase 07) did not fix the spider-class failure**
(val spider F1 still 0.0) and was not a clear win on macro F1 either —
see `docs/RESULTS.md`'s "did it help?" section.

## Error analysis

See `docs/ERROR_ANALYSIS.md`. Complete failure on "spider" (0/7 test,
0/7 val even after class-weighted retraining); "swiss cheese" has high
precision but weak recall. Grad-CAM (Phase 09) shows the model attends
to a real, localized feature for spider images — it just consistently
mislabels it as "impact ejecta," suggesting genuine visual similarity
between the two classes rather than a training failure.

## Explainability

Grad-CAM and calibration run against the real trained checkpoint (Phase
09, done). ECE 0.1131 → 0.0644 after temperature scaling — the model is
measurably overconfident. See `docs/EXPLAINABILITY.md`.

## Results

See `docs/RESULTS.md` for full detail.

### Performance target dashboard

```text
Minimum desired genuine performance: 85%
Upper target boundary:               95%
Actual measured performance:         93.11% (test accuracy; macro F1 71.56%)
```

The 85–95% range is an engineering target, not a rule for manipulating
results. The measured 93.11% falls inside it — reported as measured, not
engineered to land there (see `docs/RESULTS.md` for what was and wasn't
adjusted). Macro F1 (71.56%) is substantially lower, reflecting weak
minority-class performance — see `docs/ERROR_ANALYSIS.md`.

## Limitations

See `docs/LIMITATIONS.md` for the full consolidated list. Headline items:
complete failure on the "spider" class (`docs/ERROR_ANALYSIS.md`),
run-to-run non-determinism despite a fixed seed
(`docs/REPRODUCIBILITY.md`), and no Grad-CAM/calibration run against the
real checkpoint yet (`docs/EXPLAINABILITY.md`).

## Reproducibility

See `docs/REPRODUCIBILITY.md`. Configuration defaults live in
`configs/config.yaml`; the experiment registry is
`experiments/experiments.csv`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Runnable via `notebooks/colab_train.ipynb` (used for the results above)
or `python -m src.training.train --config configs/config.yaml --model
resnet18 --experiment-id <id>`.

## Evaluation

`src/training/metrics.py`'s `compute_metrics()` is used by both paths.
Final test evaluation is a separate, manually-triggered step (see
`docs/TRAINING.md`'s test-set discipline section) — already run once,
see `docs/RESULTS.md`.

## Inference / API

`src/api/main.py` (FastAPI): `GET /health`, `POST /predict`,
`POST /explain` (Grad-CAM heatmap). Run with:

```bash
MODEL_CHECKPOINT_PATH=checkpoints/exp_resnet18_finetuned.pt \
  uvicorn src.api.main:app --reload
```

`frontend/index.html` is a single static file (no build step) that talks
to this API — drag/drop an image, see the prediction, top-5 confidences,
and a Grad-CAM overlay. Open it directly or serve it
(`python -m http.server 8080 --directory frontend`); see
`docs/MODEL_CARD.md` before trusting any prediction it shows.

## Citation / data attribution

Dataset: Gary Doran, Steven Lu, Lukas Mandrake, Kiri Wagstaff. "Mars
orbital image (HiRISE) labeled data set version 3" (Version 3.0.0)
[Dataset]. Zenodo, 2019. https://doi.org/10.5281/zenodo.2538136. Licensed
under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
Related paper: Wagstaff, K.L., Lu, Y., Stanboli, A., Grimes, K., Gowda,
T., Padams, J. "Deep Mars: CNN Classification of Mars Imagery for the PDS
Imaging Atlas." IAAI 2018. Full provenance and verification method in
`docs/DATASET.md`.
