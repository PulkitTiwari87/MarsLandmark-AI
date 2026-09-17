# MarsLandmark-AI

A reproducible, scientifically honest computer-vision project for
classifying Martian geological landmarks from real NASA/JPL imagery.

**Status: baseline + transfer learning trained and evaluated (Phase 06/08
done).** No metric below is fabricated or estimated — anything not yet
measured is labeled `NOT YET MEASURED`/`NOT MEASURED`.

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
| exp_resnet18_frozen (4 runs) | ResNet18 (frozen) | 227 | 5 | 0.845–0.855 | 0.520–0.576 | NOT MEASURED | NOT MEASURED | ~10.6 min |
| exp_resnet18_finetuned | ResNet18 (fine-tuned) | 227 | 15 | 0.9013 | 0.7188 | **0.9311** | **0.7156** | 52.9 min |

Full detail: `docs/EXPERIMENTS.md`, `docs/RESULTS.md`, `experiments/experiments.csv`.
Frozen-backbone macro F1 varied 0.520–0.576 across 4 identical runs with
the same seed — a measured non-determinism, see `docs/REPRODUCIBILITY.md`.

## Error analysis

Done for the final-test confusion matrix — see `docs/ERROR_ANALYSIS.md`.
Key finding: complete failure on the "spider" class (0/7 correct, mostly
misclassified as "other"); "swiss cheese" has high precision (0.997) but
poor recall (0.576). No image-level failure gallery yet (Phase 09).

## Explainability

Not yet performed — see `docs/EXPLAINABILITY.md` once written (Phase 09).

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

See `docs/LIMITATIONS.md` once written (not yet — Phase 13). Known
limitations measured so far: complete failure on the "spider" class
(`docs/ERROR_ANALYSIS.md`), run-to-run non-determinism despite a fixed
seed (`docs/REPRODUCIBILITY.md`), and no image-level explainability yet
(Phase 09).

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

Not yet implemented — see Phase 11.

## Citation / data attribution

Dataset: Gary Doran, Steven Lu, Lukas Mandrake, Kiri Wagstaff. "Mars
orbital image (HiRISE) labeled data set version 3" (Version 3.0.0)
[Dataset]. Zenodo, 2019. https://doi.org/10.5281/zenodo.2538136. Licensed
under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
Related paper: Wagstaff, K.L., Lu, Y., Stanboli, A., Grimes, K., Gowda,
T., Padams, J. "Deep Mars: CNN Classification of Mars Imagery for the PDS
Imaging Atlas." IAAI 2018. Full provenance and verification method in
`docs/DATASET.md`.
