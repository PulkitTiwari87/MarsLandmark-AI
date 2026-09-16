# MarsLandmark-AI

A reproducible, scientifically honest computer-vision project for
classifying Martian geological landmarks from real NASA/JPL imagery.

**Status: early development.** No model has been trained. No metric below
is fabricated or estimated — anything not yet measured is labeled
`NOT YET MEASURED`.

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

Primary dataset: **Mars orbital image (HiRISE) labeled data set**
(DeepMars, Wagstaff et al., NASA/JPL, HiRISE camera on Mars
Reconnaissance Orbiter). Full provenance, including fields still pending
verification, is in `docs/DATASET.md`. Dataset comparison and selection
rationale is in `docs/DATASET_SELECTION.md`.

**Known blocker:** this repository is being developed inside a sandboxed
session whose network cannot reach the hosts serving the dataset
(`zenodo.org`, `data.nasa.gov`). Real acquisition and training require an
environment with unrestricted network access. See `docs/DATASET.md` for
details and next steps.

## Problem definition

Primary task: multi-class image classification, 8 classes (crater, bright
dune, dark dune, slope streak, impact ejecta, swiss cheese, spider,
other). See `docs/DATASET_SELECTION.md` §"Decision" for why this task and
dataset were chosen over alternatives (AI4Mars terrain segmentation,
DeepMars MSL surface classification).

## Architecture

`NOT YET MEASURED` / not yet finalized — see `docs/MODEL_ARCHITECTURE.md`
once written (Phase 06).

## Data pipeline

Not yet implemented — see `docs/DATA_PIPELINE.md` once written (Phase 04).

## Models

Not yet trained — see `docs/EXPERIMENTS.md`.

## Experimental methodology

Train/validation/test integrity rules, leakage-prevention policy, and the
85–95% target-performance policy (a target, not a manipulation rule) are
described in the project's master instructions and will be elaborated in
`docs/DATA_SPLIT.md` and `docs/TRAINING.md`.

## Benchmark results

| Experiment | Model | Params | Image Size | Epochs | Accuracy | Macro F1 | Precision | Recall | Train Time | Inference Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _none yet_ | — | — | — | — | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

## Error analysis

Not yet performed — see `docs/ERROR_ANALYSIS.md` once written (Phase 09).

## Explainability

Not yet performed — see `docs/EXPLAINABILITY.md` once written (Phase 09).

## Results

### Performance target dashboard

```text
Minimum desired genuine performance: 85%
Upper target boundary:               95%
Actual measured performance:         NOT YET MEASURED
```

The 85–95% range is an engineering target, not a rule for manipulating
results. If the honestly measured result falls outside this range, it
will be reported as measured — see project master instructions §35.

## Limitations

See `docs/LIMITATIONS.md` once written. Current, known limitation: dataset
provenance fields pending verification (`docs/DATASET.md`); no real data
has been downloaded yet.

## Reproducibility

See `docs/REPRODUCIBILITY.md` once written (Phase 13). Configuration
defaults live in `configs/config.yaml`; the experiment registry is
`experiments/experiments.csv`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Not yet runnable end-to-end (blocked on data acquisition). Intended entry
point: `src/training/train.py --config configs/config.yaml` (to be added
in Phase 05/06).

## Evaluation

Not yet runnable (see above).

## Inference / API

Not yet implemented — see Phase 11.
Implement everything

## Citation / data attribution

Wagstaff, K.L., Lu, Y., Stanboli, A., Grimes, K., Gowda, T., Padams, J.
"Deep Mars: CNN Classification of Mars Imagery for the PDS Imaging
Atlas." Proceedings of the Thirtieth Annual Conference on Innovative
Applications of Artificial Intelligence (IAAI), 2018. Dataset hosted by
NASA/JPL via Zenodo — exact DOI/version pending verification, see
`docs/DATASET.md`.
