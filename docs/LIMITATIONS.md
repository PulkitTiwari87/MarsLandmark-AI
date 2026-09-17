# Limitations — MarsLandmark-AI

Consolidated from `docs/DATASET.md`, `docs/DATA_SPLIT.md`,
`docs/ERROR_ANALYSIS.md`, `docs/REPRODUCIBILITY.md`,
`docs/EXPLAINABILITY.md`, `docs/MODEL_CARD.md` — each linked entry has
the full detail; this page is the single place to check before relying
on this project's model or numbers.

## Data

- **No official train/val/test split** ships with this dataset — this
  project's split (`docs/DATA_SPLIT.md`) is its own construction, not an
  externally validated one.
- **License unresolved beyond the primary Zenodo record's own field.**
  The record states CC-BY 4.0; no independent legal review has been
  done.
- **173 vs. 180 source-strip discrepancy** (`docs/DATASET.md`) —
  unresolved, does not affect split correctness but is an open question
  about the dataset's own internal grouping.

## Model

- **Complete failure on the "spider" class** (0% recall/F1 on 7 test
  examples) — see `docs/ERROR_ANALYSIS.md`. Do not use this model for
  spider-class identification.
- **"swiss cheese" recall is 57.6%**, **"dark dune" precision is 48.8%**
  — both real, measured weaknesses, not edge cases.
- **Macro F1 (71.56%) is far below accuracy (93.11%)** — accuracy alone
  substantially overstates how well this model serves all 8 classes.
- **Class-weighted loss (Phase 07) has not yet been shown to help** —
  the code exists (`src/training/imbalance.py`) but the comparison run
  has not completed; do not assume it fixes the above without checking
  `docs/EXPERIMENTS.md` for a logged result.

## Reproducibility

- **Fixed seed does not guarantee identical results** — measured
  variance of up to 0.057 macro F1 across 4 identical runs, due to
  cuDNN non-determinism not disabled by the training code. See
  `docs/REPRODUCIBILITY.md`.
- **Exact Python/PyTorch/CUDA versions used for training were not
  pinned or recorded** beyond "whatever Colab provided on 2026-09-17."

## Explainability / calibration

- **No Grad-CAM has been run against the real trained checkpoint** —
  whether the model attends to plausible terrain features is unknown.
- **No calibration (ECE, temperature scaling) has been measured** for
  the real model — raw softmax confidence, as returned by the API,
  should not be treated as a calibrated probability.

## Scope

- **Classification only** — no detection, localization, or segmentation
  capability (the dataset doesn't support it, see
  `docs/DATASET_SELECTION.md`).
- **Trained and evaluated on crops from 180 source HiRISE strips only**
  — generalization to HiRISE imagery from other observations is
  untested.
