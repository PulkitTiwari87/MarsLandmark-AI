# Project Overview — MarsLandmark-AI

## What this is

A reproducible computer-vision project to classify Martian geological
landmarks from real NASA/JPL imagery. It is being built phase by phase,
with every dataset claim and every metric traceable to either a cited
primary source or an experiment actually run.

## Current status

| Phase | Status |
|---|---|
| 00 — Repository & project initialization | Done |
| 01 — NASA dataset discovery & selection | Done (see `docs/DATASET.md`) |
| 02 — Data acquisition | Done — archive acquired, checksummed, extracted, provenance confirmed (`docs/DATASET.md`) |
| 03 — Data validation & EDA | Done — `reports/data_validation.json`, `reports/eda_summary.json`, `docs/EDA.md`, `docs/DATA_SPLIT.md` |
| 04 — Data pipeline (grouped split, Dataset/DataLoader) | Done — `src/data/split.py`, `src/data/dataset.py`, `docs/DATA_PIPELINE.md` |
| 05 — Baseline | Done — trained in Colab 2026-09-17: SimpleCNN val macro F1 0.2978 (`docs/EXPERIMENTS.md`) |
| 06 — Deep learning / transfer learning | Done — ResNet18 frozen (val macro F1 0.52–0.58, 4 runs) and fine-tuned (0.7188, selected) (`docs/EXPERIMENTS.md`) |
| 07 — Optimization (imbalance, overfitting/underfitting investigation) | Done — class-weighted loss run and measured NOT to fix the spider-class failure (val spider F1 still 0.0); val macro F1 (0.7153) inconclusive vs. unweighted baseline given measured noise. See `docs/EXPERIMENTS.md`, `docs/RESULTS.md`. |
| 08 — Final evaluation | Done, **twice** (two independent sessions): 93.11%/71.56% and 93.63%/73.56% test accuracy/macro F1 — both real, both kept, see `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md` |
| 09 — Explainability & robustness | Grad-CAM + calibration run against the real checkpoint — spider attention is real but mislabeled as impact ejecta; ECE 0.1131→0.0644 after temperature scaling. See `docs/EXPLAINABILITY.md`, `docs/ERROR_ANALYSIS.md`. Robustness testing not started. |
| 10 — Detection/localization | Not applicable — dataset only supports classification, see `docs/DATASET_SELECTION.md` |
| 11 — API | Done — `src/api/main.py`, tested, verified live in a browser |
| 12 — Frontend/dashboard | Done — `frontend/index.html`, verified live end-to-end (predict + Grad-CAM) in a browser |
| 13 — Final documentation | Done — `docs/MODEL_CARD.md`, `docs/LIMITATIONS.md`, `docs/EXPLAINABILITY.md` added; README fully updated |

## Primary task

Multi-class image classification of HiRISE-derived Martian landmark crops
into 8 classes: crater, bright dune, dark dune, slope streak, impact
ejecta, swiss cheese, spider, other. See `docs/DATASET_SELECTION.md` for
the comparison against alternative NASA datasets and `docs/DATASET.md` for
full provenance.

## Former blockers (resolved)

Phase 00/01 were originally scaffolded inside a cloud sandbox whose network
egress could not reach `zenodo.org`/`data.nasa.gov`. The user supplied the
real archive directly into a local session instead; it has since been
extracted, checksummed, and measured (see `docs/DATASET.md`).

Training itself was declined on the local machine's GPU (project owner's
explicit instruction); the project owner instead ran
`notebooks/colab_train.ipynb` in Google Colab (T4 GPU) on 2026-09-17 and
supplied the results, which are now real, measured, and recorded in
`docs/EXPERIMENTS.md` and `docs/RESULTS.md` — see the performance target
dashboard in `README.md`.

## Engineering principles (non-negotiable)

See the project's master instructions. In short: no fabricated data,
metrics, or NASA attribution; test-set integrity; leakage prevention;
every experiment reproducible and logged; failures documented, not hidden.
