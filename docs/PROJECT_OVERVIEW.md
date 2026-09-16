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
| 01 — NASA dataset discovery & selection | Done (see caveats in `docs/DATASET.md`) |
| 02 — Data acquisition | Blocked on real download — see below |
| 04 — Data pipeline (code) | Written & unit-tested against synthetic fixtures; not run on real data |
| 05/06 — Baseline & model code | Majority baseline, from-scratch CNN, ResNet-50 transfer learning implemented; none trained on real data |
| 03, 07–13 | Not started — require real data |

## Primary task

Multi-class image classification of HiRISE-derived Martian landmark crops
into 8 classes: crater, bright dune, dark dune, slope streak, impact
ejecta, swiss cheese, spider, other. See `docs/DATASET_SELECTION.md` for
the comparison against alternative NASA datasets and `docs/DATASET.md` for
full provenance.

## Known blocker

This project is being developed inside a sandboxed session whose network
egress proxy blocks the hosts that serve the actual dataset
(`zenodo.org`, `data.nasa.gov`). Real data acquisition, training, and
evaluation cannot happen from inside this sandbox. The only other
environment available in this account has the same restriction, so the
plan is for the dataset to be downloaded externally and uploaded into the
session. The acquisition pipeline, data pipeline, and model code
(`src/data/`, `src/models/`, `src/training/`, `src/evaluation/`) have been
built and unit-tested against synthetic fixtures so they are ready to run
against the real files as soon as they arrive. No metric in this
repository is real until that has happened — see the performance target
dashboard in `README.md`, which will read `NOT YET MEASURED` until then.

## Engineering principles (non-negotiable)

See the project's master instructions. In short: no fabricated data,
metrics, or NASA attribution; test-set integrity; leakage prevention;
every experiment reproducible and logged; failures documented, not hidden.
