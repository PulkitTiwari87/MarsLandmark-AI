# Project Overview — MarsLandmark-AI

## What this is

A reproducible computer-vision project to classify Martian geological
landmarks from real NASA/JPL imagery. It is being built phase by phase,
with every dataset claim and every metric traceable to either a cited
primary source or an experiment actually run.

## Current status

| Phase | Status |
|---|---|
| 00 — Repository & project initialization | In progress |
| 01 — NASA dataset discovery & selection | Done (see caveats in `docs/DATASET.md`) |
| 02 — Data acquisition | Blocked — see below |
| 03–13 | Not started |

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
evaluation cannot happen from inside this sandbox. The acquisition
pipeline, configs, and model code are being built so that a developer (or
a session with unrestricted network access) can run them end-to-end. No
metric in this repository is real until that has happened — see the
performance target dashboard in `README.md`, which will read `NOT YET
MEASURED` until then.

## Engineering principles (non-negotiable)

See the project's master instructions. In short: no fabricated data,
metrics, or NASA attribution; test-set integrity; leakage prevention;
every experiment reproducible and logged; failures documented, not hidden.
