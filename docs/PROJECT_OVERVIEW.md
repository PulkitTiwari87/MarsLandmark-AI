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
| 03 — Data validation & EDA | In progress — integrity checks done (`reports/data_validation.json`); full EDA/leakage write-up pending |
| 04–13 | Not started |

## Primary task

Multi-class image classification of HiRISE-derived Martian landmark crops
into 8 classes: crater, bright dune, dark dune, slope streak, impact
ejecta, swiss cheese, spider, other. See `docs/DATASET_SELECTION.md` for
the comparison against alternative NASA datasets and `docs/DATASET.md` for
full provenance.

## Former blocker (resolved)

Phase 00/01 were originally scaffolded inside a cloud sandbox whose network
egress could not reach `zenodo.org`/`data.nasa.gov`. The user supplied the
real archive directly into a local session instead; it has since been
extracted, checksummed, and measured (see `docs/DATASET.md`). No training
metric in this repository is real until training actually happens — see
the performance target dashboard in `README.md`, which reads `NOT YET
MEASURED` until then.

## Engineering principles (non-negotiable)

See the project's master instructions. In short: no fabricated data,
metrics, or NASA attribution; test-set integrity; leakage prevention;
every experiment reproducible and logged; failures documented, not hidden.
