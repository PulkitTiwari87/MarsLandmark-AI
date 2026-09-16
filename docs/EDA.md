# Exploratory Data Analysis — Findings

Status: measured directly from `data/raw/` via `src/data/eda.py` and
`src/data/validate.py`. Raw numbers: `reports/eda_summary.json`,
`reports/data_validation.json`. Figures: `reports/figures/`.

## FACT: severe class imbalance

Class 0 ("other") is 83.6% of all 73,031 images; the rarest class
(impact ejecta) is 0.3%. See `reports/figures/class_distribution.png` and
`docs/DATASET.md`'s measured class-distribution table.

**INTERPRETATION:** plain accuracy will be a misleading metric for this
dataset — a model predicting "other" for every image would score ~83.6%
accuracy while learning nothing about the 7 landmark classes that are the
actual point of the project.

**HYPOTHESIS (not yet tested):** class-weighted loss or a weighted
sampler will be needed to get usable minority-class recall; this must be
tested empirically in Phase 07, not assumed.

## FACT: source strips vary enormously in crop count

173 unique source RED strips were identified; the smallest contributes 7
crops (incl. augmentation) and the largest contributes 1,953 — a
279x spread. See `reports/figures/crops_per_source_strip.png` and the
`crops_per_strip_median` (350.0) in `reports/eda_summary.json`.

**INTERPRETATION:** a small number of source strips dominate the dataset
by volume. Any grouped split (`docs/DATA_SPLIT.md`) must account for this
— naive random group assignment could put a disproportionate share of
total images in one split purely by chance if a few large groups land
together. This is a concrete design constraint for Phase 04, not yet
solved here.

## FACT: measured brightness differs by class in the expected direction

From a fixed-seed random sample of 6,000 images (`reports/eda_summary.json`,
`brightness_mean_by_class`):

| Class | Mean pixel intensity (0-255) | n sampled |
|---|---:|---:|
| dark dune | 64.3 | 97 |
| spider | 100.9 | 37 |
| crater | 120.1 | 383 |
| other | 120.4 | 5,053 |
| swiss cheese | 122.6 | 115 |
| impact ejecta | 125.8 | 15 |
| slope streak | 151.7 | 172 |
| bright dune | 156.7 | 128 |

**INTERPRETATION:** "dark dune" measuring darkest and "bright dune"
measuring brightest is consistent with their names — a sanity check that
the labels correspond to real, physically distinct visual content rather
than being scrambled or arbitrary. This does not by itself indicate model
performance; it is a labeling-sanity signal, not a classification
benchmark.

## FACT: images are uniform and clean

All 73,031 images are exactly 227×227 px, single-channel grayscale
(PIL mode `L`), with 0 corrupted/unreadable files and 0 exact
byte-identical duplicates (`reports/data_validation.json`).

**INTERPRETATION:** no dedicated cleaning step (corrupt-file removal,
resizing, dedup) is needed before Phase 04. The grayscale-only fact does
matter for architecture choice — a pretrained ImageNet backbone expects
3-channel RGB and will need explicit channel replication or a modified
first conv layer (Phase 06 decision, not yet made).

## What this EDA does NOT establish

- Whether class-weighted loss, resampling, or another imbalance strategy
  actually improves minority-class performance — `NOT YET MEASURED`,
  Phase 07.
- Whether brightness alone would let a trivial classifier beat the
  learned model — not tested; would require an actual baseline run
  (Phase 05).
- Full resolution of the 173-vs-180 source-strip discrepancy noted in
  `docs/DATASET.md`.
