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
- **Class-weighted loss (Phase 07) was tried and measured NOT to fix the
  spider-class failure** — val-set spider precision/recall/F1 were still
  0.0 with balanced weighting. Val macro F1 (0.7153) was also not a clear
  improvement over the unweighted models (0.6926–0.7188, itself a noisy
  range). See `docs/EXPERIMENTS.md`, `docs/RESULTS.md`.

## Reproducibility

- **Fixed seed does not guarantee identical results.** Frozen-backbone
  runs vary by up to 0.057 macro F1 across 6 identical runs; full
  fine-tuning varies more (val macro F1 0.6926 vs. 0.7188 across two
  full sessions), and this propagates to the final test evaluation
  (93.11%/71.56% vs. 93.63%/73.56% test accuracy/macro F1). See
  `docs/REPRODUCIBILITY.md`. Two independent, honestly-kept final test
  results exist for this reason — neither is "the" ground truth.
- **Exact Python/PyTorch/CUDA versions used for training were not
  pinned or recorded** beyond "whatever Colab provided on 2026-09-17."

## Explainability / calibration

- **Grad-CAM (4 spider + 4 swiss-cheese images) shows the model attends
  to a real, localized feature for spider — but names it "impact
  ejecta."** Not a spurious-attention problem; more likely genuine
  visual similarity between the two classes at this crop scale. See
  `docs/ERROR_ANALYSIS.md`. Only 8 images inspected — not a systematic
  study.
- **Calibration (measured on the class-weighted model only): ECE 0.1131
  before, 0.0644 after temperature scaling.** The model is measurably
  overconfident; raw softmax confidence (as returned by the API) should
  not be treated as a calibrated probability without this correction.
  Not cross-checked against the unweighted Phase 08 model.

## Scope

- **Classification only** — no detection, localization, or segmentation
  capability (the dataset doesn't support it, see
  `docs/DATASET_SELECTION.md`).
- **Trained and evaluated on crops from 180 source HiRISE strips only**
  — generalization to HiRISE imagery from other observations is
  untested.
