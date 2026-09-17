# Error Analysis — MarsLandmark-AI

Status: derived directly from `reports/final_test_evaluation.json`'s
confusion matrix (the one-shot final test run, ResNet18 fine-tuned,
2026-09-17). No cherry-picking — every misclassification category below
is read off the full 8x8 matrix, not a hand-picked sample of image files
(no image-level gallery yet; the notebook doesn't currently save
misclassified filenames — see "What's missing" below).

## FACT: "spider" is a complete failure (0/7 correct)

Confusion matrix row for true label "spider" (n=7):

| Predicted as | other | crater | dark dune | slope streak | bright dune | impact ejecta | swiss cheese | spider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

**INTERPRETATION:** 6 of 7 spider images were classified as "other" (the
dominant class), 1 as "swiss cheese", 0 correctly. This is consistent
with the model defaulting to the majority class when it has no confident
signal — exactly the failure mode class-imbalance handling (Phase 07) is
meant to address.

**HYPOTHESIS (not yet tested):** with only 7 test examples and similarly
few training/validation examples (see `docs/DATA_SPLIT.md`'s per-split
class counts), the model may never have seen enough "spider" examples
during training to form a usable representation — this is a data-volume
hypothesis, not yet distinguished from a visual-similarity hypothesis
(spiders vs. dark terrain features) without further investigation
(Grad-CAM in Phase 09, or simply checking the "spider" training-set size,
which `docs/DATA_SPLIT.md` already lists).

## FACT: "swiss cheese" has high precision but poor recall (57.6%)

Confusion matrix row for true label "swiss cheese" (n=623):

| Predicted as | other | crater | dark dune | slope streak | bright dune | impact ejecta | swiss cheese | spider |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 150 | 54 | 60 | 0 | 0 | 0 | 359 | 0 |

**INTERPRETATION:** when the model does predict "swiss cheese," it's
almost always right (precision 0.997 — 359/360 predictions correct,
`docs/RESULTS.md`). But it misses 264 of 623 actual swiss-cheese images,
mostly confusing them with "other" (150), "dark dune" (60), and "crater"
(54). This is a recall problem, not a precision problem — the model is
conservative about calling "swiss cheese," not confused about what it
looks like when it does.

## FACT: "dark dune" has weak precision (48.8%) despite decent recall (70.5%)

The model over-predicts "dark dune": it correctly identifies most true
dark-dune images (recall 0.705) but also mislabels other terrain as dark
dune often enough that fewer than half of its dark-dune predictions are
correct. Cross-referencing the "swiss cheese" row above (60 swiss-cheese
images predicted as dark dune) shows at least part of the source of this
confusion — swiss cheese and dark dune are seemingly visually similar to
the model in some cases.

## What this does NOT establish

- **No image-level gallery of failure cases exists yet.** The notebook
  logs the confusion matrix but not individual misclassified filenames —
  project rule §19 calls for "a gallery of representative failure cases,"
  which requires re-running evaluation with per-image predictions saved.
  Not yet done; flagged rather than skipped silently.
- **No Grad-CAM or other explainability check has been run** to see
  *where* the model is looking when it fails on spider/swiss-cheese
  images (Phase 09, not started).
- Whether these failure patterns are stable across different random
  seeds/splits, or specific to this one run, is untested.
