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

## FACT: Grad-CAM confirms the model attends to a real, localized feature for "spider" — it just names it wrong

Run against the class-weighted model (`exp_resnet18_finetuned_classweighted`)
on real val-set images — see `reports/figures/gradcam_classweighted_run2.png`.

- **4/4 sampled "spider" images: predicted "impact ejecta", 0/4 correct.**
  Every heatmap shows a tight, high-confidence activation centered on a
  single compact blob-like feature in the crop — the model is not
  attending to noise, borders, or compression artifacts. It has learned
  *a* feature, and it fires reliably on it, but consistently maps it to
  the wrong class label. This shifted from the earlier unweighted model's
  behavior (which mostly predicted "other" for spider, per the original
  Phase 08 confusion matrix) — class weighting changed *which* wrong
  answer the model gives, not whether it gets spider right.
- **4/4 sampled "swiss cheese" images: predicted "swiss cheese" correctly.**
  Heatmaps show diffuse but plausible attention across the pockmarked,
  textured terrain area — consistent with genuinely learning the
  swiss-cheese texture pattern, not a shortcut.

**INTERPRETATION:** the spider/impact-ejecta confusion looks like a
genuine visual-similarity problem (both classes may present as compact,
roughly circular features at this crop scale and resolution) compounded
by spider's small sample size (`docs/DATA_SPLIT.md`), rather than a
training or attention-mechanism failure. This reframes the earlier
Phase 08 hypothesis ("not yet distinguished from a visual-similarity
hypothesis") — the visual-similarity explanation now has direct
supporting evidence, though it is still not proven (would need a larger
Grad-CAM sample and/or a human domain expert's judgment on the actual
crops to confirm).

## What this does NOT establish

- **No image-level gallery of failure cases exists yet** beyond the 4+4
  Grad-CAM examples above. Project rule §19 calls for a broader gallery
  of representative failures, which requires re-running evaluation with
  per-image predictions saved at scale. Not yet done.
- Only 4 spider and 4 swiss-cheese examples were inspected with Grad-CAM
  — not the full val or test set. This is a qualitative spot-check, not
  a systematic study.
- Whether these failure patterns are stable across different random
  seeds/splits is untested — though the fact that spider failed
  identically (0/7 test recall) across two independently-trained
  unweighted models plus the class-weighted model is itself evidence the
  failure is not a one-off fluke of a single run.
