# Data Split Methodology — MarsLandmark-AI

Status: **Decision documented, not yet implemented.** This document
records the leakage-prevention analysis and the chosen splitting strategy.
The actual split-generation code belongs to Phase 04 (Data Pipeline) and
has not been written yet — see `docs/PROJECT_OVERVIEW.md` for phase
status.

## Why a custom split is required

`docs/DATASET.md` confirms this archive (`hirise-map-proj-v3.zip`) ships
**no official train/validation/test split** — only one flat label file
covering all 73,031 images. Any split used by this project must therefore
be constructed here, and the methodology must be documented before any
model is trained, per project rule §0.7 (leakage prevention) and §0.6
(test-set integrity).

## Leakage risk in this dataset

Each of the 10,433 original landmark crops was augmented into 6 additional
images (90°/180°/270° rotation, horizontal flip, vertical flip, brightness
adjustment) — see `docs/DATASET.md`. An augmented image is visually and
semantically almost identical to its original. **A naive random split over
all 73,031 images would very likely place an original in one split and its
own rotated/flipped/brightened sibling in another split** (e.g. original in
train, its horizontal flip in test). That would leak information: the
model could effectively "see" a near-duplicate of a test example during
training, inflating measured test accuracy without reflecting real
generalization.

Filenames encode which crops share a source HiRISE image strip (e.g.
`ESP_011623_2100_RED-0069.jpg` and `ESP_011623_2100_RED-0069-r90.jpg` are
the same underlying landmark). More broadly, crops from the *same* source
strip may depict adjacent or repeated terrain features, so even distinct
crop IDs from one strip are not fully independent.

## Chosen strategy: group split by source HiRISE RED-strip ID

Split at the **source-strip level**, not the image level: every image
(original and every augmented variant, and every distinct crop) whose
filename shares a source RED-strip ID is assigned to exactly one of
train/validation/test. This guarantees:

- No augmented sibling of a training image appears in validation or test.
- No two crops from the same source strip are split across sets.

This was measured as feasible directly from the data: regex-parsing all
73,031 filenames against `{MISSION}_{ORBIT}_{ANGLE}_RED-{crop_id}[-{aug}].jpg`
succeeded for 100% of files with 0 parse failures, yielding 173 group keys
(see `docs/DATASET.md` for the 173-vs-180 discrepancy note, which does not
affect the validity of this grouping).

## What is NOT yet decided

- **Split ratios.** No train/val/test percentages have been chosen or
  computed yet. `configs/config.yaml`'s `val_fraction`/`test_fraction` are
  intentionally `null` until Phase 04.
- **Stratification.** Class distribution is severely imbalanced (see
  `docs/DATASET.md`); whether/how to stratify the *group*-level split by
  each group's class composition (since groups vary from 7 to 1,953 crops)
  needs a concrete algorithm, not just "stratified group split" as a
  phrase. This will be designed and implemented in Phase 04, with the
  resulting per-split class distributions reported (not assumed) once
  computed.
- **Reproducibility.** The split will be generated with the project's fixed
  seed (`configs/config.yaml: seed: 42`) and the resulting file/group
  assignment will be persisted (e.g. a split manifest file) so it is
  identical across reruns — mechanism TBD in Phase 04.

## Rule this document exists to satisfy

Project rule §0.7: *"If grouping by ... source image ... is appropriate,
use grouped splitting. Document the decision in `docs/DATA_SPLIT.md`."*
This document is that record. The split itself — the actual assignment of
each group to train/val/test — is Phase 04 work and is not implemented in
this repository yet.
