# Explainability & Calibration — MarsLandmark-AI

Status: **Code implemented and unit-tested** (`src/explainability/gradcam.py`,
`src/training/calibration.py`); **not yet run against the real trained
checkpoint**. Nothing in this document is a claim about the actual
model's behavior — that requires an actual run, not yet done.

## Grad-CAM (`src/explainability/gradcam.py`)

Standard Grad-CAM (Selvaraju et al. 2017). Exposed in the API's
`/explain` endpoint (`src/api/main.py`) using `model.layer4[-1]` as the
target layer for ResNet18. Verified end-to-end in a browser against a
freshly-initialized (untrained) checkpoint — the plumbing (hook
registration, heatmap generation, PNG encoding, frontend display) works;
what an actual heatmap looks like for the real fine-tuned model has not
been examined.

**Open question this would answer, given the measured Phase 06/08
results** (`docs/ERROR_ANALYSIS.md`): when the model fails on "spider"
(predicting "other" 6/7 times) or misses "swiss cheese" (42% of the
time), is it looking at plausible-but-wrong terrain features, or
something spurious (image borders, compression artifacts, etc.)?
Answering this requires running Grad-CAM against real spider/swiss-cheese
test images with the actual `exp_resnet18_finetuned` checkpoint — either
via `/explain` once a checkpoint is deployed, or directly in
`notebooks/colab_train.ipynb` (see that notebook's explainability
section) while the trained model is still in memory.

## Calibration (`src/training/calibration.py`)

`expected_calibration_error()` and `TemperatureScaler` are implemented
and tested against synthetic logits with known calibration properties.
**Not yet run against the real model's validation-set logits** — doing
so requires collecting per-example softmax outputs from the trained
`exp_resnet18_finetuned` checkpoint on the val set, which the current
notebook does not save (it only logs aggregate metrics). This is a
concrete, scoped follow-up: extend the notebook's evaluation cell to
collect and save `(logits, labels)` for the val set, then call
`compute_ece()` / `TemperatureScaler.fit()` on the result.

## Robustness (project rule §22)

Not started. Would test the trained model's sensitivity to realistic
input variation (resolution, brightness, cropping) — not done because it
requires the real checkpoint and a defined test protocol, neither of
which exist yet.

## What this document does NOT claim

- Nothing about *why* the model fails on spider/swiss-cheese beyond what
  `docs/ERROR_ANALYSIS.md` already establishes from the confusion matrix
  alone (a fact, not an explainability finding).
- No calibration number (ECE, reliability diagram) for the real model —
  `NOT YET MEASURED`.
