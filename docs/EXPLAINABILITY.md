# Explainability & Calibration — MarsLandmark-AI

Status: **Run against the real trained checkpoint (Phase 09, 2026-09-17).**
Grad-CAM: `reports/figures/gradcam_classweighted_run2.png`. Calibration:
`reports/calibration_report.json`. Findings below are measured, not
projected.

## Grad-CAM (`src/explainability/gradcam.py`)

Standard Grad-CAM (Selvaraju et al. 2017), run against
`exp_resnet18_finetuned_classweighted` on real val-set images (4 spider,
4 swiss-cheese). Full write-up and interpretation:
`docs/ERROR_ANALYSIS.md`. Summary: the model attends to a real, tightly
localized feature for spider images (not noise/artifacts) but
consistently mislabels it as "impact ejecta" — 0/4 correct. For swiss
cheese, attention is plausibly on the actual pockmarked terrain texture,
and predictions were 4/4 correct.

Also exposed in the API's `/explain` endpoint (`src/api/main.py`) using
`model.layer4[-1]` as the target layer — verified end-to-end in a browser
against a freshly-initialized checkpoint (plumbing only; the real-model
qualitative results above came from the notebook, not the API, since no
trained checkpoint has been deployed to a running API instance yet).

## Calibration (`src/training/calibration.py`)

Run against `exp_resnet18_finetuned_classweighted`'s val-set logits
(n=10,955):

| Metric | Value |
|---|---:|
| ECE before calibration | 0.1131 |
| Learned temperature | 1.7422 |
| ECE after calibration | 0.0644 |

**FACT:** temperature scaling roughly halved the calibration error
(0.1131 → 0.0644). **INTERPRETATION:** the model is measurably
overconfident (temperature > 1 means logits needed softening) — raw
softmax confidence from this checkpoint should not be read as a
calibrated probability without applying this scaling. This was measured
on the class-weighted model, not the unweighted one selected for Phase 08
— the two have not been compared for calibration.

## Robustness (project rule §22)

Not started. Would test the trained model's sensitivity to realistic
input variation (resolution, brightness, cropping) — not done because it
requires a defined test protocol that doesn't exist yet.

## What this document does NOT claim

- **Not a systematic study.** Grad-CAM was checked on 4+4 images, not the
  full val/test sets — a qualitative spot-check, not a quantitative
  attention-quality metric.
- **Calibration was measured on one model (class-weighted), not
  cross-checked against the unweighted Phase 08 model** — whether
  calibration differs between them is unmeasured.
- No reliability diagram (visual ECE breakdown) has been produced, only
  the scalar ECE.
