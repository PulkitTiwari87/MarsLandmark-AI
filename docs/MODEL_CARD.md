# Model Card — MarsLandmark-AI

## Model name / version

`exp_resnet18_finetuned` (ResNet18, fully fine-tuned) — the model
selected after Phase 06, evaluated once in Phase 08. See
`experiments/experiments.csv` for the exact run record and
`docs/REPRODUCIBILITY.md` for how to reproduce it.

## Task

Multi-class image classification: given a 227×227 grayscale HiRISE
landmark crop, predict one of 8 classes (other, crater, dark dune, slope
streak, bright dune, impact ejecta, swiss cheese, spider). Not detection
or localization — the model does not output bounding boxes or a location
within a larger image (see `docs/DATASET_SELECTION.md` for why this
dataset only supports classification).

## Training data

HiRISE landmark dataset v3 (NASA/JPL, Zenodo DOI `10.5281/zenodo.2538136`,
CC-BY 4.0), 73,031 images. Trained on the 51,128-image `train` split from
a leakage-safe, source-strip-grouped assignment (`docs/DATA_SPLIT.md`).
Full provenance: `docs/DATASET.md`.

## Preprocessing

Grayscale replicated to 3 channels, ImageNet-normalized. No additional
augmentation beyond what the archive already ships (6x per original
landmark — rotations, flips, brightness). See `docs/DATA_PIPELINE.md`.

## Training procedure

Pretrained ImageNet weights (`torchvision` `ResNet18_Weights.DEFAULT`),
fully fine-tuned (no frozen layers), AdamW, lr=1e-4, weight_decay=1e-4,
cross-entropy loss (unweighted), 15 epochs, batch size 32, seed 42, on a
Tesla T4 GPU in Google Colab. Full hyperparameters:
`experiments/experiments.csv` row `exp_resnet18_finetuned`.

## Intended use

Research/triage assistance for classifying HiRISE landmark crops that
match this dataset's distribution (same instrument, similar crop size
and framing, similar terrain types). **Not validated for:** full HiRISE
scenes (this model only sees pre-cropped landmarks), other Mars imagery
instruments, or any safety-critical or autonomous decision-making use.

## Measured performance (docs/RESULTS.md)

| Metric | Value |
|---|---:|
| Test accuracy | 93.11% |
| Test macro F1 | 71.56% |

**Per-class reliability is not uniform** — see `docs/RESULTS.md`'s
per-class table. In particular:
- **"spider": 0% recall, 0% F1 (n=7 test examples).** The model has
  never correctly identified a spider-class landmark in testing. Do not
  trust this model for spider-class detection.
- **"swiss cheese": 57.6% recall** — the model misses close to half of
  actual swiss-cheese landmarks (usually calling them "other").
- **"dark dune": 48.8% precision** — roughly half of the model's
  dark-dune predictions are wrong.
- The dominant "other" and well-represented classes (crater, bright
  dune, impact ejecta, slope streak) perform substantially better
  (F1 0.74–0.97).

## Known limitations

- Complete failure on the rarest class ("spider") — see
  `docs/ERROR_ANALYSIS.md`.
- Run-to-run non-determinism despite a fixed seed (cuDNN not set to
  deterministic mode) — see `docs/REPRODUCIBILITY.md`. Retraining this
  exact configuration will not reproduce bit-identical metrics.
- No calibration has been measured yet — raw softmax confidence (as
  returned by `src/api/main.py`'s `/predict` endpoint) should not be
  interpreted as a calibrated probability (project rule §21;
  `src/training/calibration.py` exists but has not been run against this
  checkpoint — see `docs/EXPLAINABILITY.md`).
- No Grad-CAM/explainability check has been run against this specific
  checkpoint yet — whether the model is looking at plausible terrain
  features (vs. something spurious) is unconfirmed.
- Trained and evaluated on one grouped split; performance on a
  differently-drawn split, or on HiRISE imagery from source strips not
  in this 180-strip dataset, is untested.

## Ethical / scientific considerations

This model classifies terrain features for planetary-science research
triage, not a domain with direct safety-critical consequences for
people. Its class-imbalance-driven failure modes (above) mean it should
not be used as the sole basis for any downstream scientific claim about
rare landmark types (spider, impact ejecta) without human verification.

## Computational requirements

Inference: CPU-feasible (ResNet18, single image, no batching required)
via `src/api/main.py`; training took 52.9 minutes on a Tesla T4 GPU.
