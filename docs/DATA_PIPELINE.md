# Data Pipeline — MarsLandmark-AI

Status: **code written and unit-tested against synthetic fixtures; not yet
run against the real dataset** (blocked on acquisition — see
`docs/DATASET.md`).

## Components

- `src/data/labels.py` — parses the dataset's `filename label_idx` label
  files and its class-index-to-name map. The class ordering is a
  documented placeholder (`DEFAULT_CLASS_NAMES`) pending confirmation
  against the real class-map file — see the module docstring.
- `src/data/splits.py` — loads named splits and performs a filename-level
  leakage check. See `docs/DATA_SPLIT.md`.
- `src/data/dataset.py` — `HiRISELandmarkDataset`, a PyTorch `Dataset`
  that loads a crop as grayscale (`convert("L")`, since HiRISE crops are
  single-channel reflectance data, not RGB) and applies a transform.
- `src/data/transforms.py` — `build_transforms(train, image_size,
  three_channel)`. Every included augmentation has a stated physical
  justification in the module docstring (flips/90° rotation for
  orientation-invariant overhead imagery); color jitter is deliberately
  excluded. Channel replication to 3-channel is applied only to satisfy
  an ImageNet-pretrained backbone's input shape and is explicitly
  documented as not adding information.

## Determinism

`src/training/seed.py::set_seed` seeds `random`, `numpy`, and `torch`
(CPU and CUDA). Full bitwise reproducibility across hardware/driver
versions is not claimed — see `docs/REPRODUCIBILITY.md` once training is
actually run on target hardware.

## What is validated so far

`tests/test_labels.py`, `tests/test_splits.py`, `tests/test_dataset.py`,
and `tests/test_transforms.py` exercise this code against small synthetic
fixtures (solid-color PNGs/JPEGs generated in the test, explicitly not
real NASA imagery) to confirm parsing, shape, and channel-count
correctness. They do not and cannot validate anything about the real
dataset's actual content, class balance, or image statistics — that is
Phase 03 (EDA), which requires the real files.

## Explicitly out of scope until real data arrives

- Per-channel normalization statistics computed from the real dataset
  (currently using ImageNet stats, justified only by pretrained-backbone
  reuse — see `docs/MODEL_ARCHITECTURE.md`).
- Any augmentation tuned by observed validation performance.
- Class-imbalance handling (weighted loss/sampler) — cannot be sized
  correctly without the real class distribution (Phase 03).
