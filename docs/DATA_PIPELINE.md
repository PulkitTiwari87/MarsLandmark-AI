# Data Pipeline — MarsLandmark-AI

Status: **Implemented** (`src/data/dataset.py`), tested against a
synthetic fixture (`tests/test_dataset.py`); not yet run against the full
73,031-image archive end-to-end in a training loop (that happens in the
Colab training notebook — local GPU training was declined for this
project, see `docs/PROJECT_OVERVIEW.md`).

## Components

1. **Acquisition** — `src/data/download.py` (Phase 02, done).
2. **Validation** — `src/data/validate.py` (Phase 03, done).
3. **Split** — `src/data/split.py` (Phase 04, done) — see `docs/DATA_SPLIT.md`.
4. **Dataset/DataLoader** — `src/data/dataset.py`:
   - `HiRISELandmarkDataset(data_dir, split_manifest_path, split)` reads
     `labels-map-proj-v3.txt` and the split manifest, filters to one split.
   - `build_transform()`: `Grayscale(num_output_channels=3)` →
     `ToTensor()` → `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.

## Preprocessing decisions and why

- **Grayscale → 3-channel replication**, not a modified first conv layer.
  Simpler, and lets Phase 06 use standard torchvision pretrained weights
  unmodified. Measured fact this responds to: all 73,031 images are
  single-channel (`docs/DATASET.md`).
- **ImageNet normalization stats**, applied identically to the 3
  replicated channels. Standard practice when reusing ImageNet-pretrained
  weights; not derived from this dataset's own pixel statistics.
- **No additional geometric/color augmentation.** The archive already
  ships 6x augmentation per original landmark (rotation/flip/brightness —
  `docs/DATASET.md`). Adding more was a deliberate choice, not an
  oversight — see `src/data/dataset.py`'s docstring. Revisit only if
  Phase 07 measures overfitting that this doesn't already address.
- **No resizing.** Source images are already a uniform 227×227
  (measured, 0 exceptions) — resizing would be a no-op that adds risk of
  silently masking a future data-quality regression, so it's omitted.

## Reproducibility

- Split assignment is deterministic (no RNG) — see `docs/DATA_SPLIT.md`.
- `configs/config.yaml: seed: 42` seeds model init/training (Phase 05+),
  not the split (which needs no seed).
- `data/processed/split_manifest.csv` is generated, not hand-edited, and
  is git-ignored — regenerate via `python -m src.data.split`.
