# Model Architecture — MarsLandmark-AI

Status: **Code implemented and unit-tested** (forward-pass shape checks,
CPU); **not yet trained** on the real dataset. No accuracy/loss numbers
in this document — see `docs/EXPERIMENTS.md`, which will read `NOT YET
MEASURED` until the Colab training notebook is actually run.

## Baseline (Phase 05) — `src/models/baseline.py`

- **Majority-class baseline** (`majority_class_baseline_accuracy`): not a
  model, a reference number. Given the measured 83.6% "other" class share
  (`docs/DATASET.md`), always predicting "other" would score ~83.6%
  accuracy — this baseline exists specifically so that number is on
  record and any trained model's accuracy is interpreted against it, not
  in isolation.
- **SimpleCNN**: 3 conv blocks (16→32→64 channels, BatchNorm, ReLU,
  MaxPool) + global average pool + linear classifier. Trained from
  scratch (no pretrained weights). Input: 3×227×227 (grayscale replicated
  to 3 channels, see `docs/DATA_PIPELINE.md`).

## Transfer learning (Phase 06) — `src/models/resnet.py`

`build_resnet(architecture, num_classes, pretrained, freeze_backbone)`
wraps `torchvision.models.resnet18`/`resnet50` with the final fully
connected layer replaced for 8 classes. `pretrained=True` loads standard
ImageNet weights via `torchvision`'s `Weights.DEFAULT` enum.

**What "pretrained" does and does not mean here** (project rule §11): the
backbone's convolutional filters were learned on ImageNet (natural photos
of everyday objects), not on Mars imagery. Any transfer-learning result
reflects how well those generic visual features happen to transfer to
grayscale HiRISE crops — it is not evidence the model learned anything
Mars-specific from ImageNet. `freeze_backbone=True` supports the
frozen-backbone-first, then progressive-fine-tuning workflow the project
rules require testing, but which architecture/freezing strategy performs
best is `NOT YET MEASURED`.

## Why ResNet, and why not everything

Project rule §10: "Do not implement every architecture unnecessarily."
ResNet18/50 were chosen as the initial candidates because: they are
well-established baselines for transfer learning on limited compute,
`torchvision` ships pretrained weights directly (no extra dependency —
`torchvision` is already required), and the dataset (73k images, 8
classes, single source domain) does not obviously call for a
Transformer-scale model. EfficientNet/ConvNeXt/ViT are not ruled out, but
are not implemented until a measured ResNet result shows they're needed —
avoiding premature architecture proliferation.

## Known constraint carried from the data pipeline

Images are grayscale; this project replicates to 3 channels rather than
modifying the backbone's first conv layer (see `docs/DATA_PIPELINE.md`
for the reasoning). This is a documented engineering choice, not a
"novel architecture," and can be revisited if a measured result suggests
the replication is losing useful information.
