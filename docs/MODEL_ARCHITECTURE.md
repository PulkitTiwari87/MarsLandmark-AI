# Model Architecture — MarsLandmark-AI

Status: **architectures implemented and unit-tested (forward-pass shape
checks only); no training has occurred.** Nothing below is a performance
claim.

## Candidates implemented

1. **Majority-class baseline** (`src/models/baseline.py`) — predicts the
   most frequent training class for every input. This is the mandatory
   reference point (project rule on baselines): any real model that
   cannot beat this is not doing useful classification.
2. **Simple CNN, trained from scratch** (`src/models/simple_cnn.py`) —
   three conv/BN/ReLU/pool blocks + global average pool + linear head.
   Establishes what is achievable without any pretrained knowledge,
   before attributing gains to transfer learning.
3. **ResNet-50, ImageNet-pretrained** (`src/models/factory.py`) — the
   primary transfer-learning candidate. The final fully-connected layer
   is replaced for the dataset's class count.

Only these two real architectures (plus the trivial baseline) were
selected, per the project's YAGNI guidance ("do not implement every
architecture unnecessarily") — additional architectures (EfficientNet,
ViT) will only be added if the dataset's size/difficulty (confirmed in
Phase 03) motivates it, not speculatively.

## Why ImageNet-pretrained and not from-scratch-only

HiRISE crops (10,815 pre-augmentation, per `docs/DATASET_SELECTION.md`,
itself pending verification) are a modest dataset size for training a
large CNN from scratch. Transfer learning from ImageNet is the standard,
well-justified approach for small/medium real-world image datasets. This
does **not** mean the model "learned Mars from Mars data alone" — the
README and model card will state the pretraining source explicitly, per
project rule 11 ("Do not imply the model learned exclusively from Mars
data if pretrained weights were used").

## Input channel handling

HiRISE crops are single-channel. The pretrained ResNet-50 expects 3
channels (ImageNet RGB). `src/data/transforms.py` replicates the single
channel three times to satisfy this shape requirement — this is a
compatibility step, not a claim that the data has RGB color information.
The from-scratch `SimpleCNN` is used with `in_channels=1` when trained
directly on single-channel input (no replication needed).

## What is not yet decided

- Frozen-vs-fine-tuned backbone strategy (Section 11 of the project
  brief) — requires a first training run to establish a frozen-backbone
  baseline before progressive unfreezing.
- Final hyperparameters in `configs/config.yaml` are engineering defaults,
  not the result of a hyperparameter search (none has been run).
