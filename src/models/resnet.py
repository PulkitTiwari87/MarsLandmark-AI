"""Transfer-learning model for Phase 06: torchvision ResNet with a
replaced final layer for this dataset's 8 classes.

Uses standard ImageNet-pretrained weights (see docs/MODEL_ARCHITECTURE.md
for what "pretrained" means here and why it does not imply the model
learned Mars-specific features from ImageNet — project rule §11). Input
is expected to already be 3-channel (grayscale replicated) and
ImageNet-normalized — see src/data/dataset.py.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet(
    architecture: str = "resnet18",
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # always trainable, even if backbone frozen
    return model
