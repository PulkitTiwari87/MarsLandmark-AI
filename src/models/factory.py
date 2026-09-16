"""Model construction. Adding an architecture here is a deliberate choice,
not a default — see docs/MODEL_ARCHITECTURE.md for which candidates were
selected and why."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

from src.models.simple_cnn import SimpleCNN

SUPPORTED_MODELS = ("simple_cnn", "resnet50")


def build_model(name: str, num_classes: int, pretrained: bool = True, in_channels: int = 3) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model {name!r}, expected one of {SUPPORTED_MODELS}")
