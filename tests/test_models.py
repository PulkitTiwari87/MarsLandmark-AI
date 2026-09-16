"""Tests for model construction and baseline computation (CPU, no data needed)."""

import torch

from src.models.baseline import SimpleCNN, majority_class_baseline_accuracy
from src.models.resnet import build_resnet


def test_majority_class_baseline_matches_hand_computed_values():
    # 8 "0", 1 "1", 1 "2" -> majority class 0, accuracy 8/10
    labels = [0] * 8 + [1, 2]
    result = majority_class_baseline_accuracy(labels)
    assert result["majority_class"] == 0
    assert result["accuracy"] == 0.8
    assert 0 < result["macro_f1"] < result["accuracy"]  # macro F1 penalized by ignored classes


def test_simple_cnn_forward_pass_shape():
    model = SimpleCNN(num_classes=8)
    x = torch.randn(2, 3, 227, 227)
    out = model(x)
    assert out.shape == (2, 8)


def test_build_resnet_forward_pass_and_output_classes():
    model = build_resnet("resnet18", num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 227, 227)
    out = model(x)
    assert out.shape == (2, 8)


def test_build_resnet_frozen_backbone_keeps_fc_trainable():
    model = build_resnet("resnet18", num_classes=8, pretrained=False, freeze_backbone=True)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    fc_params = [p for name, p in model.named_parameters() if name.startswith("fc.")]
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in fc_params)


def test_build_resnet_rejects_unknown_architecture():
    try:
        build_resnet("not_a_real_model", num_classes=8)
        assert False, "expected ValueError"
    except ValueError:
        pass
