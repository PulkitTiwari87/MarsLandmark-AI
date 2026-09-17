"""Tests for Grad-CAM, against untrained models (structure/shape/range
correctness — not visual correctness, which needs a real trained model)."""

import torch

from src.explainability.gradcam import GradCAM
from src.models.baseline import SimpleCNN
from src.models.resnet import build_resnet


def test_gradcam_on_simplecnn_returns_valid_heatmap():
    model = SimpleCNN(num_classes=8)
    target_layer = model.features[-2]  # last conv block before pooling
    cam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 227, 227)
    heatmap, predicted_class = cam.generate(x)

    assert heatmap.shape == (227, 227)
    assert 0 <= predicted_class < 8
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    cam.remove_hooks()


def test_gradcam_on_resnet_returns_valid_heatmap():
    model = build_resnet("resnet18", num_classes=8, pretrained=False)
    target_layer = model.layer4[-1]  # standard Grad-CAM target for ResNet
    cam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 227, 227)
    heatmap, predicted_class = cam.generate(x)

    assert heatmap.shape == (227, 227)
    assert 0 <= predicted_class < 8
    cam.remove_hooks()


def test_gradcam_respects_explicit_target_class():
    model = build_resnet("resnet18", num_classes=8, pretrained=False)
    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    x = torch.randn(1, 3, 227, 227)
    _, used_class = cam.generate(x, target_class=3)
    assert used_class == 3
    cam.remove_hooks()


def test_gradcam_rejects_wrong_input_shape():
    model = SimpleCNN(num_classes=8)
    target_layer = model.features[-2]
    cam = GradCAM(model, target_layer)
    try:
        cam.generate(torch.randn(2, 3, 227, 227))  # batch size 2, not 1
        assert False, "expected ValueError"
    except ValueError:
        pass
    finally:
        cam.remove_hooks()
