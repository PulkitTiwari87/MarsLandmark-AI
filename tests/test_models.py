import torch

from src.models.baseline import MajorityClassBaseline
from src.models.factory import build_model


def test_majority_class_baseline():
    baseline = MajorityClassBaseline().fit([0, 0, 1, 0, 2])
    assert baseline.majority_class_ == 0
    assert baseline.predict(4) == [0, 0, 0, 0]


def test_majority_class_baseline_requires_fit():
    baseline = MajorityClassBaseline()
    try:
        baseline.predict(3)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_simple_cnn_forward_shape():
    model = build_model("simple_cnn", num_classes=8, in_channels=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 8)


def test_resnet50_forward_shape():
    model = build_model("resnet50", num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 8)


def test_unknown_model_raises():
    try:
        build_model("not_a_model", num_classes=8)
        assert False, "expected ValueError"
    except ValueError:
        pass
