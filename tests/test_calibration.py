"""Tests for ECE and temperature scaling."""

import torch

from src.training.calibration import TemperatureScaler, expected_calibration_error


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # 10 predictions all at confidence 0.8, exactly 8/10 correct -> ECE ~0
    confidences = torch.full((10,), 0.8)
    predictions = torch.zeros(10, dtype=torch.long)
    labels = torch.zeros(10, dtype=torch.long)
    labels[8:] = 1  # 8 correct, 2 wrong -> accuracy 0.8, matches confidence
    ece = expected_calibration_error(confidences, predictions, labels)
    assert abs(ece) < 1e-5


def test_ece_is_high_for_overconfident_wrong_predictions():
    # Always 99% confident, but always wrong -> large ECE
    confidences = torch.full((10,), 0.99)
    predictions = torch.zeros(10, dtype=torch.long)
    labels = torch.ones(10, dtype=torch.long)
    ece = expected_calibration_error(confidences, predictions, labels)
    assert ece > 0.9


def test_ece_rejects_mismatched_shapes():
    try:
        expected_calibration_error(torch.rand(5), torch.zeros(4, dtype=torch.long), torch.zeros(5, dtype=torch.long))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_temperature_scaling_reduces_nll_on_miscalibrated_logits():
    # A realistic overconfident model: 75% correct, but very high-confidence
    # on every prediction (including the 25% it gets wrong) -> genuinely
    # miscalibrated, unlike a perfect-margin/zero-error toy case where
    # temperature scaling has nothing useful to fix.
    torch.manual_seed(0)
    n, n_classes = 500, 4
    true_labels = torch.randint(0, n_classes, (n,))
    predicted_labels = true_labels.clone()
    wrong_idx = torch.randperm(n)[:125]
    predicted_labels[wrong_idx] = (true_labels[wrong_idx] + 1) % n_classes

    logits = torch.randn(n, n_classes) * 0.2
    logits[torch.arange(n), predicted_labels] += 6.0

    criterion = torch.nn.CrossEntropyLoss()
    nll_before = criterion(logits, true_labels).item()

    scaler = TemperatureScaler()
    nll_after = scaler.fit(logits, true_labels)

    assert nll_after < nll_before
    # Predictions (argmax) must be unchanged by temperature scaling
    assert torch.equal(logits.argmax(dim=1), scaler.forward(logits).argmax(dim=1))
