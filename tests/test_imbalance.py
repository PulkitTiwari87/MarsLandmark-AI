"""Tests for class-weight computation, including against the real measured
train-split counts from reports/split_report.json (docs/DATASET.md)."""

import math

import pytest
import torch

from src.training.imbalance import compute_class_weights


def test_balanced_weights_hand_computed_small_example():
    # 2 classes: 8 of class 0, 2 of class 1, total 10
    weights = compute_class_weights({0: 8, 1: 2})
    # weight_c = N / (n_classes * count_c)
    assert math.isclose(weights[0].item(), 10 / (2 * 8))
    assert math.isclose(weights[1].item(), 10 / (2 * 2))
    assert weights[1] > weights[0]  # rarer class gets a larger weight


def test_uniform_counts_give_uniform_weights():
    weights = compute_class_weights({0: 100, 1: 100, 2: 100})
    assert torch.allclose(weights, torch.ones(3))


def test_real_train_split_counts_from_dataset_measurement():
    # From reports/split_report.json's train split (docs/DATASET.md) — the
    # actual measured class distribution used for the Phase 07 experiment.
    train_counts = {0: 43533, 1: 3143, 2: 756, 3: 1981, 4: 1029, 5: 91, 6: 133, 7: 462}
    weights = compute_class_weights(train_counts)

    assert weights.shape == (8,)
    # class 0 (dominant) gets the smallest weight, class 5 (rarest) the largest
    assert weights.argmin().item() == 0
    assert weights.argmax().item() == 5
    assert math.isclose(weights[5].item(), 51128 / (8 * 91), rel_tol=1e-6)


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_class_weights({})


def test_rejects_non_contiguous_classes():
    with pytest.raises(ValueError):
        compute_class_weights({0: 5, 2: 5})  # missing class 1


def test_rejects_zero_count():
    with pytest.raises(ValueError):
        compute_class_weights({0: 5, 1: 0})
