"""Baseline models for Phase 05: majority-class baseline and a small CNN.

Both exist to establish a reference point before transfer learning
(Phase 06) — project rule §9: "Before using sophisticated models,
establish a baseline."
"""

from __future__ import annotations

import collections

import torch
import torch.nn as nn


def majority_class_baseline_accuracy(labels: list[int]) -> dict:
    """Accuracy/macro-F1 of always predicting the most frequent class in `labels`.

    A trivial reference point: any real model should clearly beat this,
    especially given the measured 83.6% class-0 imbalance (docs/DATASET.md).
    """
    counts = collections.Counter(labels)
    majority_class, majority_count = counts.most_common(1)[0]
    n = len(labels)
    accuracy = majority_count / n

    # Macro F1 for "always predict majority_class": recall=1 for that
    # class (all its instances correctly predicted), recall=0 for every
    # other class (never predicted); precision for majority_class equals
    # its accuracy, precision undefined (treated as 0) for others.
    n_classes = len(counts)
    precision_majority = majority_count / n
    f1_majority = 2 * precision_majority * 1.0 / (precision_majority + 1.0) if precision_majority > 0 else 0.0
    macro_f1 = f1_majority / n_classes  # every other class contributes F1=0

    return {
        "majority_class": majority_class,
        "majority_class_count": majority_count,
        "total": n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


class SimpleCNN(nn.Module):
    """Small from-scratch CNN baseline (not pretrained) for 3x227x227 input."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 227 -> 113
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 113 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
