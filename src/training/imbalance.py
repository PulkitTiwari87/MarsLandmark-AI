"""Class-imbalance mitigation for Phase 07.

Measured motivation (docs/DATASET.md, docs/ERROR_ANALYSIS.md): the
training split is 85.15% class 0 ("other"), and the Phase 06 fine-tuned
model completely failed on "spider" (0/7 test images correct) while
under-recalling "swiss cheese" (57.6%). This module computes standard
"balanced" inverse-frequency class weights for use in a weighted
cross-entropy loss — one candidate mitigation, to be compared against the
unweighted baseline's actual measured result (val macro F1 0.7188,
docs/EXPERIMENTS.md), not assumed to help.
"""

from __future__ import annotations

import torch


def compute_class_weights(class_counts: dict[int, int]) -> torch.Tensor:
    """Standard 'balanced' weights: weight_c = N / (n_classes * count_c).

    Returned as a tensor indexed 0..max(class_counts), suitable for
    nn.CrossEntropyLoss(weight=...). Every class in 0..max key must be
    present in class_counts with count > 0, or this raises.
    """
    if not class_counts:
        raise ValueError("class_counts must not be empty")
    n_classes = len(class_counts)
    total = sum(class_counts.values())
    max_class = max(class_counts)
    if set(class_counts) != set(range(max_class + 1)):
        raise ValueError(f"class_counts must cover 0..{max_class} contiguously, got keys {sorted(class_counts)}")

    weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls, count in class_counts.items():
        if count <= 0:
            raise ValueError(f"class {cls} has non-positive count {count}")
        weights[cls] = total / (n_classes * count)
    return weights
