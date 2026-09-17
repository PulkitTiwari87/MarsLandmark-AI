"""Calibration for Phase 09.

Project rule §21: "Do not interpret raw softmax confidence as guaranteed
probability." This module measures how miscalibrated a model's softmax
outputs are (Expected Calibration Error) and provides temperature
scaling (Guo et al. 2017) as one standard, minimal-assumption fix — a
single learned scalar dividing the logits, fit on validation data, never
changing which class is predicted (argmax is invariant to positive
scaling of logits).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_calibration_error(
    confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE: bin predictions by confidence, compare each bin's average
    confidence to its actual accuracy, weight by bin size."""
    if not (confidences.shape == predictions.shape == labels.shape):
        raise ValueError("confidences, predictions, labels must have the same shape")
    correct = predictions.eq(labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    n = confidences.size(0)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum().item()
        if bin_size == 0:
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (bin_size / n) * torch.abs(bin_confidence - bin_accuracy)
    return float(ece.item())


class TemperatureScaler(nn.Module):
    """Wraps a trained classifier; learns a single scalar temperature on
    held-out logits to minimize NLL, per Guo et al. 2017."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """Fits self.temperature via LBFGS on (logits, labels), typically
        computed once over the full validation set. Returns the final NLL."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            final_loss = nll_criterion(self.forward(logits), labels).item()
        return final_loss

    def calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(logits), dim=1)
