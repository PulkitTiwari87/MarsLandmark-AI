"""Reproducibility helper. Full bitwise determinism is not guaranteed on
all hardware/CUDA versions — see docs/REPRODUCIBILITY.md once training is
actually run on target hardware."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
