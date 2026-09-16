"""Majority-class baseline — the reference point every real model must beat."""

from __future__ import annotations

from collections import Counter


class MajorityClassBaseline:
    def __init__(self):
        self.majority_class_: int | None = None

    def fit(self, labels: list[int]) -> "MajorityClassBaseline":
        if not labels:
            raise ValueError("Cannot fit on an empty label list")
        self.majority_class_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, n: int) -> list[int]:
        if self.majority_class_ is None:
            raise RuntimeError("Call fit() before predict()")
        return [self.majority_class_] * n
