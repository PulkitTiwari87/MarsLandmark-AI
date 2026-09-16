"""Classification metrics. Wraps sklearn so accuracy alone never stands in
for real evaluation on this class-imbalanced dataset (project rule §14:
"A high overall accuracy must not hide catastrophic minority-class
performance.")
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: dict[int, str]) -> dict:
    labels = sorted(class_names.keys())
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[cls]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [class_names[c] for c in labels],
    }
