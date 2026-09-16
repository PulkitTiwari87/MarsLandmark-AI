"""Classification metrics. A high overall accuracy must not hide poor
minority-class performance, so macro F1 and per-class metrics are always
computed alongside accuracy (project rule on class imbalance)."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    return {
        "accuracy": accuracy,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }
