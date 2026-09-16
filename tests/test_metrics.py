"""Tests for compute_metrics against hand-verifiable small examples."""

from src.training.metrics import compute_metrics


def test_perfect_predictions_score_1_0():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2]
    class_names = {0: "a", 1: "b", 2: "c"}
    result = compute_metrics(y_true, y_pred, class_names)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    for cls in class_names.values():
        assert result["per_class"][cls]["f1"] == 1.0


def test_all_wrong_scores_0():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    class_names = {0: "a", 1: "b"}
    result = compute_metrics(y_true, y_pred, class_names)
    assert result["accuracy"] == 0.0
    assert result["macro_f1"] == 0.0


def test_confusion_matrix_shape_and_labels():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    class_names = {0: "a", 1: "b", 2: "c"}
    result = compute_metrics(y_true, y_pred, class_names)
    assert len(result["confusion_matrix"]) == 3
    assert len(result["confusion_matrix"][0]) == 3
    assert result["confusion_matrix_labels"] == ["a", "b", "c"]


def test_macro_f1_penalized_by_ignored_minority_class():
    # majority-class-always predictor: class 0 dominant, class 1 rare and always missed
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    class_names = {0: "majority", 1: "minority"}
    result = compute_metrics(y_true, y_pred, class_names)
    assert result["accuracy"] == 0.9
    assert result["macro_f1"] < result["accuracy"]  # macro F1 must not hide the missed minority class
    assert result["per_class"]["minority"]["recall"] == 0.0
