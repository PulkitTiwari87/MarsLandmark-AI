from src.training.metrics import compute_classification_metrics


def test_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["a", "b", "c"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_all_wrong_predictions_flagged_low():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["a", "b"])
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_confusion_matrix_shape():
    y_true = [0, 1]
    y_pred = [0, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=["a", "b"])
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
