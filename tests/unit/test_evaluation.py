"""Unit tests for the evaluation stage."""

import pandas as pd

from src.modeling.evaluation import compute_test_metrics


def test_compute_test_metrics_perfect_predictions() -> None:
    """Perfect predictions give 1.0 on every metric."""
    y = pd.Series(["spam", "ham", "spam", "ham"])
    metrics = compute_test_metrics(y, y)
    assert metrics == {
        "test_accuracy": 1.0,
        "test_precision_macro": 1.0,
        "test_recall_macro": 1.0,
        "test_f1_macro": 1.0,
    }


def test_compute_test_metrics_partial_predictions() -> None:
    """Metrics reflect the errors and are rounded to 4 decimals."""
    y_true = pd.Series(["spam", "spam", "ham", "ham"])
    y_pred = pd.Series(["spam", "ham", "ham", "ham"])
    metrics = compute_test_metrics(y_true, y_pred)
    assert metrics["test_accuracy"] == 0.75
    assert metrics["test_recall_macro"] == 0.75
    assert metrics["test_precision_macro"] == round((1.0 + 2 / 3) / 2, 4)
