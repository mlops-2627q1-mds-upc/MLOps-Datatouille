"""Unit tests for the model selection stage."""
import json
from pathlib import Path

import pytest

from src.modeling.phase_5_selection import build_selection, load_cv_metrics, select_best_model


def test_load_cv_metrics_reads_models_in_order(tmp_path: Path) -> None:
    """Metrics are loaded per model, keeping the given model order."""
    for model, value in (("logreg", 0.9), ("naive_bayes", 0.8)):
        (tmp_path / f"{model}.json").write_text(json.dumps({"cv_f1_macro": value}))
    metrics = load_cv_metrics(tmp_path, ["naive_bayes", "logreg"])
    assert list(metrics) == ["naive_bayes", "logreg"]
    assert metrics["logreg"] == {"cv_f1_macro": 0.9}


def test_load_cv_metrics_missing_file_raises(tmp_path: Path) -> None:
    """A model listed in params but never trained is an error, not silently skipped."""
    with pytest.raises(FileNotFoundError):
        load_cv_metrics(tmp_path, ["logreg"])


def test_select_best_model_picks_highest_metric() -> None:
    """The model with the highest metric value wins."""
    metrics = {"logreg": {"cv_f1_macro": 0.9}, "linear_svm": {"cv_f1_macro": 0.95}}
    assert select_best_model(metrics, "cv_f1_macro") == ("linear_svm", 0.95)


def test_select_best_model_tie_keeps_first() -> None:
    """Ties go to the first model in order, so the selection is deterministic."""
    metrics = {"logreg": {"cv_f1_macro": 0.9}, "naive_bayes": {"cv_f1_macro": 0.9}}
    assert select_best_model(metrics, "cv_f1_macro") == ("logreg", 0.9)


def test_select_best_model_missing_metric_raises() -> None:
    """A metric absent from any model raises instead of being ignored."""
    metrics = {"logreg": {"cv_f1_macro": 0.9}, "naive_bayes": {"cv_accuracy": 0.9}}
    with pytest.raises(ValueError, match="naive_bayes"):
        select_best_model(metrics, "cv_f1_macro")


def test_select_best_model_no_models_raises() -> None:
    """Selecting from no models raises."""
    with pytest.raises(ValueError):
        select_best_model({}, "cv_f1_macro")


def test_build_selection_points_to_model_file() -> None:
    """The selection points to the existing model file instead of copying it."""
    assert build_selection("logreg", "cv_f1_macro", 0.9, Path("models")) == {
        "model": "logreg",
        "metric": "cv_f1_macro",
        "value": 0.9,
        "model_path": "models/logreg.joblib",
    }
