"""Regression tests for the trained spam-classification models."""

import os

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.config import DATA_DIR, MODELS_DIR
from src.modeling.phase_3_training import load_xy

MODEL_NAMES = {"logreg", "naive_bayes", "linear_svm", "random_forest"}
MODEL_NAME = os.environ.get("MODEL_NAME", "logreg")
MIN_ACCURACY = 0.95
MIN_PRECISION_MACRO = 0.95
MIN_RECALL_MACRO = 0.95
MIN_F1_MACRO = 0.95
LABEL_COLUMN = "label"

if MODEL_NAME not in MODEL_NAMES:
    raise ValueError(f"Unknown MODEL_NAME {MODEL_NAME!r}; expected one of {sorted(MODEL_NAMES)}")

MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.joblib"
TEST_PATH = DATA_DIR / "dataset" / "test.csv"
if not (MODEL_PATH.exists() and TEST_PATH.exists()):
    pytest.skip(
        f"{MODEL_PATH.name} or test.csv not found; run `dvc pull` or `dvc repro` first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def model() -> Pipeline:
    """Load the selected model artifact."""
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load every example from the held-out test split."""
    features, labels, _ = load_xy(TEST_PATH, LABEL_COLUMN)
    return features, labels


@pytest.fixture(scope="module")
def model_metrics(
    model: Pipeline,
    test_data: tuple[pd.DataFrame, pd.Series],
) -> dict[str, float]:
    """Calculate classification metrics against all held-out labels."""
    features, labels = test_data
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro")),
        "recall_macro": float(recall_score(labels, predictions, average="macro")),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


@pytest.mark.parametrize(
    "metric, minimum",
    [
        ("accuracy", MIN_ACCURACY),
        ("precision_macro", MIN_PRECISION_MACRO),
        ("recall_macro", MIN_RECALL_MACRO),
        ("f1_macro", MIN_F1_MACRO),
    ],
)
def test_model_meets_minimum_metrics(
    model_metrics: dict[str, float],
    metric: str,
    minimum: float,
) -> None:
    """Require minimum classification metrics on the full held-out test set."""
    score = model_metrics[metric]
    assert score >= minimum, (
        f"Model {MODEL_NAME!r} has {metric}={score:.4f} on the held-out test set; "
        f"minimum required is {minimum:.2f}."
    )
