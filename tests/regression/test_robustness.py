"""
Behavioural regression tests for the trained spam-classification models.

The invariance tests apply changes that should not affect the label (case changes, typos, and
obfuscated URLs, REQ-NF-11). The directional test applies a change that should not lower the
spam score (appending a typical spam phrase to legitimate messages).

Run for each model, for example: MODEL_NAME=logreg pytest -q tests/regression/
"""

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import DATA_DIR, MODELS_DIR
from src.data.perturb import append_text, perturb_dataframe
from src.modeling.train import TEXT_COL, get_count_columns

MODEL_NAMES = {"logreg", "naive_bayes", "linear_svm", "random_forest"}
MODEL_NAME = os.environ.get("MODEL_NAME", "logreg")
LABEL_COLUMN = "label"
SPAM_LABEL = "spam"
NOT_SPAM_LABEL = "not_spam"
SEED = 42
MIN_PERTURBED_F1_MACRO = 0.90
MIN_UNCHANGED_PREDICTIONS = 0.90
MIN_NON_DECREASING_SCORES = 0.90
SPAM_PHRASE = "Congratulations, you won a free prize! Click here to claim your reward now."

if MODEL_NAME not in MODEL_NAMES:
    raise ValueError(f"Unknown MODEL_NAME {MODEL_NAME!r}; expected one of {sorted(MODEL_NAMES)}")

MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.joblib"
TEST_PATH = DATA_DIR / "dataset" / "test.csv"
if not (MODEL_PATH.exists() and TEST_PATH.exists()):
    pytest.skip(
        f"{MODEL_PATH.name} or test.csv not found; run `dvc pull` or `dvc repro` first",
        allow_module_level=True,
    )


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Select the columns used by the model, as the training stage does."""
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df[[TEXT_COL] + get_count_columns(df)]


def spam_score(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return a score that increases with the confidence of the model in the spam class.

    Probabilistic models return the probability of spam. The linear SVM has no probabilities,
    so its decision function is used, oriented so that higher values mean spam.
    """
    spam_index = list(model.classes_).index(SPAM_LABEL)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, spam_index]
    scores = model.decision_function(features)
    return scores if spam_index == 1 else -scores


@pytest.fixture(scope="module")
def model() -> Pipeline:
    """Load the trained model under test."""
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    """Load the held-out test split."""
    return pd.read_csv(TEST_PATH)


@pytest.fixture(scope="module")
def perturbed_df(test_df: pd.DataFrame) -> pd.DataFrame:
    """Perturbed copy of the test split, reproducible through a fixed seed."""
    return perturb_dataframe(test_df, text_col="text", seed=SEED)


def test_macro_f1_on_perturbed_split(model: Pipeline, perturbed_df: pd.DataFrame) -> None:
    """Invariance (REQ-NF-11): the macro F1 score stays high on perturbed messages."""
    predictions = model.predict(model_inputs(perturbed_df))
    score = f1_score(perturbed_df[LABEL_COLUMN], predictions, average="macro")
    assert score >= MIN_PERTURBED_F1_MACRO, (
        f"Model {MODEL_NAME!r} has macro F1={score:.4f} on the perturbed test split; "
        f"minimum required is {MIN_PERTURBED_F1_MACRO:.2f}."
    )


def test_predictions_unchanged_by_perturbation(
    model: Pipeline, test_df: pd.DataFrame, perturbed_df: pd.DataFrame
) -> None:
    """Invariance: most individual predictions do not change after the perturbation."""
    original = model.predict(model_inputs(test_df))
    perturbed = model.predict(model_inputs(perturbed_df))
    unchanged = float(np.mean(original == perturbed))
    assert unchanged >= MIN_UNCHANGED_PREDICTIONS, (
        f"Model {MODEL_NAME!r} keeps {unchanged:.2%} of its predictions after perturbation; "
        f"minimum required is {MIN_UNCHANGED_PREDICTIONS:.0%}."
    )


def test_spam_phrase_does_not_lower_spam_score(model: Pipeline, test_df: pd.DataFrame) -> None:
    """Directional: appending a spam phrase to legitimate messages does not lower their score."""
    legitimate = test_df[test_df[LABEL_COLUMN] == NOT_SPAM_LABEL]
    extended = append_text(legitimate, SPAM_PHRASE, text_col="text")
    before = spam_score(model, model_inputs(legitimate))
    after = spam_score(model, model_inputs(extended))
    non_decreasing = float(np.mean(after >= before))
    assert non_decreasing >= MIN_NON_DECREASING_SCORES, (
        f"Model {MODEL_NAME!r} keeps or raises the spam score of {non_decreasing:.2%} of the "
        f"extended messages; minimum required is {MIN_NON_DECREASING_SCORES:.0%}."
    )
