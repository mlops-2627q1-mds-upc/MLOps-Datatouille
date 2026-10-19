"""
Unit tests for the identification of the most influential features (REQ-NF-08).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.modeling.evaluation import save_top_features
from src.modeling.explain import classifier_scores, top_features
from src.modeling.train import TEXT_COL, build_pipeline

TEXTS = [
    "win free prize now",
    "claim free reward money",
    "urgent prize call now",
    "free money win claim",
    "are we meeting today",
    "see you at lunch",
    "the report is attached",
    "thanks for the meeting notes",
]
LABELS = ["spam"] * 4 + ["not_spam"] * 4


def _fitted_pipeline(classifier: BaseEstimator):
    """
    Fit the project pipeline on a tiny dataset.
    """
    x = pd.DataFrame({TEXT_COL: TEXTS, "n_words": [len(t.split()) for t in TEXTS]})
    pipeline = build_pipeline(classifier, ["n_words"], max_features=50)
    return pipeline.fit(x, pd.Series(LABELS))


@pytest.mark.parametrize(
    "classifier, score_type",
    [
        (LogisticRegression(random_state=0), "coefficient"),
        (LinearSVC(random_state=0), "coefficient"),
        (MultinomialNB(), "log_probability_ratio"),
        (RandomForestClassifier(n_estimators=5, random_state=0), "importance"),
    ],
)
def test_top_features_sorted_by_absolute_score(classifier: BaseEstimator, score_type: str) -> None:
    """
    Every model returns at most 20 features, sorted by decreasing absolute score.
    """
    result = top_features(_fitted_pipeline(classifier))
    scores = [abs(feature["score"]) for feature in result["features"]]

    assert result["score_type"] == score_type
    assert result["positive_class"] == "spam"
    assert 0 < len(result["features"]) <= 20
    assert scores == sorted(scores, reverse=True)
    assert all(
        feature["feature"].startswith(("tfidf__", "counts__")) for feature in result["features"]
    )


def test_top_features_respects_n() -> None:
    """The number of returned features can be limited."""
    result = top_features(_fitted_pipeline(LogisticRegression(random_state=0)), n=3)
    assert len(result["features"]) == 3


def test_spam_words_push_towards_spam() -> None:
    """The most influential words of a linear model on spam-like text favour the spam class."""
    result = top_features(_fitted_pipeline(LogisticRegression(random_state=0)))
    scores = {feature["feature"]: feature["score"] for feature in result["features"]}
    assert scores["tfidf__free"] > 0


def test_classifier_scores_rejects_unsupported_models() -> None:
    """A classifier without coefficients, probabilities, or importances is rejected."""
    with pytest.raises(TypeError, match="Unsupported classifier"):
        classifier_scores(object())


def test_save_top_features_writes_json(tmp_path: Path) -> None:
    """
    The features are written as JSON, creating missing parent directories.
    """
    path = tmp_path / "features" / "logreg.json"
    features = {"score_type": "coefficient", "positive_class": "spam", "features": []}
    save_top_features(features, path)
    assert json.loads(path.read_text(encoding="utf-8")) == features
