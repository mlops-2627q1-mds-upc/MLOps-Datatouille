from omegaconf import OmegaConf
import optuna
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.modeling.train import get_classifier


def test_get_classifier_alternative_models():
    # Mock an Optuna trial and Hydra config space
    trial = optuna.trial.FixedTrial({"C": 1.0, "alpha": 0.5, "n_estimators": 100})
    cfg = OmegaConf.create({
        "C": {"low": 0.1, "high": 10.0, "log": True},
        "alpha": {"low": 0.1, "high": 1.0, "log": True},
        "n_estimators": {"low": 50, "high": 200}
    })

    # Test Naive Bayes
    assert isinstance(get_classifier("naive_bayes", trial, cfg), MultinomialNB)

    # Test Linear SVM
    assert isinstance(get_classifier("linear_svm", trial, cfg), LinearSVC)

    # Test Random Forest
    assert isinstance(get_classifier("random_forest", trial, cfg), RandomForestClassifier)

    # Test unknown model raises ValueError
    with pytest.raises(ValueError, match="Unknown model"):
        get_classifier("unknown_model", trial, cfg)
