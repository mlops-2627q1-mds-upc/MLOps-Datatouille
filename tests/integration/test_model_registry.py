"""
Integration tests for the publication and loading of the served model.

A small classifier is registered with ``src.modeling.register`` in a throwaway local MLflow
store, and then loaded with ``src.api.model_loader``, as the API does at startup. The tests
need neither network access nor MLflow credentials.
"""

import json
from pathlib import Path
import sys

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.api.model_loader import ModelLoadError, check_model, load_production_model
from src.modeling import register
from src.modeling.predict import predict_messages
from src.modeling.tracking import MODEL_TAG, RUN_ROLE_TAG

EXPERIMENT = "test-registry"
REGISTERED_NAME = "spam-classifier-test"
ALIAS = "production"
MODEL = "naive_bayes"
TRAIN_METRICS = {"cv_f1_macro": 0.98, "test_f1_macro": 0.97, "train_energy_kwh": 0.001}


@pytest.fixture
def selection_path(tracking_uri: str, tmp_path: Path, spam_pipeline: Pipeline) -> Path:
    """
    Selection pointer to a saved model whose MLflow run holds its metrics.

    A trial run of the same model is also logged, which the registration must ignore.
    """
    model_path = tmp_path / f"{MODEL}.joblib"
    joblib.dump(spam_pipeline, model_path)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(tags={MODEL_TAG: MODEL, RUN_ROLE_TAG: "model"}):
        mlflow.log_metrics(TRAIN_METRICS)
    with mlflow.start_run(tags={MODEL_TAG: MODEL, RUN_ROLE_TAG: "trial"}):
        mlflow.log_metric("cv_f1_macro", 0.5)
    selection = {
        "model": MODEL,
        "metric": "cv_f1_macro",
        "value": 0.98,
        "model_path": str(model_path),
    }
    path = tmp_path / "best_model.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    return path


def register_selected(selection_path: Path) -> str:
    """Register the selected model and return its version."""
    selection = register.load_selection(selection_path)
    run_id = register.find_model_run(MlflowClient(), EXPERIMENT, MODEL)
    return register.register_model(selection, run_id, REGISTERED_NAME, ALIAS)


def test_registered_model_is_loaded_through_alias(selection_path: Path) -> None:
    """Test of the round trip: register a model, then load it through its alias."""
    version = register_selected(selection_path)
    loaded = load_production_model(REGISTERED_NAME, ALIAS)

    assert loaded.version == version == "1"
    assert loaded.alias == ALIAS
    assert loaded.metrics == {"cv_f1_macro": 0.98, "test_f1_macro": 0.97}
    [prediction] = predict_messages(loaded.pipeline, ["WIN a FREE prize now!!!"], 0.5)
    assert prediction.label == "spam"


def test_registration_uses_model_run_and_moves_alias(selection_path: Path) -> None:
    """Test of the registration on the model run, and of the alias following new versions."""
    register_selected(selection_path)
    second = register_selected(selection_path)

    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_NAME, ALIAS)
    assert str(version.version) == second == "2"
    run = client.get_run(version.run_id)
    assert run.data.tags[RUN_ROLE_TAG] == "model"
    assert "cv_f1_macro = 0.98" in client.get_model_version(REGISTERED_NAME, "2").description


def test_main_registers_from_command_line(
    selection_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the command-line entry point of the registration."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register",
            "--selection",
            str(selection_path),
            "--experiment",
            EXPERIMENT,
            "--registered-name",
            REGISTERED_NAME,
        ],
    )
    register.main()
    version = MlflowClient().get_model_version_by_alias(REGISTERED_NAME, ALIAS)
    assert str(version.version) == "1"


def test_load_selection_rejects_incomplete_pointer(tmp_path: Path) -> None:
    """Test of the error for a selection pointer without the model path."""
    path = tmp_path / "best_model.json"
    path.write_text(json.dumps({"model": MODEL, "metric": "cv_f1_macro", "value": 1.0}))
    with pytest.raises(KeyError, match="model_path"):
        register.load_selection(path)


def test_find_model_run_fails_without_experiment_or_run(tracking_uri: str) -> None:
    """Test of the errors when the experiment or the model run does not exist."""
    client = MlflowClient()
    with pytest.raises(LookupError, match="does not exist"):
        register.find_model_run(client, "missing-experiment", MODEL)
    mlflow.set_experiment(EXPERIMENT)
    with pytest.raises(LookupError, match="No MLflow run"):
        register.find_model_run(client, EXPERIMENT, MODEL)


def test_load_fails_for_unknown_alias(selection_path: Path) -> None:
    """Test of the startup error when the alias points to no version."""
    register_selected(selection_path)
    with pytest.raises(ModelLoadError, match="staging"):
        load_production_model(REGISTERED_NAME, "staging")


def test_check_model_rejects_model_without_probabilities() -> None:
    """Test of the rejection of a classifier that cannot give a confidence score."""
    pipeline = Pipeline([("classifier", LinearSVC())])
    with pytest.raises(ModelLoadError, match="predict_proba"):
        check_model(pipeline)


def test_check_model_rejects_model_without_spam_class() -> None:
    """Test of the rejection of a classifier trained on other labels."""
    pipeline = Pipeline([("classifier", MultinomialNB())]).fit([[1], [2]], ["ham", "other"])
    with pytest.raises(ModelLoadError, match="'spam' class"):
        check_model(pipeline)
