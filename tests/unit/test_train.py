"""Unit tests for the training stage (Optuna study and MLflow nesting)."""

from contextlib import nullcontext
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB

from src.modeling import train
from src.modeling.tracking import RUN_ROLE_TAG, start_model_run

EXPERIMENT = "test-experiment"


@pytest.fixture
def cfg() -> DictConfig:
    """Small Hydra-like config tuning naive Bayes with a few trials."""
    return OmegaConf.create(
        {
            "model": {
                "name": "naive_bayes",
                "search_space": {"alpha": {"low": 0.01, "high": 10.0, "log": True}},
            },
            "train": {"max_features": 50, "cv_folds": 2, "random_state": 42, "n_trials": 3},
        }
    )


@pytest.fixture
def xy(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.Series]:
    """Tiny dataset, with emissions tracking disabled so tests stay fast and offline."""
    monkeypatch.setattr(train, "track_emissions", lambda *args, **kwargs: nullcontext())
    texts = ["win free prize now", "are we meeting today", "claim reward", "see you later"]
    labels = ["spam", "ham", "spam", "ham"]
    x = pd.DataFrame({train.TEXT_COL: texts * 3, "n_words": [4, 4, 2, 3] * 3})
    return x, pd.Series(labels * 3, name="label")


def test_build_pipeline_ends_with_classifier() -> None:
    """The pipeline applies the features step, then the given classifier."""
    classifier = MultinomialNB()
    pipeline = train.build_pipeline(classifier, ["n_words"], max_features=10)
    assert list(pipeline.named_steps) == ["features", "classifier"]
    assert pipeline.named_steps["classifier"] is classifier


def test_save_metrics_writes_json(tmp_path: Path) -> None:
    """Metrics are written as JSON, creating missing parent directories."""
    path = tmp_path / "nested" / "metrics.json"
    train.save_metrics({"cv_f1_macro": 0.9}, path)
    assert json.loads(path.read_text()) == {"cv_f1_macro": 0.9}


def test_run_study_is_deterministic(tracking_uri: str, cfg: DictConfig, xy: tuple) -> None:
    """The seeded sampler gives the same best hyperparameters on every run."""
    x, y = xy
    mlflow.set_experiment(EXPERIMENT)
    best_params = [train.run_study(cfg, x, y, ["n_words"]).best_params for _ in range(2)]
    assert best_params[0] == best_params[1]


def test_best_trial_keeps_all_cv_metrics(tracking_uri: str, cfg: DictConfig, xy: tuple) -> None:
    """The best trial stores every CV metric, not only the tuned one."""
    x, y = xy
    mlflow.set_experiment(EXPERIMENT)
    _, metrics = train.train_and_evaluate(cfg, x, y, ["n_words"])
    assert set(metrics) == {f"cv_{name}" for name in train.SCORING} | {"cv_f1_macro_std"}


def test_trials_nest_under_model_run(
    tracking_uri: str, cfg: DictConfig, xy: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trial runs are children of the model run, which evaluation can still resume."""
    monkeypatch.setenv("DVC_EXP_NAME", "my-exp")
    x, y = xy
    with start_model_run("naive_bayes", EXPERIMENT) as model_run:
        train.run_study(cfg, x, y, ["n_words"])
    with start_model_run("naive_bayes", EXPERIMENT, resume=True) as resumed_run:
        mlflow.log_metric("test_f1_macro", 1.0)

    client = MlflowClient(tracking_uri)
    experiment_id = client.get_experiment_by_name(EXPERIMENT).experiment_id
    trials = client.search_runs([experiment_id], f"tags.{RUN_ROLE_TAG} = 'trial'")
    assert len(trials) == cfg.train.n_trials
    assert {r.data.tags["mlflow.parentRunId"] for r in trials} == {model_run.info.run_id}
    assert sorted(r.info.run_name for r in trials)[0] == "trial-0"
    assert resumed_run.info.run_id == model_run.info.run_id