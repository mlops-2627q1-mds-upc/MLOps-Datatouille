"""
Unit tests for the training stage (Optuna study and MLflow nesting).
"""

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB

from src.modeling import train
from src.modeling.tracking import RUN_ROLE_TAG, start_model_run

EXPERIMENT = "test-experiment"
FAKE_RECORD = {"project_name": "fake", "duration": 2.0, "energy_consumed": 0.5, "emissions": 0.1}


@contextmanager
def fake_track_emissions(*args: object, **kwargs: object) -> Iterator[SimpleNamespace]:
    """
    Stand-in for ``track_emissions`` yielding a stopped tracker with a fixed measurement.
    """
    yield SimpleNamespace(final_emissions_data=SimpleNamespace(values=dict(FAKE_RECORD)))


@pytest.fixture
def cfg() -> DictConfig:
    """
    Small Hydra-like config tuning naive Bayes with a few trials.
    """
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
    """
    Tiny dataset, with emissions tracking faked so tests stay fast and offline.
    """
    monkeypatch.setattr(train, "track_emissions", fake_track_emissions)
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
    """
    Metrics are written as JSON, creating missing parent directories.
    """
    path = tmp_path / "nested" / "metrics.json"
    train.save_metrics({"cv_f1_macro": 0.9}, path)
    assert json.loads(path.read_text()) == {"cv_f1_macro": 0.9}


def test_run_study_is_deterministic(tracking_uri: str, cfg: DictConfig, xy: tuple) -> None:
    """
    The seeded sampler gives the same best hyperparameters on every run.
    """
    x, y = xy
    mlflow.set_experiment(EXPERIMENT)
    best_params = [train.run_study(cfg, x, y, ["n_words"]).best_params for _ in range(2)]
    assert best_params[0] == best_params[1]


def test_best_trial_keeps_all_cv_metrics(tracking_uri: str, cfg: DictConfig, xy: tuple) -> None:
    """
    The best trial stores every CV metric, not only the tuned one.
    """
    x, y = xy
    mlflow.set_experiment(EXPERIMENT)
    _, metrics, _ = train.train_and_evaluate(cfg, x, y, ["n_words"])
    assert set(metrics) == {f"cv_{name}" for name in train.SCORING} | {"cv_f1_macro_std"}


def test_trials_nest_under_model_run(
    tracking_uri: str, cfg: DictConfig, xy: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Trial runs are children of the model run, which evaluation can still resume.
    """
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


def test_emission_rows_cover_every_trial_and_final_fit(
    tracking_uri: str, cfg: DictConfig, xy: tuple
) -> None:
    """
    Training returns one labelled measurement per trial plus one for the final fit.
    """
    mlflow.set_experiment(EXPERIMENT)
    x, y = xy
    _, _, rows = train.train_and_evaluate(cfg, x, y, ["n_words"])
    assert [row["step"] for row in rows] == ["trial-0", "trial-1", "trial-2", "final"]
    assert {row["model"] for row in rows} == {"naive_bayes"}
    assert all(row["emissions"] == FAKE_RECORD["emissions"] for row in rows)


def test_emission_metric_names_do_not_depend_on_model(
    tracking_uri: str, cfg: DictConfig, xy: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Emission metrics share names across models, so MLflow can plot them together.
    """
    prefixes = []

    @contextmanager
    def recording_track_emissions(
        stage: str, metric_prefix: str | None = None
    ) -> Iterator[SimpleNamespace]:
        prefixes.append(metric_prefix or stage)
        with fake_track_emissions() as tracker:
            yield tracker

    monkeypatch.setattr(train, "track_emissions", recording_track_emissions)
    mlflow.set_experiment(EXPERIMENT)
    x, y = xy
    train.train_and_evaluate(cfg, x, y, ["n_words"])
    assert set(prefixes) == {"train_trial", "train_final"}
