"""Unit tests for the MLflow tracking helpers of the training stage."""

import hashlib
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from src.modeling.tracking import (
    EXP_REF_TAG,
    NOTE_TAG,
    RUN_ROLE_TAG,
    build_exp_ref,
    build_exp_tags,
    file_md5,
    log_dataset,
    start_model_run,
)

EXPERIMENT = "test-experiment"


def _runs(uri: str) -> list:
    """Return all runs of the test experiment."""
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    return client.search_runs([experiment.experiment_id])


def test_build_exp_ref_matches_dvc_format() -> None:
    """The git ref follows DVC's refs/exps/<sha[:2]>/<sha[2:]>/<name> layout."""
    assert build_exp_ref("my-exp", "abc123") == "refs/exps/ab/c123/my-exp"


def test_build_exp_tags_include_restore_info() -> None:
    """Tags hold the exp ref and a note with the restore commands."""
    tags = build_exp_tags("my-exp", "abc123", "child")
    assert tags[EXP_REF_TAG] == "refs/exps/ab/c123/my-exp"
    assert "dvc exp pull origin my-exp" in tags[NOTE_TAG]
    assert "dvc exp apply my-exp" in tags[NOTE_TAG]


def test_build_exp_tags_without_baseline_rev() -> None:
    """Without a baseline commit there is no exp ref, but the restore note is kept."""
    tags = build_exp_tags("my-exp", None, "parent")
    assert EXP_REF_TAG not in tags
    assert "dvc exp apply my-exp" in tags[NOTE_TAG]


def test_file_md5_matches_hashlib(tmp_path: Path) -> None:
    """The dataset hash matches hashlib's MD5 of the file content."""
    path = tmp_path / "data.csv"
    path.write_bytes(b"text,label\nhi,ham\n")
    assert file_md5(path) == hashlib.md5(b"text,label\nhi,ham\n").hexdigest()


def test_runs_share_one_parent_under_dvc_exp(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All models of one DVC experiment log child runs under a single parent."""
    monkeypatch.setenv("DVC_EXP_NAME", "my-exp")
    monkeypatch.setenv("DVC_EXP_BASELINE_REV", "abc123")

    for model in ("logreg", "naive_bayes"):
        with start_model_run(model, EXPERIMENT):
            mlflow.log_metric("cv_f1_macro", 0.9)

    runs = _runs(tracking_uri)
    parents = [r for r in runs if r.data.tags[RUN_ROLE_TAG] == "parent"]
    children = [r for r in runs if r.data.tags[RUN_ROLE_TAG] == "child"]
    assert len(parents) == 1
    assert parents[0].info.run_name == "my-exp"
    assert parents[0].info.status == "FINISHED"
    assert sorted(r.info.run_name for r in children) == ["logreg", "naive_bayes"]
    assert {r.data.tags["mlflow.parentRunId"] for r in children} == {parents[0].info.run_id}
    assert {r.data.tags[EXP_REF_TAG] for r in runs} == {"refs/exps/ab/c123/my-exp"}


def test_new_dvc_exp_gets_new_parent(tracking_uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each DVC experiment gets its own parent run."""
    for exp_name in ("exp-a", "exp-b"):
        monkeypatch.setenv("DVC_EXP_NAME", exp_name)
        with start_model_run("logreg", EXPERIMENT):
            pass

    parents = [r for r in _runs(tracking_uri) if r.data.tags[RUN_ROLE_TAG] == "parent"]
    assert sorted(r.info.run_name for r in parents) == ["exp-a", "exp-b"]


def test_standalone_run_outside_dvc_exp(tracking_uri: str) -> None:
    """Outside dvc exp run, a standalone run named after the model is logged."""
    with start_model_run("logreg", EXPERIMENT):
        pass

    runs = _runs(tracking_uri)
    assert len(runs) == 1
    assert runs[0].info.run_name == "logreg"
    assert "mlflow.parentRunId" not in runs[0].data.tags


def test_resume_reopens_model_child_run(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluation adds its metrics to the model's training run, not a new run."""
    monkeypatch.setenv("DVC_EXP_NAME", "my-exp")
    for model in ("logreg", "naive_bayes"):
        with start_model_run(model, EXPERIMENT):
            mlflow.log_metric("cv_f1_macro", 0.9)
    with start_model_run("logreg", EXPERIMENT, resume=True):
        mlflow.log_metric("test_f1_macro", 0.8)

    children = [r for r in _runs(tracking_uri) if r.data.tags[RUN_ROLE_TAG] == "child"]
    assert len(children) == 2
    logreg = next(r for r in children if r.info.run_name == "logreg")
    assert logreg.data.metrics == {"cv_f1_macro": 0.9, "test_f1_macro": 0.8}
    assert logreg.info.status == "FINISHED"


def test_resume_creates_child_run_when_missing(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If training was skipped (cached by DVC), evaluation still logs a child run."""
    monkeypatch.setenv("DVC_EXP_NAME", "my-exp")
    with start_model_run("logreg", EXPERIMENT, resume=True):
        mlflow.log_metric("test_f1_macro", 0.8)

    children = [r for r in _runs(tracking_uri) if r.data.tags[RUN_ROLE_TAG] == "child"]
    assert [r.info.run_name for r in children] == ["logreg"]
    assert "mlflow.parentRunId" in children[0].data.tags


def test_log_dataset_uses_full_file_md5_as_digest(tracking_uri: str, tmp_path: Path) -> None:
    """The dataset is logged with its context and the whole-file MD5 as digest."""
    path = tmp_path / "test.csv"
    df = pd.DataFrame({"clean_text": ["hi", "win now"], "label": ["ham", "spam"]})
    df.to_csv(path, index=False)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        log_dataset(df[["clean_text"]], df["label"], path, context="testing")

    dataset_input = MlflowClient(tracking_uri).get_run(run.info.run_id).inputs.dataset_inputs[0]
    assert dataset_input.dataset.digest == file_md5(path)
    assert dataset_input.dataset.name == "test"
    assert {tag.key: tag.value for tag in dataset_input.tags}["mlflow.data.context"] == "testing"
