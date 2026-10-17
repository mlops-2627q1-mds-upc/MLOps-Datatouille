"""MLflow tracking for the training stage.

One DVC experiment (``dvc exp run``) maps to one MLflow parent run, and each model trained by
the ``train`` foreach stage logs a nested child run under it. Every ``train@<model>`` stage is
a separate process, so the parent run is looked up by tags and reused instead of recreated.
Parent and child runs record the DVC experiment name, git ref and restore commands, so a
run can be reproduced from the MLflow UI with ``dvc exp pull`` + ``dvc exp apply``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import mlflow
from mlflow.data.pandas_dataset import from_pandas
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import pandas as pd

EXP_NAME_TAG = "dvc.exp_name"
BASELINE_REV_TAG = "dvc.baseline_rev"
EXP_REF_TAG = "dvc.exp_ref"
RUN_ROLE_TAG = "run_role"
MODEL_TAG = "model"
NOTE_TAG = "mlflow.note.content"
EXPS_NAMESPACE = "refs/exps"
GIT_REMOTE = "origin"
HASH_CHUNK_SIZE = 1 << 20


def get_dvc_exp_context() -> tuple[str | None, str | None]:
    """Read the DVC experiment name and baseline commit set by ``dvc exp run``.

    Returns:
        Tuple ``(exp_name, baseline_rev)``. Each is None outside ``dvc exp run``.
    """
    return os.getenv("DVC_EXP_NAME") or None, os.getenv("DVC_EXP_BASELINE_REV") or None


def build_exp_ref(exp_name: str, baseline_rev: str) -> str:
    """Build the git ref under which DVC stores an experiment.

    Args:
        exp_name: DVC experiment name.
        baseline_rev: Full SHA of the git commit the experiment is based on.

    Returns:
        Ref in DVC's format ``refs/exps/<sha[:2]>/<sha[2:]>/<exp_name>``.
    """
    return f"{EXPS_NAMESPACE}/{baseline_rev[:2]}/{baseline_rev[2:]}/{exp_name}"


def build_restore_note(exp_name: str, baseline_rev: str | None) -> str:
    """Build the MLflow run description explaining how to restore the DVC experiment.

    Args:
        exp_name: DVC experiment name.
        baseline_rev: Git commit the experiment is based on, if known.

    Returns:
        Markdown text with the commands to restore the experiment in a workspace.
    """
    lines = [
        f"DVC experiment `{exp_name}`. To restore its code, params and outputs:",
        "",
        "```bash",
        f"dvc exp pull {GIT_REMOTE} {exp_name}",
        f"dvc exp apply {exp_name}",
        "```",
    ]
    if baseline_rev:
        lines += ["", f"Git ref: `{build_exp_ref(exp_name, baseline_rev)}`"]
    return "\n".join(lines)


def build_exp_tags(exp_name: str, baseline_rev: str | None, role: str) -> dict[str, str]:
    """Build the MLflow tags linking a run to its DVC experiment.

    Args:
        exp_name: DVC experiment name.
        baseline_rev: Git commit the experiment is based on, if known.
        role: Run role, ``"parent"`` or ``"child"``.

    Returns:
        Tag name to value.
    """
    tags = build_lookup_tags(exp_name, baseline_rev, role)
    tags[NOTE_TAG] = build_restore_note(exp_name, baseline_rev)
    if baseline_rev:
        tags[EXP_REF_TAG] = build_exp_ref(exp_name, baseline_rev)
    return tags


def file_md5(path: Path) -> str:
    """Compute the MD5 hash of a file, used to record the dataset version.

    Args:
        path: File to hash.

    Returns:
        Hex digest of the file content.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.md5()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_dataset(x: pd.DataFrame, y: pd.Series, path: Path, context: str) -> None:
    """Log a dataset as an input of the active MLflow run.

    The digest is the MD5 of the whole file (MLflow's default only hashes the first rows), so it
    matches the hash DVC records in ``dvc.lock``.

    Args:
        x: Features of the dataset.
        y: Target of the dataset; its name is logged as the target column.
        path: CSV file the dataset was loaded from.
        context: Role of the dataset in the run, e.g. ``"training"`` or ``"testing"``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    dataset = from_pandas(
        x.assign(**{y.name: y}),
        source=LocalArtifactDatasetSource(str(path)),
        targets=y.name,
        name=Path(path).stem,
        digest=file_md5(path),
    )
    mlflow.log_input(dataset, context=context)


def find_latest_run(client: MlflowClient, experiment_id: str, tags: dict[str, str]) -> str | None:
    """Return the most recent run whose tags match all of ``tags``.

    Args:
        client: MLflow client.
        experiment_id: MLflow experiment holding the runs.
        tags: Tag name to expected value.

    Returns:
        Run ID of the latest matching run, or None if there is none.
    """
    filter_string = " and ".join(f"tags.`{key}` = '{value}'" for key, value in tags.items())
    runs = client.search_runs(
        [experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def build_lookup_tags(exp_name: str, baseline_rev: str | None, role: str) -> dict[str, str]:
    """Build the tags identifying a run of a DVC experiment, used to look it up.

    Args:
        exp_name: DVC experiment name.
        baseline_rev: Git commit the DVC experiment is based on, if known.
        role: Run role, ``"parent"`` or ``"child"``.

    Returns:
        Tag name to value.
    """
    tags = {EXP_NAME_TAG: exp_name, RUN_ROLE_TAG: role}
    if baseline_rev:
        tags[BASELINE_REV_TAG] = baseline_rev
    return tags


def find_or_create_parent_run(
    client: MlflowClient,
    experiment_id: str,
    exp_name: str,
    baseline_rev: str | None,
) -> str:
    """Return the parent run of a DVC experiment, creating it on first use.

    Args:
        client: MLflow client.
        experiment_id: MLflow experiment holding the runs.
        exp_name: DVC experiment name.
        baseline_rev: Git commit the DVC experiment is based on, if known.

    Returns:
        Run ID of the parent run.
    """
    run_id = find_latest_run(
        client, experiment_id, build_lookup_tags(exp_name, baseline_rev, "parent")
    )
    if run_id:
        return run_id

    run = client.create_run(
        experiment_id,
        run_name=exp_name,
        tags=build_exp_tags(exp_name, baseline_rev, "parent"),
    )
    return run.info.run_id


@contextmanager
def start_model_run(model_name: str, experiment_name: str, resume: bool = False) -> Iterator[Run]:
    """Start the MLflow run for one model.

    Under ``dvc exp run`` the run is a child of the experiment's parent run. With ``resume``,
    the latest child run of this model in the experiment is reopened (e.g. to add test metrics
    to the training run), and a new one is created only if none exists. Outside ``dvc exp run``
    (e.g. ``dvc repro``) a new standalone run named after the model is started.

    Args:
        model_name: Name of the model.
        experiment_name: MLflow experiment to log into (created if missing).
        resume: If True, reopen the model's existing child run instead of creating one.

    Yields:
        The active MLflow run for the model.
    """
    exp_name, baseline_rev = get_dvc_exp_context()
    experiment = mlflow.set_experiment(experiment_name)
    if not exp_name:
        with mlflow.start_run(run_name=model_name) as run:
            yield run
        return

    client = MlflowClient()
    parent_id = find_or_create_parent_run(client, experiment.experiment_id, exp_name, baseline_rev)
    child_tags = build_lookup_tags(exp_name, baseline_rev, "child") | {MODEL_TAG: model_name}
    child_id = find_latest_run(client, experiment.experiment_id, child_tags) if resume else None
    if child_id:
        child_args = {"run_id": child_id}
    else:
        child_args = {
            "run_name": model_name,
            "tags": build_exp_tags(exp_name, baseline_rev, "child") | {MODEL_TAG: model_name},
        }
    with mlflow.start_run(run_id=parent_id), mlflow.start_run(nested=True, **child_args) as run:
        yield run


def start_trial_run(model_name: str, trial_number: int) -> Run:
    """Start the MLflow run of one hyperparameter-search trial, nested in the active run.

    Must be called while the model's run (from ``start_model_run``) is active. The run gets
    the ``trial`` role, so ``start_model_run(resume=True)`` never mistakes it for the model run.

    Args:
        model_name: Name of the model being tuned.
        trial_number: Optuna trial number.

    Returns:
        The started MLflow run, usable as a context manager.
    """
    return mlflow.start_run(
        run_name=f"trial-{trial_number}",
        nested=True,
        tags={RUN_ROLE_TAG: "trial", MODEL_TAG: model_name},
    )
