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
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

PARENT_RUN_SUFFIX = "train"
EXP_NAME_TAG = "dvc.exp_name"
BASELINE_REV_TAG = "dvc.baseline_rev"
EXP_REF_TAG = "dvc.exp_ref"
RUN_ROLE_TAG = "run_role"
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


def build_run_name(prefix: str | None, suffix: str) -> str:
    """Build an MLflow run name prefixed by the DVC experiment name.

    Args:
        prefix: DVC experiment name, or None when not running under ``dvc exp run``.
        suffix: Run-specific part of the name (e.g. the model name).

    Returns:
        ``"<prefix>-<suffix>"``, or ``suffix`` alone when there is no prefix.
    """
    return f"{prefix}-{suffix}" if prefix else suffix


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
    tags = {
        EXP_NAME_TAG: exp_name,
        RUN_ROLE_TAG: role,
        NOTE_TAG: build_restore_note(exp_name, baseline_rev),
    }
    if baseline_rev:
        tags[BASELINE_REV_TAG] = baseline_rev
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
    filters = [f"tags.`{EXP_NAME_TAG}` = '{exp_name}'", f"tags.`{RUN_ROLE_TAG}` = 'parent'"]
    if baseline_rev:
        filters.append(f"tags.`{BASELINE_REV_TAG}` = '{baseline_rev}'")
    runs = client.search_runs(
        [experiment_id],
        filter_string=" and ".join(filters),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id

    run = client.create_run(
        experiment_id,
        run_name=build_run_name(exp_name, PARENT_RUN_SUFFIX),
        tags=build_exp_tags(exp_name, baseline_rev, "parent"),
    )
    return run.info.run_id


@contextmanager
def start_training_run(model_name: str, experiment_name: str) -> Iterator[Run]:
    """Start the MLflow run for one trained model.

    Under ``dvc exp run`` the run is a child of the experiment's parent run. Outside it
    (e.g. ``dvc repro``) a standalone run named after the model is started.

    Args:
        model_name: Name of the model being trained.
        experiment_name: MLflow experiment to log into (created if missing).

    Yields:
        The active MLflow run for the model.
    """
    exp_name, baseline_rev = get_dvc_exp_context()
    experiment = mlflow.set_experiment(experiment_name)
    if not exp_name:
        with mlflow.start_run(run_name=model_name) as run:
            yield run
        return

    parent_id = find_or_create_parent_run(
        MlflowClient(), experiment.experiment_id, exp_name, baseline_rev
    )
    tags = build_exp_tags(exp_name, baseline_rev, "child")
    with mlflow.start_run(run_id=parent_id), mlflow.start_run(
        run_name=build_run_name(exp_name, model_name), nested=True, tags=tags
    ) as run:
        yield run
