"""
Registration of the selected model in the MLflow Model Registry.

The ``select`` stage of the pipeline writes a pointer to the best model in
``reports/best_model.json``. This script publishes that model for serving:

1. It finds the latest MLflow run of the selected model (training and evaluation run).
2. It logs the trained pipeline to that run with the scikit-learn flavour.
3. It registers the logged model as a new version of the registered model.
4. It points the serving alias (``production`` by default) to the new version.

The API loads the model through this alias (``models:/<name>@<alias>``), so moving the alias
changes the served model without changing the API code. The script is run manually after a
pipeline run whose model should be served. It is not a DVC stage, since publishing a model is a
deliberate decision and not a reproducible output of the pipeline.

Usage (from the root of the repo, with the MLflow credentials in ``.env``):
    python -m src.modeling.register --selection reports/best_model.json
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.api.settings import MLFLOW_MODEL_ALIAS, MLFLOW_MODEL_NAME
from src.config import MLFLOW_EXPERIMENT_NAME
from src.modeling.tracking import MODEL_TAG, RUN_ROLE_TAG

logger = logging.getLogger(__name__)

LOGGED_MODEL_NAME = "model"
TRIAL_ROLE = "trial"


def load_selection(path: Path) -> dict[str, str | float]:
    """
    Read the pointer to the selected model written by the ``select`` stage.

    Args:
        path: JSON file with the keys ``model``, ``metric``, ``value`` and ``model_path``.

    Returns:
        The selection record.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required key is missing.
    """
    selection = json.loads(path.read_text(encoding="utf-8"))
    missing = {"model", "metric", "value", "model_path"} - set(selection)
    if missing:
        raise KeyError(f"Keys {sorted(missing)} not found in {path}")
    return selection


def find_model_run(client: MlflowClient, experiment_name: str, model: str) -> str:
    """
    Return the latest MLflow run holding the training and test metrics of a model.

    Trial runs of the hyperparameter search are excluded, since they hold a single
    cross-validation result and not the final model.

    Args:
        client: MLflow client.
        experiment_name: MLflow experiment holding the runs.
        model: Name of the model, as used in training (e.g. ``naive_bayes``).

    Returns:
        Run ID of the latest matching run.

    Raises:
        LookupError: If the experiment or the run does not exist.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"MLflow experiment '{experiment_name}' does not exist")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`{MODEL_TAG}` = '{model}' and tags.`{RUN_ROLE_TAG}` != '{TRIAL_ROLE}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"No MLflow run found for model '{model}' in '{experiment_name}'")
    return runs[0].info.run_id


def register_model(
    selection: dict[str, str | float], run_id: str, registered_name: str, alias: str
) -> str:
    """
    Log the selected pipeline to its run, register it and point the alias to it.

    Args:
        selection: Pointer to the selected model, as returned by ``load_selection``.
        run_id: MLflow run of the selected model.
        registered_name: Name of the model in the Model Registry.
        alias: Alias that the API loads (e.g. ``production``).

    Returns:
        The new model version number.
    """
    pipeline = joblib.load(str(selection["model_path"]))
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=LOGGED_MODEL_NAME,
            registered_model_name=registered_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    version = str(model_info.registered_model_version)

    client = MlflowClient()
    client.update_model_version(
        registered_name,
        version,
        description=(
            f"Model '{selection['model']}' selected by {selection['metric']} = "
            f"{selection['value']}."
        ),
    )
    client.set_registered_model_alias(registered_name, alias, version)
    return version


def main() -> None:
    """Register the selected model and point the serving alias to the new version."""
    parser = argparse.ArgumentParser(description="Registers the selected model in MLflow.")
    parser.add_argument("--selection", required=True, help="Selection pointer (.json)")
    parser.add_argument("--experiment", default=MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--registered-name", default=MLFLOW_MODEL_NAME)
    parser.add_argument("--alias", default=MLFLOW_MODEL_ALIAS)
    args = parser.parse_args()

    selection = load_selection(Path(args.selection))
    run_id = find_model_run(MlflowClient(), args.experiment, str(selection["model"]))
    logger.info("Registering model '%s' from MLflow run %s", selection["model"], run_id)

    version = register_model(selection, run_id, args.registered_name, args.alias)
    logger.info(
        "Registered %s version %s with alias '%s'", args.registered_name, version, args.alias
    )


if __name__ == "__main__":
    main()
