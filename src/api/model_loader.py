"""
Loading of the production spam classifier from the MLflow Model Registry.

The model is resolved through an alias (``models:/<name>@<alias>``), so the API always serves
the version that the alias points to when the API starts. The alias is first resolved to a
version number, and that exact version is then loaded. This avoids loading a different version
than the one reported if the alias is moved during startup.

The tracking URI and the credentials are read by MLflow from the environment
(``MLFLOW_TRACKING_URI``, ``MLFLOW_TRACKING_USERNAME``, ``MLFLOW_TRACKING_PASSWORD``), which
``src.config`` loads from ``.env``.
"""

from dataclasses import dataclass, field

import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.modeling.predict import SPAM_LABEL

REPORTED_METRIC_PREFIXES = ("cv_", "test_")


class ModelLoadError(RuntimeError):
    """Raised when the production model cannot be loaded or cannot serve predictions."""


@dataclass(frozen=True)
class LoadedModel:
    """
    Production model loaded in memory, with the metadata reported by the API.

    Attributes:
        pipeline: Fitted scikit-learn pipeline.
        name: Name of the model in the Model Registry.
        alias: Alias through which the model was resolved.
        version: Registered model version.
        run_id: MLflow run that produced the model, if known.
        metrics: Cross-validation and test metrics recorded in that run.
    """

    pipeline: Pipeline
    name: str
    alias: str
    version: str
    run_id: str | None
    metrics: dict[str, float] = field(default_factory=dict)


def check_model(pipeline: Pipeline) -> None:
    """
    Check that a pipeline can produce spam probabilities.

    Args:
        pipeline: Loaded scikit-learn pipeline.

    Raises:
        ModelLoadError: If the pipeline has no ``predict_proba``, no ``spam`` class, or no
            recorded input columns.
    """
    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(
            f"{type(pipeline).__name__} has no predict_proba, so no confidence can be returned"
        )
    if SPAM_LABEL not in list(getattr(pipeline, "classes_", [])):
        raise ModelLoadError(f"The model has no '{SPAM_LABEL}' class")
    if not hasattr(pipeline, "feature_names_in_"):
        raise ModelLoadError("The model does not record its input columns (feature_names_in_)")


def load_run_metrics(client: MlflowClient, run_id: str | None) -> dict[str, float]:
    """
    Read the cross-validation and test metrics of the run that produced a model.

    Args:
        client: MLflow client.
        run_id: MLflow run of the model, or None if unknown.

    Returns:
        Metric name to value, restricted to the ``cv_`` and ``test_`` metrics.
    """
    if run_id is None:
        return {}
    metrics = client.get_run(run_id).data.metrics
    return {
        name: value
        for name, value in sorted(metrics.items())
        if name.startswith(REPORTED_METRIC_PREFIXES)
    }


def load_production_model(name: str, alias: str) -> LoadedModel:
    """
    Load the registered model version that an alias points to.

    Args:
        name: Name of the model in the Model Registry.
        alias: Alias of the version to serve (e.g. ``production``).

    Returns:
        The loaded model and its metadata.

    Raises:
        ModelLoadError: If the alias cannot be resolved, the model cannot be downloaded, or it
            cannot produce spam probabilities.
    """
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(name, alias)
        pipeline = mlflow.sklearn.load_model(f"models:/{name}/{model_version.version}")
        metrics = load_run_metrics(client, model_version.run_id)
    except Exception as error:
        raise ModelLoadError(f"Cannot load model 'models:/{name}@{alias}': {error}") from error

    check_model(pipeline)
    return LoadedModel(
        pipeline=pipeline,
        name=name,
        alias=alias,
        version=str(model_version.version),
        run_id=model_version.run_id,
        metrics=metrics,
    )
