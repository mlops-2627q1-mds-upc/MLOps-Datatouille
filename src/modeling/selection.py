"""
Selection stage: choice of the best trained model by a cross-validation metric.

The selection writes a pointer to the best model (name, metric, value and path) instead of
copying it, so each model is stored only once. Test metrics are not used, so the test set
stays an unbiased estimate of the selected model.

The list of models is read from ``train.models`` in params.yaml (via ``dvc.api``), since DVC
cannot interpolate lists into stage commands.

Usage (from the root of the repo):
    python -m src.modeling.selection --metric cv_f1_macro \
        --metrics-dir reports/metrics/train --models-dir models \
        --output reports/best_model.json
"""

import argparse
import json
import logging
from pathlib import Path

import dvc.api

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_EXTENSION = ".joblib"


def load_cv_metrics(metrics_dir: Path, models: list[str]) -> dict[str, dict[str, float]]:
    """
    Load the cross-validation metrics written by the training stage.

    Args:
        metrics_dir: Directory holding one ``<model>.json`` file per model.
        models: Names of the models to load, in the order given in params.yaml.

    Returns:
        Model name to its metrics, in the order of ``models``.

    Raises:
        FileNotFoundError: If the metrics file of a model is missing.
    """
    return {
        model: json.loads((metrics_dir / f"{model}.json").read_text(encoding="utf-8"))
        for model in models
    }


def select_best_model(
    metrics_by_model: dict[str, dict[str, float]], metric: str
) -> tuple[str, float]:
    """
    Select the model with the highest value of ``metric``.

    Ties are broken by the order of ``metrics_by_model``, so the result is deterministic.

    Args:
        metrics_by_model: Model name to its metrics.
        metric: Metric to maximize (higher is better).

    Returns:
        Tuple ``(model_name, metric_value)`` of the best model.

    Raises:
        ValueError: If there are no models, or a model lacks ``metric``.
    """
    if not metrics_by_model:
        raise ValueError("No models to select from.")
    missing = [model for model, metrics in metrics_by_model.items() if metric not in metrics]
    if missing:
        raise ValueError(f"Metric '{metric}' not found for models: {missing}")
    best = max(metrics_by_model, key=lambda model: metrics_by_model[model][metric])
    return best, metrics_by_model[best][metric]


def build_selection(
    model: str, metric: str, value: float, models_dir: Path
) -> dict[str, str | float]:
    """
    Build the pointer to the selected model.

    Args:
        model: Name of the selected model.
        metric: Metric used for the selection.
        value: Value of ``metric`` for the selected model.
        models_dir: Directory holding the trained models.

    Returns:
        Selection record with the model name, metric, value and model path.
    """
    return {
        "model": model,
        "metric": metric,
        "value": value,
        "model_path": (models_dir / f"{model}{MODEL_EXTENSION}").as_posix(),
    }


def main() -> None:
    """Select the best model by a CV metric and write a pointer to it."""
    parser = argparse.ArgumentParser(description="Selects the best model by a CV metric.")
    parser.add_argument("--metric", required=True, help="CV metric to maximize")
    parser.add_argument("--metrics-dir", required=True, help="Directory of CV metrics (.json)")
    parser.add_argument("--models-dir", required=True, help="Directory of trained models")
    parser.add_argument("--output", required=True, help="Output path of the selection (.json)")
    args = parser.parse_args()

    models = dvc.api.params_show(stages="select")["train"]["models"]
    metrics_by_model = load_cv_metrics(Path(args.metrics_dir), models)
    model, value = select_best_model(metrics_by_model, args.metric)
    logger.info("Best model by %s: %s (%s)", args.metric, model, value)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selection = build_selection(model, args.metric, value, Path(args.models_dir))
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
