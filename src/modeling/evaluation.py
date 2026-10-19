"""
Phase 4 of the pipeline: evaluation of a trained model on the held-out test set.

Usage (from the root of the repo):
    python -m src.modeling.evaluation --model logreg \
        --model-path models/logreg.joblib --test data/dataset/test.csv \
        --metrics-out reports/metrics/test/logreg.json \
        --features-out reports/features/logreg.json

Test metrics and the most influential features are logged to the model's MLflow run from
training (see src.modeling.tracking).
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import MLFLOW_EXPERIMENT_NAME
from src.modeling.explain import top_features
from src.modeling.tracking import log_dataset, start_model_run
from src.modeling.train import load_xy, save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_test_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """
    Compute test-set classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Metric name (prefixed with ``test_``) to value, rounded to 4 decimals.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "test_precision_macro": round(float(precision), 4),
        "test_recall_macro": round(float(recall), 4),
        "test_f1_macro": round(float(f1), 4),
    }


def save_top_features(features: dict, path: Path) -> None:
    """
    Write the most influential features as JSON so DVC can version them.

    Args:
        features: Output of ``top_features``.
        path: Output JSON file. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Evaluate one trained model on the test set, log to MLflow and save its metrics."""
    parser = argparse.ArgumentParser(description="Evaluates a trained model on the test set.")
    parser.add_argument("--model", required=True, help="Model name, as used in training")
    parser.add_argument("--model-path", required=True, help="Trained model (.joblib)")
    parser.add_argument("--test", required=True, help="Test CSV")
    parser.add_argument("--metrics-out", required=True, help="Output path of test metrics (.json)")
    parser.add_argument(
        "--features-out", required=True, help="Output path of the top influential features (.json)"
    )
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    X, y, _ = load_xy(args.test, args.label_col)
    pipeline = joblib.load(args.model_path)
    metrics = compute_test_metrics(y, pipeline.predict(X))
    features_path = Path(args.features_out)
    save_top_features(top_features(pipeline), features_path)
    logger.info("Saved %s", features_path)

    with start_model_run(args.model, MLFLOW_EXPERIMENT_NAME, resume=True) as run:
        log_dataset(X, y, Path(args.test), context="testing")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(features_path), artifact_path="features")
        logger.info("Test metrics: %s (MLflow run %s)", metrics, run.info.run_id)

    save_metrics(metrics, Path(args.metrics_out))
    logger.info("Saved %s", args.metrics_out)


if __name__ == "__main__":
    main()
