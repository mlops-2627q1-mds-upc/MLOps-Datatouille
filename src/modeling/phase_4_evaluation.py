"""Phase 4 of the pipeline: evaluation of a trained model on the held-out test set.

Usage (from the root of the repo):
    python -m src.modeling.phase_4_evaluation --model-path models/logreg.joblib \
        --test data/dataset/test.csv
"""
import argparse
import logging

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.modeling.phase_3_training import load_xy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluates a trained model on the test set.")
    parser.add_argument("--model-path", required=True, help="Trained model (.joblib)")
    parser.add_argument("--test", required=True, help="Test CSV")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    X, y, _ = load_xy(args.test, args.label_col)
    pipeline = joblib.load(args.model_path)
    y_pred = pipeline.predict(X)

    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro")
    metrics = {
        "test_accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "test_precision_macro": round(float(precision), 4),
        "test_recall_macro": round(float(recall), 4),
        "test_f1_macro": round(float(f1), 4),
    }
    logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
