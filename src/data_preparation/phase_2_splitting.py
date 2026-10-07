"""Phase 2 of the pipeline: cleaned dataset -> train / test.

The split is stratified by the label and reproducible (random_state).

Usage (from the root of the repo):
    python -m src.data_preparation.phase_2_splitting --input data/processed/clean.csv \
        --train data/dataset/train.csv --test data/dataset/test.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_preparation import preprocess as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train / test split.")
    parser.add_argument("--input", required=True, help="Preprocessed CSV")
    parser.add_argument("--train", required=True, help="Output train CSV")
    parser.add_argument("--test", required=True, help="Output test CSV")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input)

    train_df, test_df = pp.split_data(df, args.label_col, args.test_size, args.random_state)

    for path, split in ((args.train, train_df), (args.test, test_df)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        logger.info("Saved %s %s", path, split.shape)


if __name__ == "__main__":
    main()