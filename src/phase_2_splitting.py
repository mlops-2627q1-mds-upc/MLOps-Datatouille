"""Etapa 2 de la pipeline: dataset preprocessat -> train / test.

El split és estratificat per l'etiqueta i reproduïble (random_state).

Ús (des de l'arrel del projecte):
    python -m src.split_dataset --input data/processed/clean.csv \
        --train data/processed/train.csv --test data/processed/test.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import preprocess as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train / test split.")
    parser.add_argument("--input", required=True, help="CSV preprocessat")
    parser.add_argument("--train", required=True, help="CSV de train de sortida")
    parser.add_argument("--test", required=True, help="CSV de test de sortida")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logger.info("Llegint %s", args.input)
    df = pd.read_csv(args.input)

    train_df, test_df = pp.split_data(df, args.label_col, args.test_size, args.random_state)

    for path, split in ((args.train, train_df), (args.test, test_df)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        logger.info("Guardat %s %s", path, split.shape)


if __name__ == "__main__":
    main()