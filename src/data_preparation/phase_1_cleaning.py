""" First step to clean the raw dataset. It is a command line script that reads a CSV file and outputs a cleaned CSV file.

STEPS:
    1. Remove duplicates (text exacte)
    2. Count features sobre el text original
    3. Clean the text (the emojis are preserved)
    4. Delete duplicates (text net)

usage (from the root of the repo):
    python -m src.preprocess_dataset --input data/raw/dataset.csv --output data/processed/clean.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_preparation import preprocess as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_dataframe(df, text_col, label_col, emojis_as_text=False):
    """Apply all the transformations to a DataFrame and return it."""
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"Columns {missing} not found. Columns in CSV: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    logger.info("Files inicials: %d", len(df))

    df = pp.remove_duplicates(df, text_col)
    logger.info("After removing exact duplicates: %d", len(df))

    df = pp.add_count_features(df, text_col)
    df = pp.add_clean_text(df, text_col, output_column="clean_text", emojis_as_text=emojis_as_text)

    df = df[df["clean_text"].str.len() > 0]
    df = pp.remove_duplicates(df, "clean_text")
    logger.info("After removing duplicates on cleaned text: %d", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocessa el dataset raw.")
    parser.add_argument("--input", required=True, help="CSV raw")
    parser.add_argument("--output", required=True, help="CSV preprocessat")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--emojis-as-text", action="store_true",
                        help="Converteix els emojis a paraules en lloc de conservar-los")
    args = parser.parse_args()

    logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input, encoding=args.encoding)
    df = preprocess_dataframe(df, args.text_col, args.label_col, args.emojis_as_text)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved %s %s", output, df.shape)


if __name__ == "__main__":
    main()