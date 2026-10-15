"""Phase 1 of the pipeline: raw dataset -> cleaned dataset.

Steps:
    1. Optionally anonymize email addresses and phone numbers in the text.
    2. Remove exact duplicates on the text.
    3. Compute count features on the text.
    4. Clean the text (emojis are preserved by default).
    5. Remove duplicates on the cleaned text.

Usage (from the root of the repo):
    python -m src.data.phase_1_cleaning \
        --input data/raw/daisy_dataset_spam_detection.csv --output data/processed/clean.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data import preprocess as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_dataframe(df, text_col, label_col, emojis_as_text=False, anonymize=False):
    """Optionally anonymize, then clean and engineer features for the dataset."""
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"Columns {missing} not found. Columns in CSV: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    logger.info("Initial rows: %d", len(df))

    if anonymize:
        logger.info("Anonymizing email addresses and phone numbers")
        df[text_col] = df[text_col].apply(pp.anonymize_text)

    df = pp.remove_duplicates(df, text_col)
    logger.info("After removing exact duplicates: %d", len(df))

    df = pp.add_count_features(df, text_col)
    df = pp.add_clean_text(df, text_col, output_column="clean_text", emojis_as_text=emojis_as_text)

    df = df[df["clean_text"].str.len() > 0]
    df = pp.remove_duplicates(df, "clean_text")
    logger.info("After removing duplicates on cleaned text: %d", len(df))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Cleans and optionally anonymizes the raw dataset."
    )
    parser.add_argument("--input", required=True, help="Raw CSV")
    parser.add_argument("--output", required=True, help="Cleaned CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--emojis-as-text",
        type=lambda value: value.lower() == "true",
        default=False,
        help="Convert emojis to words instead of keeping them (true/false)",
    )
    parser.add_argument(
        "--anonymize",
        type=lambda value: value.lower() == "true",
        default=False,
        help="Replace email addresses and phone numbers with placeholders (true/false)",
    )
    args = parser.parse_args()

    logger.info("Reading %s", args.input)
    df = pd.read_csv(args.input, encoding=args.encoding)
    df = preprocess_dataframe(
        df,
        args.text_col,
        args.label_col,
        emojis_as_text=args.emojis_as_text,
        anonymize=args.anonymize,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved %s %s", output, df.shape)


if __name__ == "__main__":
    main()
