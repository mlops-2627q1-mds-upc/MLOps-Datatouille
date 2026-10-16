"""
Phase 1 of the pipeline: raw dataset -> cleaned dataset.

Steps:
    1. Optionally anonymize email addresses and phone numbers in the text.
    2. Remove exact duplicates on the text.
    3. Compute count features on the text.
    4. Clean the text (emojis are preserved by default).
    5. Remove duplicates on the cleaned text.

Usage (from the root of the repo):
    python -m src.data.cleaning \
        --input data/raw/raw.csv --output data/processed/clean.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data import preprocess as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    emojis_as_text: bool = False,
    anonymize: bool = False,
) -> pd.DataFrame:
    """
    Optionally anonymize, then clean and engineer features for the dataset.

    Args:
        df: Raw input dataframe with text and label columns.
        text_col: Name of the text column.
        label_col: Name of the label column.
        emojis_as_text: If True, convert emojis to words instead of keeping them.
        anonymize: If True, replace email addresses and phone numbers
            with placeholders.

    Returns:
        Cleaned dataframe with count features and a `clean_text` column.

    Raises:
        KeyError: If the text or label columns are missing.
    """
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


def main() -> None:
    """
    Clean the raw dataset and write the cleaned CSV.

    Args:
        None: Arguments are parsed from the command line.

    Returns:
        None.
    """
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
