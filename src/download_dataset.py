"""
Download the raw spam dataset from Hugging Face.

Usage (from the root of the repo):
    python -m src.download_dataset \
        --dataset-id diegovelilla/email-classification-dataset \
        --output data/raw/raw.csv
"""

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from loguru import logger

DATASET_ID = "diegovelilla/email-classification-dataset"
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "raw.csv"


def download_dataset(
    dataset_id: str = DATASET_ID,
    output_path: Path = RAW_PATH,
) -> Path:
    """
    Download HF dataset and save it as a raw CSV.

    Args:
        dataset_id: Hugging Face dataset identifier.
        output_path: Destination CSV file path.

    Returns:
        Path to the written CSV file.
    """
    ds = load_dataset(dataset_id)
    dataset = concatenate_datasets(list(ds.values()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(str(output_path))
    logger.info(f"Saved raw dataset to: {output_path}")
    return output_path


def main() -> None:
    """Parse CLI args and download the dataset."""
    parser = argparse.ArgumentParser(description="Download the raw dataset.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    download_dataset(dataset_id=args.dataset_id, output_path=args.output)


if __name__ == "__main__":
    main()
