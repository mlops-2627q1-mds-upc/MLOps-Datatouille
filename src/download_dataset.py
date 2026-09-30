from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from loguru import logger

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "raw.csv"
DATASET_ID = "diegovelilla/email-classification-dataset"


def download_dataset(
    dataset_id: str = DATASET_ID,
    output_path: Path = RAW_PATH,
) -> Path:
    """Download HF dataset and save it as a raw CSV.

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


if __name__ == "__main__":
    download_dataset()
