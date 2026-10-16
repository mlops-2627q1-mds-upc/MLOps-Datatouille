from pathlib import Path
from unittest.mock import MagicMock, patch

from src.download_dataset import download_dataset


def test_download_dataset_writes_csv_and_returns_path(tmp_path: Path) -> None:
    """
    Merge dataset splits and write them to the raw CSV path.

    Args:
        tmp_path: Pytest temporary directory for the output CSV.
    """
    output_path = tmp_path / "nested" / "raw.csv"
    fake_split = MagicMock()
    merged = MagicMock()
    splits = {"train": fake_split}

    with (
        patch("src.download_dataset.load_dataset", return_value=splits) as mock_load,
        patch("src.download_dataset.concatenate_datasets", return_value=merged) as mock_concat,
    ):
        result = download_dataset(dataset_id="org/fake-dataset", output_path=output_path)

    mock_load.assert_called_once_with("org/fake-dataset")
    mock_concat.assert_called_once_with([fake_split])
    merged.to_csv.assert_called_once_with(str(output_path))
    assert result == output_path
    assert output_path.parent.is_dir()
