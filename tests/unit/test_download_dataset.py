"""
Unit tests for the dataset download stage.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import RAW_DATA_DIR
from src.data.download_dataset import RAW_PATH, download_dataset


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
        patch("src.data.download_dataset.load_dataset", return_value=splits) as mock_load,
        patch(
            "src.data.download_dataset.concatenate_datasets", return_value=merged
        ) as mock_concat,
    ):
        result = download_dataset(dataset_id="org/fake-dataset", output_path=output_path)

    mock_load.assert_called_once_with("org/fake-dataset")
    mock_concat.assert_called_once_with([fake_split])
    merged.to_csv.assert_called_once_with(str(output_path))
    assert result == output_path
    assert output_path.parent.is_dir()


def test_download_dataset_merges_every_split_in_order(tmp_path: Path) -> None:
    """
    Concatenate all splits in their declared order.

    Args:
        tmp_path: Pytest temporary directory for the output CSV.
    """
    splits = {"train": MagicMock(), "test": MagicMock(), "validation": MagicMock()}

    with (
        patch("src.data.download_dataset.load_dataset", return_value=splits),
        patch("src.data.download_dataset.concatenate_datasets") as mock_concat,
    ):
        download_dataset(dataset_id="org/fake-dataset", output_path=tmp_path / "raw.csv")

    mock_concat.assert_called_once_with([splits["train"], splits["test"], splits["validation"]])


def test_download_dataset_propagates_load_error(tmp_path: Path) -> None:
    """
    Report a loading failure without writing any CSV.

    Args:
        tmp_path: Pytest temporary directory for the output CSV.
    """
    output_path = tmp_path / "raw.csv"

    with (
        patch(
            "src.data.download_dataset.load_dataset",
            side_effect=ConnectionError("offline"),
        ),
        pytest.raises(ConnectionError, match="offline"),
    ):
        download_dataset(dataset_id="org/fake-dataset", output_path=output_path)

    assert not output_path.exists()


def test_default_raw_path_uses_config_data_dir() -> None:
    """The default output path lives in the configured raw data directory."""
    assert RAW_PATH == RAW_DATA_DIR / "raw.csv"
