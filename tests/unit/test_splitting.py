"""
Unit tests for the split stage entrypoint.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data.splitting import main as splitting_main


def test_splitting_main_writes_stratified_splits(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """The split stage writes both CSVs with every input row and both labels in each split."""
    input_path = tmp_path / "clean.csv"
    train_path = tmp_path / "out" / "train.csv"
    test_path = tmp_path / "out" / "test.csv"
    sample_df.to_csv(input_path, index=False)
    argv = [
        "script",
        "--input",
        str(input_path),
        "--train",
        str(train_path),
        "--test",
        str(test_path),
        "--test-size",
        "0.4",
    ]

    with patch("sys.argv", argv):
        splitting_main()

    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    assert len(train) + len(test) == len(sample_df)
    assert set(train["label"]) == set(test["label"]) == {"spam", "ham"}