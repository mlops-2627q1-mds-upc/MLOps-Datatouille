"""
Unit tests for the src/data/split_data.py module.

Execute from the root of the project:
    pytest -v
"""

import pandas as pd
import pytest

from src.data.split_data import split_data


def test_split_data(sample_df):
    """Test of splitting the data into stratified train and test sets."""
    train, test = split_data(sample_df, "label", test_size=0.2, random_state=0)
    assert len(train) + len(test) == len(sample_df)
    assert len(test) == 2
    assert set(test["label"]) == {"spam", "ham"}


def test_split_data_is_reproducible(sample_df):
    """Test of producing identical splits when the same random state is used."""
    a, _ = split_data(sample_df, "label", random_state=1)
    b, _ = split_data(sample_df, "label", random_state=1)
    pd.testing.assert_frame_equal(a, b)


def test_split_data_rejects_missing_label_column(sample_df):
    """Test of reporting an error when the requested label column is absent."""
    with pytest.raises(KeyError, match="missing_label"):
        split_data(sample_df, "missing_label")


def test_split_data_rejects_invalid_test_size(sample_df):
    """Test of reporting an error when test_size is outside its valid range."""
    with pytest.raises(ValueError, match="test_size"):
        split_data(sample_df, "label", test_size=1.5)


def test_split_data_rejects_label_with_only_one_sample():
    """Test of stratified splitting when a class has too few samples."""
    dataframe = pd.DataFrame(
        {
            "text": ["one", "two", "three"],
            "label": ["common", "common", "rare"],
        }
    )

    with pytest.raises(ValueError, match="least populated class"):
        split_data(dataframe, "label", test_size=0.33)
