"""
Unit test of the src/data/split_data.py module.
Execute from the root of the project:
    pytest -v
"""

import pandas as pd

from src.data.split_data import split_data


def test_split_data(sample_df):
    train, test = split_data(sample_df, "label", test_size=0.2, random_state=0)
    assert len(train) + len(test) == len(sample_df)
    assert len(test) == 2
    assert set(test["label"]) == {"spam", "ham"}


def test_split_data_is_reproducible(sample_df):
    a, _ = split_data(sample_df, "label", random_state=1)
    b, _ = split_data(sample_df, "label", random_state=1)
    pd.testing.assert_frame_equal(a, b)
