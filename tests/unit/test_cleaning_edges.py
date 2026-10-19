"""
Unit tests for the `preprocess_dataframe` function in `src.data.cleaning`.
"""

import pandas as pd
import pytest

from src.data.cleaning import preprocess_dataframe


def test_preprocess_dataframe_missing_columns():
    """Missing text or label columns raises KeyError."""
    df = pd.DataFrame({"wrong_column": ["test text"]})

    with pytest.raises(KeyError, match="not found"):
        preprocess_dataframe(df, text_col="text", label_col="label")
