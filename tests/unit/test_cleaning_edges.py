import pytest
import pandas as pd
from src.data.cleaning import preprocess_dataframe

def test_preprocess_dataframe_missing_columns():
    df = pd.DataFrame({"wrong_column": ["test text"]})
    
    with pytest.raises(KeyError, match="not found"):
        preprocess_dataframe(df, text_col="text", label_col="label")