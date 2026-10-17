"""Stratified train/test split of the cleaned dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test sets, stratified by the label.

    Args:
        df: Dataset to split.
        label_column: Name of the column used to stratify the split.
        test_size: Proportion of rows assigned to the test set, between 0 and 1.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        Tuple with the train and test DataFrames, each with a reset index.

    Raises:
        KeyError: If ``label_column`` is not a column of ``df``.
        ValueError: If ``test_size`` is out of range, or if a class has too few rows to be
            present in both sets.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
