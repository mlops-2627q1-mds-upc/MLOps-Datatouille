from sklearn.model_selection import train_test_split


def split_data(df, label_column, test_size=0.2, random_state=42):
    """Train/test split stratified by the label."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
