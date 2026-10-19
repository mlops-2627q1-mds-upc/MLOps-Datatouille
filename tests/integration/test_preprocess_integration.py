"""
Integration tests for the text preprocessing workflow.
"""

import pandas as pd

from src.data import cleaning
from src.data import preprocess as pp


def test_preprocess_dataframe_pipeline():
    """Test of integrating deduplication, count features, and text cleaning."""
    raw = pd.DataFrame(
        {
            "text": [
                "WIN FREE prize!!! http://spam.com",
                "win free prize http://spam.com",
                "Hey team, meeting at 2?",
            ],
            "label": ["spam", "spam", "ham"],
        },
        index=[10, 20, 30],
    )

    result = pp.remove_duplicates(raw, "text")
    result = pp.add_count_features(result, "text")
    result = pp.add_clean_text(result, "text")
    result = pp.remove_duplicates(result, "clean_text")

    assert len(result) == 2
    assert list(result.index) == [0, 1]
    assert set(pp.COUNT_FUNCTIONS).issubset(result.columns)
    assert result.loc[0, "clean_text"] == "win free prize xxurl"
    assert result.loc[0, "n_urls"] == 1
    assert result["label"].tolist() == ["spam", "ham"]


def test_cleaning_and_tfidf_pipeline():
    """Test of integrating DataFrame cleaning, count features, and TF-IDF vectorization."""
    raw = pd.DataFrame(
        {
            "text": [
                "WIN cash prize $500 🎉 at www.spam.com",
                "Team meeting agenda tomorrow.",
                "Claim cash prize now!",
            ],
            "label": ["spam", "ham", "spam"],
        }
    )

    featured = pp.add_count_features(raw, "text")
    cleaned = pp.add_clean_text(featured, "text")
    vectorizer = pp.make_tfidf(max_features=20)
    matrix = vectorizer.fit_transform(cleaned["clean_text"])
    vocabulary = set(vectorizer.get_feature_names_out())

    assert cleaned.loc[0, "clean_text"] == "win cash prize xxmoney 🎉 xxurl"
    assert cleaned.loc[0, "n_urls"] == 1
    assert cleaned.loc[0, "n_emojis"] == 1
    assert {"xxmoney", "xxurl", "🎉"} <= vocabulary
    assert matrix.shape[0] == len(raw)
    assert matrix.getnnz(axis=1).tolist() == [6, 4, 3]
    assert raw.columns.tolist() == ["text", "label"]


def test_dataframe_cleaning_with_emoji_conversion_and_custom_output():
    """Test of integrating DataFrame cleaning with emoji conversion and a custom output column."""
    raw = pd.DataFrame(
        {
            "message": ["WIN FREE prize!!! 🎉", "Team meeting agenda."],
            "label": ["spam", "ham"],
        }
    )

    result = pp.add_clean_text(
        raw,
        "message",
        output_column="normalized_text",
        emojis_as_text=True,
    )

    assert result["normalized_text"].tolist() == [
        "win free prize parti popper",
        "team meet agenda",
    ]
    assert "normalized_text" not in raw.columns
    assert result["label"].tolist() == ["spam", "ham"]


def test_phase_one_anonymizes_before_counting_and_cleaning():
    """Test of integrating contact-detail anonymization into the complete phase-one flow."""
    original = "Write to john.doe@gmail.com or phone +34 612 345 678."
    raw = pd.DataFrame(
        {
            "text": [original, "The meeting starts tomorrow."],
            "label": ["spam", "not_spam"],
        }
    )

    result = cleaning.preprocess_dataframe(
        raw,
        text_col="text",
        label_col="label",
        anonymize=True,
    )

    assert result.loc[0, "text"].split() == [
        "Write",
        "to",
        "xxemail",
        "or",
        "phone",
        "xxphone",
        ".",
    ]
    assert {"xxemail", "xxphone"} <= set(result.loc[0, "clean_text"].split())
    assert result.loc[0, "n_uppercase"] == 1
    assert result.loc[0, "n_punctuation"] == 1
    assert not result["text"].str.contains(r"@|612", regex=True).any()
    assert raw.loc[0, "text"] == original
