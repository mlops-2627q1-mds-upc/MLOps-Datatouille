"""
Unit tests for the controlled changes of messages in src/data/perturb.py.
"""

import random

import pandas as pd
import pytest

from src.data import preprocess as pp
from src.data.perturb import (
    append_text,
    obfuscate_urls,
    perturb_dataframe,
    perturb_text,
    swap_adjacent_letters,
)


@pytest.fixture
def cleaned_df() -> pd.DataFrame:
    """
    Small dataset with the same columns as the output of the cleaning stage.
    """
    df = pd.DataFrame(
        {
            "text": [
                "WIN a FREE prize now at http://spam.com today",
                "Hey, are we still meeting for lunch tomorrow?",
                "Claim your reward at www.reward.net before midnight",
                "The quarterly report is attached for your review",
            ],
            "label": ["spam", "not_spam", "spam", "not_spam"],
        }
    )
    df = pp.add_count_features(df, "text")
    return pp.add_clean_text(df, "text", output_column="clean_text")


def test_obfuscate_urls_hides_links() -> None:
    """URLs are obfuscated so the cleaning stage no longer recognizes them."""
    text = "Visit http://win.com or www.prize.net now"
    obfuscated = obfuscate_urls(text)
    assert obfuscated == "Visit hxxp://win[.]com or www[.]prize[.]net now"
    assert pp.count_urls(obfuscated) == 0


def test_swap_adjacent_letters_keeps_the_same_letters() -> None:
    """A typo swaps two adjacent letters, so the letters and the length are preserved."""
    word = swap_adjacent_letters("meeting", random.Random(0))
    assert word != "meeting"
    assert sorted(word) == sorted("meeting")


def test_swap_adjacent_letters_keeps_words_without_adjacent_letters() -> None:
    """Words without two adjacent letters are returned unchanged."""
    assert swap_adjacent_letters("a1b2", random.Random(0)) == "a1b2"


def test_perturb_text_preserves_whitespace() -> None:
    """The number of words and the spacing between them are preserved."""
    text = "one  two three"
    perturbed = perturb_text(text, random.Random(0), case_change_rate=1.0, typo_rate=0.0)
    assert perturbed.split(" ") != text.split(" ")
    assert len(perturbed.split(" ")) == len(text.split(" "))


def test_perturb_dataframe_is_reproducible(cleaned_df: pd.DataFrame) -> None:
    """
    The same seed always produces the same perturbed dataset.
    """
    first = perturb_dataframe(cleaned_df, seed=42)
    second = perturb_dataframe(cleaned_df, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_perturb_dataframe_keeps_rows_labels_and_columns(cleaned_df: pd.DataFrame) -> None:
    """
    Perturbation changes the text but not the rows, the labels, or the columns.
    """
    perturbed = perturb_dataframe(cleaned_df, seed=42)
    assert len(perturbed) == len(cleaned_df)
    assert list(perturbed.columns) == list(cleaned_df.columns)
    assert perturbed["label"].tolist() == cleaned_df["label"].tolist()
    assert "http://" not in " ".join(perturbed["text"])


def test_perturb_dataframe_recomputes_derived_columns(cleaned_df: pd.DataFrame) -> None:
    """
    Count features and normalized text are recomputed from the perturbed text.
    """
    perturbed = perturb_dataframe(cleaned_df, seed=42)
    assert perturbed["n_urls"].sum() == 0
    assert cleaned_df["n_urls"].sum() == 2
    assert "xxurl" not in " ".join(perturbed["clean_text"])


def test_perturb_dataframe_does_not_modify_the_input(cleaned_df: pd.DataFrame) -> None:
    """
    The original dataset is left untouched.
    """
    original = cleaned_df.copy()
    perturb_dataframe(cleaned_df, seed=42)
    pd.testing.assert_frame_equal(cleaned_df, original)


def test_append_text_adds_the_suffix_and_recomputes(cleaned_df: pd.DataFrame) -> None:
    """
    The suffix is appended to every message and the derived columns are updated.
    """
    extended = append_text(cleaned_df, "Click here to claim your free prize")
    assert extended["text"].str.endswith("Click here to claim your free prize").all()
    assert (extended["n_words"] > cleaned_df["n_words"]).all()
    assert extended["clean_text"].str.contains("prize").all()
