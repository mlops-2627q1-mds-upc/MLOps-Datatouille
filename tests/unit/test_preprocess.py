"""
Unit tests for the src/data/preprocess.py module.
Execute from the root of the project:
    pytest -v
"""

import pandas as pd
import pytest

from src.data import preprocess as pp


def test_lower_case():
    """Test of converting all text characters to lowercase."""
    assert pp.lower_case("CLAIM Your PRIZE") == "claim your prize"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world!!!", "Hello world"),
        ("¿How are you? ¡Fine!", "How are you Fine"),
        ("no punctuation", "no punctuation"),
        ("", ""),
    ],
)
def test_remove_punctuation(text, expected):
    """
    Test of removing punctuation while preserving the remaining content.
    """
    assert pp.remove_punctuation(text) == expected


def test_remove_punctuation_keeps_emojis():
    """Test of removing punctuation without removing emojis."""
    assert pp.remove_punctuation("Win now!!! 🎉🔥") == "Win now 🎉🔥"


def test_remove_numbers():
    """Test of deleting numeric digits from text."""
    assert pp.remove_numbers("call 0800 123 now") == "call   now"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  how   are   you  ", "how are you"),
        ("a\t\tb\n\nc", "a b c"),
        ("   ", ""),
    ],
)
def test_remove_whitespace(text, expected):
    """
    Test of trimming and collapsing whitespace, including tabs and newlines.
    """
    assert pp.remove_whitespace(text) == expected


def test_remove_html():
    """Test of stripping HTML tags while keeping their text content."""
    assert pp.remove_html("<p>Hello <b>world</b></p><br/>") == "Hello world"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("visit http://spam.com now", "visit xxurl now"),
        ("go to https://a.b/c?d=1 and www.test.org", "go to xxurl and xxurl"),
        ("no links here", "no links here"),
    ],
)
def test_replace_urls(text, expected):
    """
    Test of replacing supported web links with the URL placeholder.
    """
    assert pp.remove_whitespace(pp.replace_urls(text)) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("win 500$ now", "win xxmoney now"),
        ("win $100 now", "win xxmoney now"),
        ("only 10.000 € today", "only xxmoney today"),
        ("get 50USD free", "get xxmoney free"),
        ("pay EUR 20,50 please", "pay xxmoney please"),
        ("call 0800 123", "call 0800 123"),
        ("EUROPE 2024", "EUROPE 2024"),
    ],
)
def test_replace_money(text, expected):
    """
    Test of replacing currency formats without changing unrelated numbers.
    """
    assert pp.remove_whitespace(pp.replace_money(text)) == expected


def test_convert_emojis():
    """Test of converting emojis into their English names."""
    result = pp.convert_emojis("go 🚀")
    assert "🚀" not in result
    assert "rocket" in result


def test_separate_emojis():
    """Test of separating adjacent emojis into individual tokens."""
    assert pp.remove_whitespace(pp.separate_emojis("hello😀😀")) == "hello 😀 😀"


def test_remove_stopwords():
    """Test of removing common English stopwords using the default set."""
    assert pp.remove_stopwords("the prize is for you and me") == "prize"


def test_remove_stopwords_custom_list():
    """Test of removing stopwords supplied by the caller."""
    assert pp.remove_stopwords("hello there friend", stopwords={"there"}) == "hello friend"


def test_stem_text():
    """Test of reducing words to their Porter stem forms."""
    assert pp.stem_text("running cats playing") == "run cat play"


def test_stem_text_keeps_special_tokens():
    """Test of preserving URL and money placeholders during stemming."""
    assert pp.stem_text("xxurl xxmoney running") == "xxurl xxmoney run"


def test_count_characters():
    """Test of counting characters and returning zero for empty text."""
    assert pp.count_characters("hello!") == 6
    assert pp.count_characters("") == 0


def test_count_words():
    """Test of counting whitespace-separated words."""
    assert pp.count_words("win a  free prize") == 4
    assert pp.count_words("") == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hi. How are you? Great!", 3),
        ("WIN NOW!!!", 1),
        ("no end mark", 1),
        ("", 0),
    ],
)
def test_count_sentences(text, expected):
    """
    Test of counting sentence fragments, including edge cases.
    """
    assert pp.count_sentences(text) == expected


def test_count_uppercase():
    """Test of counting uppercase letters."""
    assert pp.count_uppercase("CLAIM your Prize") == 6


def test_count_urls():
    """Test of counting supported URLs and handling text without URLs."""
    assert pp.count_urls("see http://a.com and www.b.com") == 2
    assert pp.count_urls("nothing") == 0


def test_count_emojis():
    """Test of counting repeated emojis and handling text without emojis."""
    assert pp.count_emojis("free 🎉🎉 gift 🎁") == 3
    assert pp.count_emojis("no emojis") == 0


def test_count_punctuation():
    """Test of counting punctuation marks."""
    assert pp.count_punctuation("WIN!!! now, ok?") == 5


def test_count_punctuation_ignores_emojis():
    """Test of excluding emoji symbols from the punctuation count."""
    assert pp.count_punctuation("🎉🎉") == 0


def test_clean_text_keeps_emojis():
    """Test of full text cleaning while preserving emojis and replacing URLs."""
    result = pp.clean_text("WINNER!!! Claim your prize 🎉 at http://spam.com")
    assert result == "winner claim prize 🎉 xxurl"


def test_clean_text_emojis_as_text():
    """Test of converting emojis to names when text-mode handling is enabled."""
    result = pp.clean_text("🎉", emojis_as_text=True)
    assert "🎉" not in result
    assert "popper" in result
    assert "partypopp" not in result


def test_clean_text_money():
    """Test of preserving a money placeholder through full text cleaning."""
    assert pp.clean_text("You won $1000!") == "won xxmoney"


def test_clean_text_makes_near_duplicates_equal():
    """Test of normalizing equivalent messages to identical cleaned text."""
    assert pp.clean_text("FREE prize!!!") == pp.clean_text("free prize")


def test_remove_duplicates(sample_df):
    """
    Test of keeping the first duplicate row and resetting the index.
    """
    result = pp.remove_duplicates(sample_df, "text")
    assert len(result) == len(sample_df) - 1
    assert result["text"].is_unique
    assert list(result.index) == list(range(len(result)))


def test_add_count_features(sample_df):
    """
    Test of adding all count features without mutating the input DataFrame.
    """
    result = pp.add_count_features(sample_df, "text")
    for col in pp.COUNT_FUNCTIONS:
        assert col in result.columns
    assert result.loc[0, "n_emojis"] == 0
    assert result.loc[0, "n_urls"] == 1
    assert "n_words" not in sample_df.columns


def test_add_clean_text_and_second_dedup(sample_df):
    """
    Test of cleaning text and removing duplicates created by normalization.
    """
    df = pp.remove_duplicates(sample_df, "text")
    df = pp.add_clean_text(df, "text")
    assert "clean_text" in df.columns
    deduped = pp.remove_duplicates(df, "clean_text")
    assert len(deduped) == len(df) - 1


def test_tfidf_keeps_emojis_and_shape():
    """Test of fitting and transforming text while retaining emoji tokens."""
    texts = pd.Series(["win prize 🎉", "hello friend", "win 🎉 🎉"])
    vec = pp.make_tfidf(max_features=10).fit(texts)
    assert "🎉" in vec.get_feature_names_out()
    assert vec.transform(texts).shape == (3, len(vec.get_feature_names_out()))


def test_tfidf_max_features():
    """Test of limiting the vocabulary to the configured maximum feature count."""
    texts = pd.Series(["a b c d e", "a b c", "a"])
    vec = pp.make_tfidf(max_features=2).fit(texts)
    assert len(vec.get_feature_names_out()) == 2


def test_tfidf_unknown_words_are_zero():
    """Test of producing a zero vector for text with only unseen tokens."""
    vec = pp.make_tfidf().fit(pd.Series(["win prize"]))
    assert vec.transform(pd.Series(["totally new words"])).sum() == 0


def test_tfidf_rejects_empty_vocabulary():
    """Test of error handling when TF-IDF receives text with no tokens."""
    with pytest.raises(ValueError, match="empty vocabulary"):
        pp.make_tfidf().fit(pd.Series([""]))


def test_clean_text_rejects_non_string_input():
    """Test of error handling when text cleaning receives a non-string value."""
    with pytest.raises(TypeError):
        pp.clean_text(None)


def test_remove_stopwords_rejects_invalid_stopword_collection():
    """Test of error handling when stopwords is not an iterable collection."""
    with pytest.raises(TypeError, match="NoneType"):
        pp.remove_stopwords("hello", stopwords=None)


def test_remove_duplicates_rejects_missing_column():
    """Test of error handling when the deduplication column is missing."""
    with pytest.raises(KeyError, match="missing"):
        pp.remove_duplicates(pd.DataFrame({"text": ["offer"]}), "missing")


def test_add_count_features_rejects_missing_text_column():
    """Test of error handling when the source text column for counts is missing."""
    with pytest.raises(KeyError, match="missing"):
        pp.add_count_features(pd.DataFrame({"label": ["spam"]}), "missing")


def test_add_clean_text_rejects_missing_text_column():
    """Test of error handling when the source text column for cleaning is missing."""
    with pytest.raises(KeyError, match="missing"):
        pp.add_clean_text(pd.DataFrame({"label": ["spam"]}), "missing")
