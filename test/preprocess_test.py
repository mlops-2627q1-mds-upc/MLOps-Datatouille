"""
Unit test of the src/data/preprocess.py module.
Execute from the root of the project:
    pytest -v
"""

import pandas as pd
import pytest

from src.data import preprocess as pp


def test_lower_case():
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
    assert pp.remove_punctuation(text) == expected


def test_remove_punctuation_keeps_emojis():
    assert pp.remove_punctuation("Win now!!! 🎉🔥") == "Win now 🎉🔥"


def test_remove_numbers():
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
    assert pp.remove_whitespace(text) == expected


def test_remove_html():
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
    assert pp.remove_whitespace(pp.replace_money(text)) == expected


def test_convert_emojis():
    result = pp.convert_emojis("go 🚀")
    assert "🚀" not in result
    assert "rocket" in result


def test_separate_emojis():
    assert pp.remove_whitespace(pp.separate_emojis("hello😀😀")) == "hello 😀 😀"


def test_remove_stopwords():
    assert pp.remove_stopwords("the prize is for you and me") == "prize"


def test_remove_stopwords_custom_list():
    assert pp.remove_stopwords("hello there friend", stopwords={"there"}) == "hello friend"


def test_stem_text():
    assert pp.stem_text("running cats playing") == "run cat play"


def test_stem_text_keeps_special_tokens():
    assert pp.stem_text("xxurl xxmoney running") == "xxurl xxmoney run"


def test_count_characters():
    assert pp.count_characters("hello!") == 6
    assert pp.count_characters("") == 0


def test_count_words():
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
    assert pp.count_sentences(text) == expected


def test_count_uppercase():
    assert pp.count_uppercase("CLAIM your Prize") == 6


def test_count_urls():
    assert pp.count_urls("see http://a.com and www.b.com") == 2
    assert pp.count_urls("nothing") == 0


def test_count_emojis():
    assert pp.count_emojis("free 🎉🎉 gift 🎁") == 3
    assert pp.count_emojis("no emojis") == 0


def test_count_punctuation():
    assert pp.count_punctuation("WIN!!! now, ok?") == 5


def test_count_punctuation_ignores_emojis():
    assert pp.count_punctuation("🎉🎉") == 0


def test_clean_text_keeps_emojis():
    result = pp.clean_text("WINNER!!! Claim your prize 🎉 at http://spam.com")
    assert result == "winner claim prize 🎉 xxurl"


def test_clean_text_emojis_as_text():
    result = pp.clean_text("🎉", emojis_as_text=True)
    assert "🎉" not in result
    assert "popper" in result
    assert "partypopp" not in result


def test_clean_text_money():
    assert pp.clean_text("You won $1000!") == "won xxmoney"


def test_clean_text_makes_near_duplicates_equal():
    assert pp.clean_text("FREE prize!!!") == pp.clean_text("free prize")


def test_remove_duplicates(sample_df):
    result = pp.remove_duplicates(sample_df, "text")
    assert len(result) == len(sample_df) - 1
    assert result["text"].is_unique
    assert list(result.index) == list(range(len(result)))


def test_add_count_features(sample_df):
    result = pp.add_count_features(sample_df, "text")
    for col in pp.COUNT_FUNCTIONS:
        assert col in result.columns
    assert result.loc[0, "n_emojis"] == 0
    assert result.loc[0, "n_urls"] == 1
    assert "n_words" not in sample_df.columns


def test_add_clean_text_and_second_dedup(sample_df):
    df = pp.remove_duplicates(sample_df, "text")
    df = pp.add_clean_text(df, "text")
    assert "clean_text" in df.columns
    deduped = pp.remove_duplicates(df, "clean_text")
    assert len(deduped) == len(df) - 1


def test_tfidf_keeps_emojis_and_shape():
    texts = pd.Series(["win prize 🎉", "hello friend", "win 🎉 🎉"])
    vec = pp.make_tfidf(max_features=10).fit(texts)
    assert "🎉" in vec.get_feature_names_out()
    assert vec.transform(texts).shape == (3, len(vec.get_feature_names_out()))


def test_tfidf_max_features():
    texts = pd.Series(["a b c d e", "a b c", "a"])
    vec = pp.make_tfidf(max_features=2).fit(texts)
    assert len(vec.get_feature_names_out()) == 2


def test_tfidf_unknown_words_are_zero():
    vec = pp.make_tfidf().fit(pd.Series(["win prize"]))
    assert vec.transform(pd.Series(["totally new words"])).sum() == 0
