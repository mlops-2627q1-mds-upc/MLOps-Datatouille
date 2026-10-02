"""Tests unitarios de src/preprocessing.py.

Ejecutar desde la raíz del proyecto:
    pytest -v
"""

import pandas as pd
import pytest

from src import preprocess as pp


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------
def test_lower_case():
    assert pp.lower_case("CLAIM Your PRIZE") == "claim your prize"


def test_upper_case():
    assert pp.upper_case("claim prize") == "CLAIM PRIZE"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world!!!", "Hello world"),
        ("¿Qué tal? ¡Bien!", "Qué tal Bien"),
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
        ("  hola   que   tal  ", "hola que tal"),
        ("a\t\tb\n\nc", "a b c"),
        ("   ", ""),
    ],
)
def test_remove_whitespace(text, expected):
    assert pp.remove_whitespace(text) == expected


def test_remove_html():
    assert pp.remove_html("<p>Hola <b>mundo</b></p><br/>") == "Hola mundo"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("visit http://spam.com now", "visit [URL] now"),
        ("go to https://a.b/c?d=1 and www.test.org", "go to [URL] and [URL]"),
        ("no links here", "no links here"),
    ],
)
def test_replace_urls(text, expected):
    assert pp.replace_urls(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("win 500$ now", "win [DINERO] now"),
        ("win $100 now", "win [DINERO] now"),
        ("only 10.000 € today", "only [DINERO] today"),
        ("get 50USD free", "get [DINERO] free"),
        ("pay EUR 20,50 please", "pay [DINERO] please"),
        ("call 0800 123", "call 0800 123"),  # números sin moneda no se tocan
        ("EUROPE 2024", "EUROPE 2024"),  # 'EUR' dentro de una palabra no cuenta
    ],
)
def test_replace_money(text, expected):
    assert pp.replace_money(text) == expected


def test_convert_emojis():
    result = pp.convert_emojis("go 🚀")
    assert "🚀" not in result
    assert "rocket" in result


def test_separate_emojis():
    assert pp.remove_whitespace(pp.separate_emojis("hola😀😀")) == "hola 😀 😀"


def test_remove_stopwords():
    assert pp.remove_stopwords("the prize is for you and me") == "prize"


def test_remove_stopwords_custom_list():
    assert pp.remove_stopwords("hola que tal", stopwords={"que"}) == "hola tal"


def test_stem_text():
    assert pp.stem_text("running cats playing") == "run cat play"


# ---------------------------------------------------------------------------
# Conteos
# ---------------------------------------------------------------------------
def test_count_characters():
    assert pp.count_characters("hola!") == 5
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


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def test_clean_text_keeps_emojis():
    result = pp.clean_text("WINNER!!! Claim your prize 🎉 at http://spam.com")
    assert result == "winner claim prize 🎉 url"


def test_clean_text_emojis_as_text():
    result = pp.clean_text("Party 🎉", emojis_as_text=True)
    assert "🎉" not in result
    assert result.startswith("parti")


def test_clean_text_money():
    assert pp.clean_text("You won $1000!") == "won dinero"


def test_clean_text_makes_near_duplicates_equal():
    assert pp.clean_text("FREE prize!!!") == pp.clean_text("free prize")


# ---------------------------------------------------------------------------
# Funciones sobre DataFrames
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "text": [
                "WIN a FREE prize!!! 🎉 http://spam.com",
                "Hey, are we meeting today?",
                "WIN a FREE prize!!! 🎉 http://spam.com",
                "win a free prize 🎉 http://spam.com",
                "See you later",
                "Claim your $500 reward now!",
                "Lunch at 2?",
                "URGENT call now",
                "ok thanks",
                "Congrats, you won!!",
            ],
            "label": ["spam", "ham", "spam", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
        }
    )


def test_remove_duplicates(sample_df):
    result = pp.remove_duplicates(sample_df, "text")
    assert len(result) == len(sample_df) - 1
    assert result["text"].is_unique
    assert list(result.index) == list(range(len(result)))


def test_add_count_features(sample_df):
    result = pp.add_count_features(sample_df, "text")
    for col in pp.COUNT_FUNCTIONS:
        assert col in result.columns
    assert result.loc[0, "n_emojis"] == 1
    assert result.loc[0, "n_urls"] == 1
    assert "n_words" not in sample_df.columns  # no modifica el original


def test_add_clean_text_and_second_dedup(sample_df):
    df = pp.remove_duplicates(sample_df, "text")
    df = pp.add_clean_text(df, "text")
    assert "clean_text" in df.columns
    deduped = pp.remove_duplicates(df, "clean_text")
    # 'WIN a FREE prize!!!...' y 'win a free prize...' quedan iguales tras limpiar
    assert len(deduped) == len(df) - 1


def test_split_data(sample_df):
    train, test = pp.split_data(sample_df, "label", test_size=0.2, random_state=0)
    assert len(train) + len(test) == len(sample_df)
    assert len(test) == 2
    assert set(test["label"]) == {"spam", "ham"}  # estratificado


def test_split_data_is_reproducible(sample_df):
    a, _ = pp.split_data(sample_df, "label", random_state=1)
    b, _ = pp.split_data(sample_df, "label", random_state=1)
    pd.testing.assert_frame_equal(a, b)


def test_tfidf_keeps_emojis_and_shape():
    texts = pd.Series(["win prize 🎉", "hello friend", "win 🎉 🎉"])
    vec = pp.fit_tfidf(texts, max_features=10)
    assert "🎉" in vec.get_feature_names_out()

    features = pp.transform_tfidf(vec, texts)
    assert features.shape == (3, len(vec.get_feature_names_out()))
    assert all(c.startswith("tfidf_") for c in features.columns)


def test_tfidf_max_features():
    texts = pd.Series(["a b c d e", "a b c", "a"])
    vec = pp.fit_tfidf(texts, max_features=2)
    assert len(vec.get_feature_names_out()) == 2


def test_tfidf_unknown_words_are_zero():
    vec = pp.fit_tfidf(pd.Series(["win prize"]))
    features = pp.transform_tfidf(vec, pd.Series(["totally new words"]))
    assert features.to_numpy().sum() == 0