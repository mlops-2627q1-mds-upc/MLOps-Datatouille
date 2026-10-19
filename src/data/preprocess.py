from collections.abc import Collection
import re
import string
import unicodedata

import emoji  # pip install emoji
from nltk.stem import PorterStemmer  # pip install nltk
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# ---------------------------------------------------------------------------
# 0. PATTERNS.
# ---------------------------------------------------------------------------

URL_PATTERN = r"https?://\S+|www\.\S+"
MONEY_PATTERN = (
    r"(?:\$|€|£|\b(?:USD|EUR|GBP)\b)\s*\d+(?:[.,]\d+)*"  # $100, € 10.000, USD 50
    r"|\d+(?:[.,]\d+)*\s*(?:\$|€|£|(?:USD|EUR|GBP)\b)"  # 500$, 10.000 €, 50USD
)
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # john.doe@mail.com
PHONE_PATTERN = r"(?<!\w)\+\d{1,3}(?:[\s().-]{0,2}\d){7,13}(?!\w)"  # +34 612 345 678
SENTENCE_END_PATTERN = r"[.!?]+"

URL_TOKEN = "xxurl"
MONEY_TOKEN = "xxmoney"
EMAIL_TOKEN = "xxemail"
PHONE_TOKEN = "xxphone"
SPECIAL_TOKENS = {URL_TOKEN, MONEY_TOKEN, EMAIL_TOKEN, PHONE_TOKEN}

_stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# 1. Cleaning text.
# ---------------------------------------------------------------------------
def lower_case(text: str) -> str:
    """
    Convert the text to lowercase.

    Args:
        text: Input text.

    Returns:
        The lowercased text.
    """
    return text.lower()


def _is_punctuation(char: str) -> bool:
    """
    Check whether a character is a punctuation mark.

    Args:
        char: Character to test.

    Returns:
        True if the character is ASCII or Unicode punctuation.
    """
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks from the text, keeping emojis.

    Args:
        text: Input text.

    Returns:
        The text without punctuation marks.
    """
    return "".join(ch for ch in text if not _is_punctuation(ch))


def remove_numbers(text: str) -> str:
    """
    Delete the numeric digits from the text.

    Args:
        text: Input text.

    Returns:
        The text without digits.
    """
    return re.sub(r"\d+", "", text)


def remove_whitespace(text: str) -> str:
    """
    Trim the text and collapse runs of whitespace into single spaces.

    Args:
        text: Input text.

    Returns:
        The normalized text.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_html(text: str) -> str:
    """
    Delete HTML tags from the text, keeping their content.

    Args:
        text: Input text.

    Returns:
        The text without HTML tags.
    """
    return re.sub(r"<[^>]+>", "", text)


def replace_urls(text: str) -> str:
    """
    Replace web links with the placeholder token.

    Args:
        text: Input text.

    Returns:
        The text with links replaced by ``URL_TOKEN``.
    """
    return re.sub(URL_PATTERN, f" {URL_TOKEN} ", text)


def replace_money(text: str) -> str:
    """
    Replace money amounts with the placeholder token.

    Args:
        text: Input text.

    Returns:
        The text with money amounts replaced by ``MONEY_TOKEN``.
    """
    return re.sub(MONEY_PATTERN, f" {MONEY_TOKEN} ", text)


def anonymize_text(text: str) -> str:
    """
    Replace email addresses and phone numbers with placeholder tokens.

    Args:
        text: Input text.

    Returns:
        The text with contact details replaced by ``EMAIL_TOKEN`` and ``PHONE_TOKEN``.
    """
    text = re.sub(EMAIL_PATTERN, f" {EMAIL_TOKEN} ", text)
    return re.sub(PHONE_PATTERN, f" {PHONE_TOKEN} ", text)


def convert_emojis(text: str) -> str:
    """
    Convert each emoji to its English name.

    Args:
        text: Input text.

    Returns:
        The text with each emoji replaced by its name.
    """
    return emoji.replace_emoji(
        text, replace=lambda chars, data: " " + data["en"].strip(":").replace("_", " ") + " "
    )


def separate_emojis(text: str) -> str:
    """
    Add spaces around each emoji so that it becomes its own token.

    Args:
        text: Input text.

    Returns:
        The text with emojis isolated.
    """
    return emoji.replace_emoji(text, replace=lambda chars, data: f" {chars} ")


def remove_stopwords(text: str, stopwords: Collection[str] = ENGLISH_STOP_WORDS) -> str:
    """
    Delete the stop words from the text. Expects lowercase text.

    Args:
        text: Input text.
        stopwords: Words to remove. Defaults to the English stop-word set.

    Returns:
        The text without stop words.
    """
    return " ".join(word for word in text.split() if word not in stopwords)


def stem_text(text: str) -> str:
    """
    Apply Porter stemming to each word, leaving the special tokens untouched.

    Args:
        text: Input text.

    Returns:
        The stemmed text.
    """
    return " ".join(w if w in SPECIAL_TOKENS else _stemmer.stem(w) for w in text.split())


# ---------------------------------------------------------------------------
# 2. Counts on the original text
# ---------------------------------------------------------------------------


def count_characters(text: str) -> int:
    """
    Count the characters in the text.

    Args:
        text: Input text.

    Returns:
        Number of characters.
    """
    return len(text)


def count_words(text: str) -> int:
    """
    Count the words in the text.

    Args:
        text: Input text.

    Returns:
        Number of words, separated by spaces.
    """
    return len(text.split())


def count_sentences(text: str) -> int:
    """
    Count the sentences in the text.

    Args:
        text: Input text.

    Returns:
        Number of sentences, separated by '.', '!' or '?'.
    """
    return len([s for s in re.split(SENTENCE_END_PATTERN, text) if s.strip()])


def count_uppercase(text: str) -> int:
    """
    Count the uppercase letters in the text.

    Args:
        text: Input text.

    Returns:
        Number of uppercase letters.
    """
    return sum(1 for ch in text if ch.isupper())


def count_urls(text: str) -> int:
    """
    Count the web links in the text.

    Args:
        text: Input text.

    Returns:
        Number of links.
    """
    return len(re.findall(URL_PATTERN, text))


def count_emojis(text: str) -> int:
    """
    Count the emojis in the text.

    Args:
        text: Input text.

    Returns:
        Number of emojis.
    """
    return emoji.emoji_count(text)


def count_punctuation(text: str) -> int:
    """
    Count the punctuation marks in the text, excluding emojis.

    Args:
        text: Input text.

    Returns:
        Number of punctuation marks.
    """
    return sum(1 for ch in text if _is_punctuation(ch))


COUNT_FUNCTIONS = {
    "n_characters": count_characters,
    "n_words": count_words,
    "n_sentences": count_sentences,
    "n_uppercase": count_uppercase,
    "n_urls": count_urls,
    "n_emojis": count_emojis,
    "n_punctuation": count_punctuation,
}


# ---------------------------------------------------------------------------
# 3. Full cleaning pipeline
# ---------------------------------------------------------------------------
def clean_text(text: str, emojis_as_text: bool = False) -> str:
    """
    Apply all transformations with information loss.

    Order: html -> urls -> money -> emojis -> lowercase -> punctuation ->
    numbers -> stop words -> stemming -> spaces.

    Args:
        text: Original text.
        emojis_as_text: If True, emojis are converted to words (🚀 -> rocket);
            if False (default), they are kept as emojis.

    Returns:
        The cleaned text.
    """
    text = remove_html(text)
    text = replace_urls(text)
    text = replace_money(text)
    text = convert_emojis(text) if emojis_as_text else separate_emojis(text)
    text = lower_case(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text


# ---------------------------------------------------------------------------
# 4. DataFrame functions
# ---------------------------------------------------------------------------


def remove_duplicates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove rows with a duplicate value in ``column``, keeping the first.

    Args:
        df: Input DataFrame.
        column: Column whose duplicate values are removed.

    Returns:
        A new DataFrame without duplicates and with a reset index.
    """
    return df.drop_duplicates(subset=column, keep="first").reset_index(drop=True)


def add_count_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Add one count column per function in ``COUNT_FUNCTIONS``.

    Args:
        df: Input DataFrame.
        text_column: Column holding the text to count over.

    Returns:
        A copy of ``df`` with the ``n_*`` count columns added.
    """
    df = df.copy()
    for name, func in COUNT_FUNCTIONS.items():
        df[name] = df[text_column].apply(func)
    return df


def add_clean_text(
    df: pd.DataFrame,
    text_column: str,
    output_column: str = "clean_text",
    emojis_as_text: bool = False,
) -> pd.DataFrame:
    """
    Add a column with the cleaned text.

    Args:
        df: Input DataFrame.
        text_column: Column holding the text to clean.
        output_column: Name of the column that receives the cleaned text.
        emojis_as_text: If True, convert emojis to words instead of keeping them.

    Returns:
        A copy of ``df`` with the cleaned-text column added.
    """
    df = df.copy()
    df[output_column] = df[text_column].apply(clean_text, emojis_as_text=emojis_as_text)
    return df


def make_tfidf(max_features: int = 3000) -> TfidfVectorizer:
    """
    Return an unfitted TF-IDF vectorizer for already-cleaned text.

    token_pattern=r"\\S+" is used because the default sklearn pattern would discard
    emojis and one-letter words.

    Args:
        max_features: Maximum vocabulary size.

    Returns:
        The configured, unfitted vectorizer.
    """
    return TfidfVectorizer(max_features=max_features, token_pattern=r"\S+")
