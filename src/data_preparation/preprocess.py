import re
import string
import unicodedata
import emoji                            # pip install emoji
from nltk.stem import PorterStemmer     # pip install nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 0. PATTERNS.
# ---------------------------------------------------------------------------

URL_PATTERN = r"https?://\S+|www\.\S+"
MONEY_PATTERN = (
    r"(?:\$|€|£|\b(?:USD|EUR|GBP)\b)\s*\d+(?:[.,]\d+)*"     # $100, € 10.000, USD 50
    r"|\d+(?:[.,]\d+)*\s*(?:\$|€|£|(?:USD|EUR|GBP)\b)"      # 500$, 10.000 €, 50USD
)
SENTENCE_END_PATTERN = r"[.!?]+"

URL_TOKEN = "xxurl"
MONEY_TOKEN = "xxmoney"
SPECIAL_TOKENS = {URL_TOKEN, MONEY_TOKEN}

_stemmer = PorterStemmer()

# ---------------------------------------------------------------------------
# 1. Cleaning text.
# ---------------------------------------------------------------------------
def lower_case(text):
    """Convert text into lowercase."""
    return text.lower()


def _is_punctuation(char):
    """True if the character is a punctuation mark (ASCII or Unicode)."""
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def remove_punctuation(text):
    """Remove punctuation marks from the text, keeping emojis."""
    return "".join(ch for ch in text if not _is_punctuation(ch))


def remove_numbers(text):
    """Delete the numeric digits from the text."""
    return re.sub(r"\d+", "", text)


def remove_whitespace(text):
    """Delete spaces at the beginning/end and reduce multiple spaces to one."""
    return re.sub(r"\s+", " ", text).strip()


def remove_html(text):
    """Delete HTML tags (like <p>, <br>, etc.)."""
    return re.sub(r"<[^>]+>", "", text)


def replace_urls(text):
    """Replace web links with the placeholder token URL_TOKEN."""
    return re.sub(URL_PATTERN, f" {URL_TOKEN} ", text)


def replace_money(text):
    """Replace money amounts (500$, 10.000 €, 50USD, $100) with MONEY_TOKEN."""
    return re.sub(MONEY_PATTERN, f" {MONEY_TOKEN} ", text)


def convert_emojis(text):
    """Convert each emoji to its English name, with words separated by spaces."""
    return emoji.replace_emoji(
        text, replace=lambda chars, data: " " + data["en"].strip(":").replace("_", " ") + " "
    )


def separate_emojis(text):
    """Add spaces around each emoji so that it becomes its own token."""
    return emoji.replace_emoji(text, replace=lambda chars, data: f" {chars} ")


def remove_stopwords(text, stopwords=ENGLISH_STOP_WORDS):
    """Delete the stop words (English by default). Expects lowercase text."""
    return " ".join(word for word in text.split() if word not in stopwords)


def stem_text(text):
    """Apply Porter stemming to each word, leaving the special tokens untouched."""
    return " ".join(w if w in SPECIAL_TOKENS else _stemmer.stem(w) for w in text.split())


# ---------------------------------------------------------------------------
# 2. Counts on the original text
# ---------------------------------------------------------------------------

def count_characters(text):
    """Number of characters."""
    return len(text)


def count_words(text):
    """Number of words (separated by spaces)."""
    return len(text.split())


def count_sentences(text):
    """Number of sentences (separated by '.', '!' or '?')."""
    return len([s for s in re.split(SENTENCE_END_PATTERN, text) if s.strip()])


def count_uppercase(text):
    """Number of uppercase letters."""
    return sum(1 for ch in text if ch.isupper())


def count_urls(text):
    """Number of links / urls."""
    return len(re.findall(URL_PATTERN, text))


def count_emojis(text):
    """Number of emojis."""
    return emoji.emoji_count(text)


def count_punctuation(text):
    """Number of punctuation marks (emojis are not counted)."""
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
def clean_text(text, emojis_as_text=False):
    """Apply all transformations with information loss.

    Order: html -> urls -> money -> emojis -> lowercase -> punctuation ->
    numbers -> stop words -> stemming -> spaces.

    Args:
        text: original text.
        emojis_as_text: if True, emojis are converted to words (🚀 -> rocket);
            if False (default), they are kept as emojis.
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

def remove_duplicates(df, column):
    """Remove rows with the same value in `column` (keep the first)."""
    return df.drop_duplicates(subset=column, keep="first").reset_index(drop=True)


def add_count_features(df, text_column):
    """Add a column for each count calculated over `text_column`."""
    df = df.copy()
    for name, func in COUNT_FUNCTIONS.items():
        df[name] = df[text_column].apply(func)
    return df


def add_clean_text(df, text_column, output_column="clean_text", emojis_as_text=False):
    """Add the column with the cleaned text."""
    df = df.copy()
    df[output_column] = df[text_column].apply(clean_text, emojis_as_text=emojis_as_text)
    return df


def split_data(df, label_column, test_size=0.2, random_state=42):
    """Train/test split stratified by the label."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_tfidf(max_features=3000):
    """Return an unfitted TF-IDF vectorizer for already-cleaned text.

    token_pattern=r"\\S+" is used because the default sklearn pattern would discard
    emojis and one-letter words.
    """
    return TfidfVectorizer(max_features=max_features, token_pattern=r"\S+")
