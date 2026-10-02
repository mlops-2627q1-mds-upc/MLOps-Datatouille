import re
import string
import unicodedata
import emoji                            # pip install emoji
import pandas as pd
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

_stemmer = PorterStemmer()

# ---------------------------------------------------------------------------
# 1. Cleaning text.
# ---------------------------------------------------------------------------
def lower_case(text):
    """Convert text into lowercase."""
    return text.lower()


def upper_case(text):
    """Convert text into uppercase."""
    return text.upper()


def _is_punctuation(char):
    """True if the character is a punctuation mark (ASCII or Unicode)."""
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def remove_punctuation(text):
    """Elimina los signos de puntuación del texto, conservando los emojis."""
    return "".join(ch for ch in text if not _is_punctuation(ch))


def remove_numbers(text):
    """Elimina los dígitos numéricos del texto."""
    return re.sub(r"\d+", "", text)


def remove_whitespace(text):
    """Elimina espacios al inicio/final y reduce múltiples espacios a uno."""
    return re.sub(r"\s+", " ", text).strip()


def remove_html(text):
    """Elimina etiquetas HTML (como <p>, <br>, etc.)."""
    return re.sub(r"<[^>]+>", "", text)


def replace_urls(text):
    """Reemplaza enlaces web por el token estandarizado [URL]."""
    return re.sub(URL_PATTERN, "[URL]", text)


def replace_money(text):
    """Reemplaza cifras de dinero (500$, 10.000 €, 50USD, $100) por [DINERO]."""
    return re.sub(MONEY_PATTERN, "[DINERO]", text)


def convert_emojis(text):
    """Convierte emojis en texto descriptivo """
    return emoji.demojize(text, delimiters=(" ", " "))


def separate_emojis(text):
    """Pone espacios alrededor de cada emoji para que sea un token propio."""
    return emoji.replace_emoji(text, replace=lambda chars, data: f" {chars} ")


def remove_stopwords(text, stopwords=ENGLISH_STOP_WORDS):
    """Elimina las stop words (inglés por defecto). Espera texto en minúsculas."""
    return " ".join(word for word in text.split() if word not in stopwords)


def stem_text(text):
    """Aplica stemming (Porter) a cada palabra del texto."""
    return " ".join(_stemmer.stem(word) for word in text.split())


# ---------------------------------------------------------------------------
# 2. Conteos sobre el texto original
# ---------------------------------------------------------------------------

def count_characters(text):
    """Número de caracteres."""
    return len(text)


def count_words(text):
    """Número de palabras (separadas por espacios)."""
    return len(text.split())


def count_sentences(text):
    """Número de frases (separadas por '.', '!' o '?')."""
    return len([s for s in re.split(SENTENCE_END_PATTERN, text) if s.strip()])


def count_uppercase(text):
    """Número de letras mayúsculas."""
    return sum(1 for ch in text if ch.isupper())


def count_urls(text):
    """Número de enlaces / urls."""
    return len(re.findall(URL_PATTERN, text))


def count_emojis(text):
    """Número de emojis."""
    return emoji.emoji_count(text)


def count_punctuation(text):
    """Número de signos de puntuación (los emojis no cuentan)."""
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
# 3. Pipeline de limpieza completo
# ---------------------------------------------------------------------------
def clean_text(text, emojis_as_text=False):
    """Aplica todas las transformaciones con pérdida de información.

    Orden: html -> urls -> dinero -> emojis -> minúsculas -> puntuación ->
    números -> stop words -> stemming -> espacios.

    Args:
        text: texto original.
        emojis_as_text: si True, convierte los emojis a palabras
            (🚀 -> rocket); si False (defecto) se mantienen como emoji.
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
# 4. Funciones sobre DataFrames
# ---------------------------------------------------------------------------

def remove_duplicates(df, column):
    """Elimina filas con el mismo valor en `column` (se queda la primera)."""
    return df.drop_duplicates(subset=column, keep="first").reset_index(drop=True)


def add_count_features(df, text_column):
    """Añade una columna por cada conteo calculado sobre `text_column`."""
    df = df.copy()
    for name, func in COUNT_FUNCTIONS.items():
        df[name] = df[text_column].apply(func)
    return df


def add_clean_text(df, text_column, output_column="clean_text", emojis_as_text=False):
    """Añade la columna con el texto limpio."""
    df = df.copy()
    df[output_column] = df[text_column].apply(clean_text, emojis_as_text=emojis_as_text)
    return df


def split_data(df, label_column, test_size=0.2, random_state=42):
    """Train/test split estratificado por la etiqueta."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_tfidf(texts, max_features=3000):
    """Ajusta un TF-IDF sobre `texts` (solo train, para evitar data leakage).
    Se usa token_pattern=r"\\S+" porque el texto ya está limpio: el patrón
    por defecto de sklearn descartaría los emojis y palabras de 1 letra.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, token_pattern=r"\S+")
    vectorizer.fit(texts)
    return vectorizer


def transform_tfidf(vectorizer, texts, prefix="tfidf_"):
    """Transforma `texts` a un DataFrame con una columna por término."""
    matrix = vectorizer.transform(texts)
    columns = [f"{prefix}{term}" for term in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=columns)