"""Controlled changes of messages, used to test the behaviour of the trained models.

Two kinds of changes are provided:
    - Perturbations that should not change the label: random case changes, typos made by
      swapping two adjacent letters, and obfuscated URLs (``http`` -> ``hxxp``, ``.`` -> ``[.]``).
    - The addition of a fixed text to every message, used to check that a typical spam phrase
      does not lower the spam score of a model.

After changing the original text, the count features and the normalized text are recomputed
with the cleaning functions of the pipeline, so the models receive the modified messages exactly
as they would in production.
"""

import random
import re

import pandas as pd

from src.data import preprocess as pp

CASE_CHANGE_RATE = 0.2
TYPO_RATE = 0.1
MIN_TYPO_LETTERS = 4
DEFAULT_SEED = 42

_URL_REGEX = re.compile(pp.URL_PATTERN)


def obfuscate_urls(text: str) -> str:
    """Obfuscate every URL so that it is no longer recognized as a link.

    Args:
        text: Original text.

    Returns:
        Text where ``http`` becomes ``hxxp`` and every dot inside a URL becomes ``[.]``.
    """

    def _obfuscate(match: re.Match) -> str:
        return match.group(0).replace("http", "hxxp", 1).replace(".", "[.]")

    return _URL_REGEX.sub(_obfuscate, text)


def swap_adjacent_letters(word: str, rng: random.Random) -> str:
    """Simulate a typo by swapping two adjacent letters of a word.

    Args:
        word: Word to modify. Punctuation attached to it is kept in place.
        rng: Random generator that chooses the position of the swap.

    Returns:
        The word with two adjacent letters swapped, or the same word if it has no two
        adjacent letters.
    """
    positions = [i for i in range(len(word) - 1) if word[i].isalpha() and word[i + 1].isalpha()]
    if not positions:
        return word
    i = rng.choice(positions)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def perturb_text(
    text: str,
    rng: random.Random,
    case_change_rate: float = CASE_CHANGE_RATE,
    typo_rate: float = TYPO_RATE,
) -> str:
    """Apply case changes, typos, and URL obfuscation to a message.

    Args:
        text: Original message.
        rng: Random generator, shared across messages so the result is reproducible.
        case_change_rate: Probability of changing the case of each word.
        typo_rate: Probability of introducing a typo in each word with enough letters.

    Returns:
        The perturbed message. Whitespace between words is preserved.
    """
    words = obfuscate_urls(text).split(" ")
    perturbed = []
    for word in words:
        if "[.]" not in word:
            if rng.random() < case_change_rate:
                word = word.upper() if rng.random() < 0.5 else word.lower()
            if (
                sum(char.isalpha() for char in word) >= MIN_TYPO_LETTERS
                and rng.random() < typo_rate
            ):
                word = swap_adjacent_letters(word, rng)
        perturbed.append(word)
    return " ".join(perturbed)


def rebuild_features(
    df: pd.DataFrame, text_col: str, emojis_as_text: bool = False
) -> pd.DataFrame:
    """Recompute the count features and the normalized text from the original text.

    Args:
        df: Dataset whose original text has been modified.
        text_col: Name of the original text column.
        emojis_as_text: Same option used by the cleaning stage of the pipeline.

    Returns:
        A copy of ``df`` with updated count features and ``clean_text`` column.
    """
    df = pp.add_count_features(df, text_col)
    return pp.add_clean_text(
        df, text_col, output_column="clean_text", emojis_as_text=emojis_as_text
    )


def perturb_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    seed: int = DEFAULT_SEED,
    emojis_as_text: bool = False,
) -> pd.DataFrame:
    """Create a perturbed copy of a cleaned dataset, keeping its labels and number of rows.

    Args:
        df: Cleaned dataset with the original text, the count features, and ``clean_text``.
        text_col: Name of the original text column.
        seed: Random seed, so the same dataset always produces the same perturbed copy.
        emojis_as_text: Same option used by the cleaning stage of the pipeline.

    Returns:
        A perturbed copy of ``df``. Duplicates are not removed.
    """
    rng = random.Random(seed)
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(perturb_text, rng=rng)
    return rebuild_features(df, text_col, emojis_as_text)


def append_text(
    df: pd.DataFrame,
    suffix: str,
    text_col: str = "text",
    emojis_as_text: bool = False,
) -> pd.DataFrame:
    """Append the same text to every message of a cleaned dataset.

    Args:
        df: Cleaned dataset with the original text, the count features, and ``clean_text``.
        suffix: Text appended at the end of every message.
        text_col: Name of the original text column.
        emojis_as_text: Same option used by the cleaning stage of the pipeline.

    Returns:
        A copy of ``df`` with the suffix appended and the derived columns recomputed.
    """
    df = df.copy()
    df[text_col] = df[text_col].astype(str) + " " + suffix
    return rebuild_features(df, text_col, emojis_as_text)
