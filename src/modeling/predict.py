"""
Inference with a trained spam classifier on raw messages.

The trained pipeline expects the features produced by the cleaning stage, not raw text. This
module reproduces that stage for new messages, in the same order as ``src.data.cleaning``:

1. Anonymize email addresses and phone numbers.
2. Compute the count features on the anonymized text.
3. Clean the text (``clean_text`` column).

The module has no dependency on the API, so it can also be used from scripts or notebooks.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import preprocess as pp

RAW_TEXT_COL = "text"

Label = Literal["spam", "not_spam"]
SPAM_LABEL: Label = "spam"
NOT_SPAM_LABEL: Label = "not_spam"


@dataclass(frozen=True)
class Prediction:
    """
    Prediction of the classifier for one message.

    Attributes:
        text: Message after anonymization, as seen by the model.
        label: Predicted label, ``spam`` or ``not_spam``.
        confidence: Probability of the predicted label, in [0, 1].
        spam_probability: Probability of the ``spam`` class, in [0, 1].
    """

    text: str
    label: Label
    confidence: float
    spam_probability: float


def build_features(texts: Sequence[str], feature_names: Sequence[str]) -> pd.DataFrame:
    """
    Turn anonymized messages into the feature table expected by the trained pipeline.

    Args:
        texts: Messages already anonymized with ``anonymize_text``.
        feature_names: Input columns of the trained pipeline, in training order
            (``pipeline.feature_names_in_``).

    Returns:
        One row per message with the columns of ``feature_names``.

    Raises:
        KeyError: If the pipeline expects a column that the preprocessing does not produce.
    """
    frame = pd.DataFrame({RAW_TEXT_COL: list(texts)})
    frame = pp.add_count_features(frame, RAW_TEXT_COL)
    frame = pp.add_clean_text(frame, RAW_TEXT_COL)
    return frame[list(feature_names)]


def predict_messages(
    pipeline: Pipeline, texts: Sequence[str], threshold: float
) -> list[Prediction]:
    """
    Classify raw messages as spam or not spam.

    Email addresses and phone numbers are anonymized before inference. A message is labelled as
    spam when its spam probability is greater than or equal to ``threshold``.

    Args:
        pipeline: Fitted pipeline with ``predict_proba`` and a ``spam`` class.
        texts: Raw messages.
        threshold: Spam probability from which a message is labelled as spam, in [0, 1].

    Returns:
        One prediction per message, in the order of ``texts``.

    Raises:
        ValueError: If ``texts`` is empty or ``threshold`` is outside [0, 1].
    """
    if not texts:
        raise ValueError("At least one message is required.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}.")

    anonymized = [pp.anonymize_text(text) for text in texts]
    features = build_features(anonymized, pipeline.feature_names_in_)
    spam_index = list(pipeline.classes_).index(SPAM_LABEL)
    spam_probabilities = pipeline.predict_proba(features)[:, spam_index]

    predictions = []
    for text, probability in zip(anonymized, spam_probabilities, strict=True):
        spam_probability = float(probability)
        is_spam = spam_probability >= threshold
        predictions.append(
            Prediction(
                text=text,
                label=SPAM_LABEL if is_spam else NOT_SPAM_LABEL,
                confidence=spam_probability if is_spam else 1.0 - spam_probability,
                spam_probability=spam_probability,
            )
        )
    return predictions
