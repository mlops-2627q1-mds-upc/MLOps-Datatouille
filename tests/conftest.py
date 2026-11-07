"""
Fixtures shared by all the test modules.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data.cleaning import preprocess_dataframe
from src.modeling.train import TEXT_COL, build_pipeline, get_count_columns


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "text": [
                "WIN a FREE prize!!! http://spam.com",
                "Hey, are we meeting today?",
                "WIN a FREE prize!!! http://spam.com",
                "win a free prize http://spam.com",
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


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """
    Point MLflow to a throwaway local store and clear DVC experiment env vars.

    Also runs the test inside ``tmp_path``, since MLflow writes artifacts of a local store
    to ``./mlruns``.
    """
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    monkeypatch.delenv("DVC_EXP_BASELINE_REV", raising=False)
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


SPAM_TRAINING_MESSAGES = [
    "WIN a FREE prize now!!! Claim your cash reward at http://win.example",
    "Exclusive discount: buy cheap pills today, limited offer 💊💰",
    "Congratulations, you are our lucky winner! Click http://claim.example",
    "URGENT: your account wins $1000, reply now to get the money",
    "Hottest deals of the year, amazing discount on luxury watches",
    "Get rich fast with this secret trick, earn cash from home",
    "Free iPhone giveaway!!! Only today, click the link to claim it",
    "Cheap loans approved instantly, no credit check, call +34 612 345 678",
    "You won a FREE cruise! Claim your prize before midnight 🎁",
    "Lose weight fast with miracle pills, exclusive offer for you",
    "Limited time offer: buy one get two free, click http://deal.example",
    "Earn money online, guaranteed income, join our amazing program",
]

NOT_SPAM_TRAINING_MESSAGES = [
    "Hi team, the project meeting has been moved to Monday at 10.",
    "Please review the dataset file I shared and send me your comments.",
    "Thanks for the research update, the results look consistent.",
    "The data pipeline finished correctly, the logs are in the report.",
    "Can we discuss the experiment design during tomorrow's call?",
    "I attached the draft of the paper, see the methods section.",
    "The server will be under maintenance on Friday afternoon.",
    "Write to ana@example.com if you cannot attend the seminar.",
    "Here are the slides from today's lecture on regression models.",
    "The code review is done, I left a few minor suggestions.",
    "Remember to update the documentation before the release.",
    "Lunch with the group at 2pm near the faculty?",
]


@pytest.fixture(scope="session")
def spam_pipeline() -> Pipeline:
    """
    Small spam classifier trained like the production model on a handful of known messages.

    It goes through the same anonymization, count features, text cleaning and pipeline as the
    DVC training stage, so it accepts the same input as the registered model.
    """
    raw = pd.DataFrame(
        {
            "text": SPAM_TRAINING_MESSAGES + NOT_SPAM_TRAINING_MESSAGES,
            "label": ["spam"] * len(SPAM_TRAINING_MESSAGES)
            + ["not_spam"] * len(NOT_SPAM_TRAINING_MESSAGES),
        }
    )
    clean = preprocess_dataframe(raw, "text", "label", anonymize=True)
    count_columns = get_count_columns(clean)
    pipeline = build_pipeline(MultinomialNB(), count_columns, max_features=500)
    return pipeline.fit(clean[[TEXT_COL] + count_columns], clean["label"])
