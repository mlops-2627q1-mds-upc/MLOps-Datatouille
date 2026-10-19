"""
Fixtures shared by all the test modules.
"""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest


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
