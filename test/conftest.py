"""Fixtures shared by all the test modules."""

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
