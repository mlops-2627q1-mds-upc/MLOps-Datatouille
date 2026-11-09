"""
Unit tests for the Gemini summarizer, with the Gemini client mocked.
"""

from unittest.mock import MagicMock

import pytest

from src.api import summarizer as sm
from src.api.summarizer import Summarizer, SummaryUnavailableError, build_summarizer

CONTACT_MESSAGE = "Call me at +34 612 345 678 or write to ana@example.com about the invoice."


@pytest.fixture
def gemini() -> MagicMock:
    """Mocked Gemini client that answers with a summary surrounded by whitespace."""
    client = MagicMock()
    client.models.generate_content.return_value.text = "  Request about an invoice.\n"
    return client


def test_summarize_returns_stripped_summary(gemini: MagicMock) -> None:
    """Test of the summary returned without surrounding whitespace, with the given model."""
    summary = Summarizer(gemini, "test-model").summarize("Some message")
    assert summary == "Request about an invoice."
    assert gemini.models.generate_content.call_args.kwargs["model"] == "test-model"


def test_summarize_sends_anonymized_text(gemini: MagicMock) -> None:
    """Test of the anonymization of contact details before the request."""
    Summarizer(gemini, "test-model").summarize(CONTACT_MESSAGE)
    contents = gemini.models.generate_content.call_args.kwargs["contents"]
    assert "xxemail" in contents and "xxphone" in contents
    assert "ana@example.com" not in contents and "612" not in contents


def test_summarize_raises_on_client_error(gemini: MagicMock) -> None:
    """Test of the error raised when the Gemini request fails."""
    gemini.models.generate_content.side_effect = TimeoutError("timed out")
    with pytest.raises(SummaryUnavailableError, match="timed out"):
        Summarizer(gemini, "test-model").summarize("Some message")


@pytest.mark.parametrize("text", [None, "", "   "], ids=["none", "empty", "blank"])
def test_summarize_raises_on_empty_answer(gemini: MagicMock, text: str | None) -> None:
    """Test of the error raised when Gemini returns no text."""
    gemini.models.generate_content.return_value.text = text
    with pytest.raises(SummaryUnavailableError, match="empty"):
        Summarizer(gemini, "test-model").summarize("Some message")


@pytest.mark.parametrize("api_key", [None, ""], ids=["none", "empty"])
def test_build_summarizer_without_key_returns_none(
    api_key: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test of the disabled summarizer when no API key is configured."""
    client_class = MagicMock()
    monkeypatch.setattr(sm.genai, "Client", client_class)
    assert build_summarizer(api_key, "test-model", 30.0) is None
    client_class.assert_not_called()


def test_build_summarizer_sets_key_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the client created with the API key and the timeout in milliseconds."""
    client_class = MagicMock()
    monkeypatch.setattr(sm.genai, "Client", client_class)
    summarizer = build_summarizer("secret-key", "test-model", 2.5)
    assert summarizer is not None
    assert summarizer.model == "test-model"
    kwargs = client_class.call_args.kwargs
    assert kwargs["api_key"] == "secret-key"
    assert kwargs["http_options"].timeout == 2500


def test_build_summarizer_returns_none_when_client_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test of the disabled summarizer when the Gemini client cannot be created."""
    monkeypatch.setattr(sm.genai, "Client", MagicMock(side_effect=ValueError("bad key")))
    assert build_summarizer("secret-key", "test-model", 30.0) is None
