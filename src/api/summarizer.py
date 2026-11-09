"""
Summarization of messages with Google Gemini, served by the ``/summarize`` endpoint.

The Gemini client is created once at startup and shared by every request. Messages are
anonymized before they are sent, so email addresses and phone numbers never leave the service.
Summarization is optional: without an API key the API still starts and ``/summarize`` answers
503.
"""

import logging

from google import genai
from google.genai import types

from src.data.preprocess import anonymize_text

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are an expert assistant specialized in text summarization. "
    "Summarize the text in one or two short sentences of plain text. "
    "CRITICAL: Do not use Markdown or any formatting: no asterisks, bullet points, "
    "headings or line breaks. "
    "CRITICAL: Always output the summary in the SAME language as the input text "
    "provided by the user."
)
PROMPT_TEMPLATE = "Summarize the following text in one or two plain-text sentences:\n\n{text}"
TEMPERATURE = 0.3


class SummaryUnavailableError(RuntimeError):
    """Raised when Gemini cannot produce a summary (error, timeout or empty answer)."""


class Summarizer:
    """Gemini client configured to summarize messages."""

    def __init__(self, client: genai.Client, model: str) -> None:
        """
        Set the client and the Gemini model used for every summary.

        Args:
            client: Gemini client, with its API key and timeout already configured.
            model: Name of the Gemini model (e.g. ``gemini-3.8-flash``).
        """
        self.client = client
        self.model = model
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION, temperature=TEMPERATURE
        )

    def summarize(self, text: str) -> str:
        """
        Summarize a message in one or two plain-text sentences.

        Args:
            text: Raw message. It is anonymized before being sent to Gemini.

        Returns:
            The summary, without leading or trailing whitespace.

        Raises:
            SummaryUnavailableError: If the request fails, times out or returns no text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=PROMPT_TEMPLATE.format(text=anonymize_text(text)),
                config=self.config,
            )
        except Exception as error:
            raise SummaryUnavailableError(f"Gemini request failed: {error}") from error
        summary = (response.text or "").strip()
        if not summary:
            raise SummaryUnavailableError("Gemini returned an empty summary")
        return summary


def build_summarizer(api_key: str | None, model: str, timeout_seconds: float) -> Summarizer | None:
    """
    Create the summarizer, or return None if summarization cannot be enabled.

    Args:
        api_key: Gemini API key, or None if it is not configured.
        model: Name of the Gemini model.
        timeout_seconds: Maximum duration of each Gemini request.

    Returns:
        The summarizer, or None if there is no API key or the client cannot be created.
    """
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set: /summarize is disabled")
        return None
    try:
        client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )
    except Exception:
        logger.exception("Gemini client cannot be created: /summarize is disabled")
        return None
    return Summarizer(client, model)
