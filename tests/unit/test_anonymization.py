"""Unit tests for email and phone-number anonymization."""

import pytest

from src.data import preprocess as pp


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Write to john.doe@gmail.com today", "Write to xxemail today"),
        ("Call +34 612 345 678 now", "Call xxphone now"),
        ("Call (555) 123-4567 now", "Call xxphone now"),
        ("Tel:612345678", "Tel: xxphone"),
    ],
)
def test_anonymize_text_replaces_contact_details(text, expected):
    """Test of replacing email addresses and phone numbers with placeholder tokens."""
    assert pp.remove_whitespace(pp.anonymize_text(text)) == expected


def test_anonymize_text_keeps_dates_and_short_numbers():
    """Test of leaving dates, times and short numbers unchanged."""
    text = "Meeting on 2026-10-15 at 10:30 in room 42"
    assert pp.anonymize_text(text) == text


def test_anonymize_text_leaves_text_without_contact_details_unchanged():
    """Test of returning the input unchanged when it has no contact details."""
    text = "Win a free prize now!"
    assert pp.anonymize_text(text) == text


def test_anonymization_tokens_survive_text_cleaning():
    """Test of preserving the placeholder tokens through full text cleaning."""
    anonymized = pp.anonymize_text("Write to john@spam.com or phone 612345678")
    tokens = pp.clean_text(anonymized).split()
    assert "xxemail" in tokens
    assert "xxphone" in tokens
