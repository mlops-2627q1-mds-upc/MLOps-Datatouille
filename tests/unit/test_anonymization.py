"""Unit tests for named-entity anonymization in phase-one cleaning."""

import pytest

from src.data import phase_1_cleaning


def test_anonymize_replaces_multiple_entities_from_right_to_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of replacing multiple named entities without shifting later spans."""
    entities = [
        {"start": 0, "end": 5, "entity_group": "PER"},
        {"start": 10, "end": 14, "entity_group": "ORG"},
        {"start": 18, "end": 23, "entity_group": "LOC"},
    ]

    def fake_ner_pipeline(model_name: str):
        return lambda text: entities

    monkeypatch.setattr(phase_1_cleaning, "_get_ner_pipeline", fake_ner_pipeline)
    result = phase_1_cleaning.anonimitzar_amb_ia("Alice met Acme in Paris")
    assert result == "[PER] met [ORG] in [LOC]"


def test_anonymize_leaves_text_unchanged_when_no_entities_are_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of leaving the input unchanged when the NER model finds no entities."""
    monkeypatch.setattr(phase_1_cleaning, "_get_ner_pipeline", lambda model_name: lambda text: [])
    text = "The meeting starts tomorrow."
    assert phase_1_cleaning.anonimitzar_amb_ia(text) == text


def test_anonymize_propagates_ner_inference_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test of propagating NER inference errors instead of returning unredacted text."""

    def fail_to_infer(model_name: str):
        def infer(text: str):
            raise RuntimeError("NER inference failed")
        return infer

    monkeypatch.setattr(phase_1_cleaning, "_get_ner_pipeline", fail_to_infer)

    with pytest.raises(RuntimeError, match="NER inference failed"):
        phase_1_cleaning.anonimitzar_amb_ia("Alice visited Paris")
