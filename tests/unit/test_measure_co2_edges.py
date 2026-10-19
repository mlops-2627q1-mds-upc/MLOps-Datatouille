"""
Unit tests for the functions in `src.modeling.measure_co2` that collect hardware
specs and select default classifiers.
"""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.modeling.measure_co2 import _cpu_model, _default_classifier, _gpu_model, _ram_total_gb


def test_default_classifiers_alternatives():
    assert isinstance(_default_classifier("naive_bayes"), MultinomialNB)
    assert isinstance(_default_classifier("linear_svm"), LinearSVC)
    assert isinstance(_default_classifier("random_forest"), RandomForestClassifier)

    with pytest.raises(ValueError, match="Unknown model"):
        _default_classifier("unknown_model")


def test_hardware_collection_does_not_crash():
    # Hardware specs vary by OS, so we simply verify they run without
    # crashing to cover the nested try/except blocks.
    assert isinstance(_cpu_model(), str)

    ram = _ram_total_gb()
    assert isinstance(ram, (float, type(None)))

    assert isinstance(_gpu_model(), str)
