"""Integration tests for the project's Great Expectations configuration.

These tests verify that the configured expectations accept valid data and reject
invalid data. They are separate from the Great Expectations validation itself,
which runs in the DVC ``validate_processed`` stage.
"""

import great_expectations as gx
import pandas as pd
import pytest

from src.data.gx_context_configuration import (
    EXPECTED_COLUMNS,
    ROOT_DIR,
    build_expectations,
)

PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean.csv"


def _validate_dataframe(dataframe: pd.DataFrame):
    """Run the project's actual expectations against an in-memory DataFrame."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="preprocessing_test_source")
    asset = data_source.add_dataframe_asset(name="preprocessing_test_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("preprocessing_test_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": dataframe})

    suite = gx.ExpectationSuite(name="preprocessing_test_suite")
    for expectation in build_expectations():
        suite.add_expectation(expectation)

    validator = context.get_validator(batch=batch, expectation_suite=suite)
    return validator.validate()


def _valid_processed_dataframe() -> pd.DataFrame:
    """Create a small DataFrame that satisfies the configured dataset expectations."""
    return pd.DataFrame(
        [
            ["Win prize", "spam", 9, 2, 1, 2, 0, 0, 1, "win prize"],
            ["Team meeting", "not_spam", 12, 2, 1, 1, 0, 0, 0, "team meet"],
        ],
        columns=EXPECTED_COLUMNS,
    )


def test_processed_data():
    """Test of validating the real processed dataset produced by the DVC pipeline."""
    if not PROCESSED_DATA_PATH.exists():
        pytest.skip("data/processed/clean.csv not found; run `dvc pull` or `dvc repro` first")
    result = _validate_dataframe(pd.read_csv(PROCESSED_DATA_PATH))
    failed_expectations = [
        expectation_result.expectation_config.type
        for expectation_result in result.results
        if not expectation_result.success
    ]
    assert result.success, f"Failing expectations: {failed_expectations}"


def test_configured_expectations_accept_valid_processed_dataframe():
    """Test of accepting an in-memory dataset that satisfies all configured expectations."""
    result = _validate_dataframe(_valid_processed_dataframe())

    assert result.success
    assert all(expectation_result.success for expectation_result in result.results)


@pytest.mark.parametrize(
    "column, invalid_value, expected_expectation",
    [
        ("label", "unknown", "expect_column_values_to_be_in_set"),
        ("clean_text", None, "expect_column_values_to_not_be_null"),
        ("n_urls", -1, "expect_column_values_to_be_between"),
        ("text", "Mail john@spam.com now", "expect_column_values_to_not_match_regex"),
        ("text", "Call +34 612 345 678 now", "expect_column_values_to_not_match_regex"),
    ],
)
def test_configured_expectations_reject_invalid_processed_data(
    column: str,
    invalid_value: object,
    expected_expectation: str,
):
    """Test of rejecting invalid labels, null text, negative counts, and contact details."""
    dataframe = _valid_processed_dataframe()
    dataframe.loc[0, column] = invalid_value

    result = _validate_dataframe(dataframe)
    failed_expectation_types = {
        expectation_result.expectation_config.type
        for expectation_result in result.results
        if not expectation_result.success
    }

    assert not result.success
    assert expected_expectation in failed_expectation_types


@pytest.mark.parametrize(
    "column, invalid_values",
    [
        ("label", [0, 1]),
        ("clean_text", [1, 2]),
        ("n_characters", ["9", "12"]),
    ],
)
def test_configured_expectations_reject_incorrect_column_types(
    column: str,
    invalid_values: list[object],
):
    """Test of rejecting non-string labels/text and non-integer count features."""
    dataframe = _valid_processed_dataframe()
    dataframe[column] = invalid_values

    result = _validate_dataframe(dataframe)
    failed_expectations = {
        (
            expectation_result.expectation_config.type,
            expectation_result.expectation_config.kwargs.get("column"),
        )
        for expectation_result in result.results
        if not expectation_result.success
    }

    assert not result.success
    assert ("expect_column_values_to_be_of_type", column) in failed_expectations
