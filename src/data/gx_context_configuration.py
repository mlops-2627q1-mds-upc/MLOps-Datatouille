"""Great Expectations configuration: what the cleaned dataset must look like.

Creates (or updates) the GX file context in gx/ with:
    - a pandas data source pointing to data/processed/,
    - the expectation suite of the cleaned dataset,
    - a validation definition and a checkpoint that runs it.

Then it runs the checkpoint and exits with an error if the processed data does not meet
the expectations. The HTML report is written to gx/uncommitted/data_docs/.

Usage (from the root of the repo):
    python -m src.data.gx_context_configuration
"""

import json
import logging
from pathlib import Path
import sys

import great_expectations as gx
import great_expectations.expectations as gxe

from src.data.preprocess import EMAIL_PATTERN, PHONE_PATTERN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("great_expectations").setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_FILE = "clean.csv"

TEXT_COL = "text"
LABEL_COL = "label"
CLEAN_TEXT_COL = "clean_text"
LABELS = ["spam", "not_spam"]
COUNT_COLUMNS = [
    "n_characters",
    "n_words",
    "n_sentences",
    "n_uppercase",
    "n_urls",
    "n_emojis",
    "n_punctuation",
]
EXPECTED_COLUMNS = [
    TEXT_COL,
    LABEL_COL,
    *COUNT_COLUMNS,
    CLEAN_TEXT_COL,
]

DATA_SOURCE_NAME = "processed_data"
ASSET_NAME = "clean_spam_dataset"
BATCH_NAME = "full_csv"
SUITE_NAME = "processed_spam_dataset_suite"
VALIDATION_NAME = "processed_spam_dataset_validation"
CHECKPOINT_NAME = "processed_spam_dataset_checkpoint"
VALIDATION_REPORT = ROOT_DIR / "data" / "processed" / "gx_validation.json"


def build_expectations():
    """Expectations that the cleaned dataset must meet before splitting."""
    expectations = [
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=EXPECTED_COLUMNS),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column=TEXT_COL),
        gxe.ExpectColumnValuesToBeOfType(column=TEXT_COL, type_="str"),
        gxe.ExpectColumnValueLengthsToBeBetween(column=TEXT_COL, min_value=1),
        gxe.ExpectColumnValuesToNotMatchRegex(
            column=TEXT_COL, regex=f"{EMAIL_PATTERN}|{PHONE_PATTERN}"
        ),
        gxe.ExpectColumnValuesToNotBeNull(column=CLEAN_TEXT_COL),
        gxe.ExpectColumnValuesToBeOfType(column=CLEAN_TEXT_COL, type_="str"),
        gxe.ExpectColumnValueLengthsToBeBetween(column=CLEAN_TEXT_COL, min_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column=LABEL_COL),
        gxe.ExpectColumnValuesToBeOfType(column=LABEL_COL, type_="str"),
        gxe.ExpectColumnValuesToBeInSet(column=LABEL_COL, value_set=LABELS),
        gxe.ExpectColumnDistinctValuesToEqualSet(column=LABEL_COL, value_set=LABELS),
    ]
    for column in COUNT_COLUMNS:
        expectations.extend(
            [
                gxe.ExpectColumnValuesToNotBeNull(column=column),
                gxe.ExpectColumnValuesToBeOfType(column=column, type_="int64"),
                gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0),
            ]
        )
    return expectations


def configure_context():
    """Create or update the GX context, data source, suite, validation and checkpoint."""
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

    data_source = context.data_sources.add_or_update_pandas_filesystem(
        name=DATA_SOURCE_NAME, base_directory=PROCESSED_DIR.relative_to(ROOT_DIR).as_posix()
    )
    asset = data_source.add_csv_asset(name=ASSET_NAME)
    batch_definition = asset.add_batch_definition_path(name=BATCH_NAME, path=PROCESSED_FILE)

    suite = gx.ExpectationSuite(name=SUITE_NAME)
    for expectation in build_expectations():
        suite.add_expectation(expectation)
    suite = context.suites.add_or_update(suite)

    validation = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=VALIDATION_NAME, data=batch_definition, suite=suite)
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=CHECKPOINT_NAME,
            validation_definitions=[validation],
            actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
        )
    )
    return checkpoint


def main():
    checkpoint = configure_context()
    logger.info("Validating %s", PROCESSED_DIR / PROCESSED_FILE)
    result = checkpoint.run()

    expectation_results = []
    for validation_result in result.run_results.values():
        for res in validation_result.results:
            config = res.expectation_config
            status = "OK  " if res.success else "FAIL"
            logger.info("%s %s %s", status, config.type, config.kwargs.get("column", ""))
            expectation_results.append(
                {
                    "expectation": config.type,
                    "column": config.kwargs.get("column"),
                    "success": res.success,
                }
            )

    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.write_text(
        json.dumps(
            {"success": result.success, "expectations": expectation_results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not result.success:
        logger.error("The processed dataset does not meet the expectations")
        sys.exit(1)
    logger.info("The processed dataset meets all the expectations; report: %s", VALIDATION_REPORT)


if __name__ == "__main__":
    main()
