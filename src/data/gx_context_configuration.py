"""Great Expectations configuration: what the raw dataset must look like.

Creates (or updates) the GX file context in gx/ with:
    - a pandas data source pointing to data/raw/,
    - the expectation suite of the raw dataset,
    - a validation definition and a checkpoint that runs it.

Then it runs the checkpoint and exits with an error if the raw data does not meet
the expectations. The HTML report is written to gx/uncommitted/data_docs/.

Usage (from the root of the repo):
    python -m src.data.gx_context_configuration
"""

import logging
import sys
from pathlib import Path
import great_expectations as gx
import great_expectations.expectations as gxe

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("great_expectations").setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILE = "daisy_dataset_spam_detection.csv"

TEXT_COL = "text"
LABEL_COL = "label"
LABELS = ["spam", "not_spam"]

DATA_SOURCE_NAME = "raw_data"
ASSET_NAME = "spam_dataset"
BATCH_NAME = "full_csv"
SUITE_NAME = "raw_spam_dataset_suite"
VALIDATION_NAME = "raw_spam_dataset_validation"
CHECKPOINT_NAME = "raw_spam_dataset_checkpoint"


def build_expectations():
    """Expectations that the raw dataset must meet before entering the pipeline."""
    return [
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=[TEXT_COL, LABEL_COL]),
        gxe.ExpectTableRowCountToBeBetween(min_value=1000),
        gxe.ExpectColumnValuesToNotBeNull(column=TEXT_COL),
        gxe.ExpectColumnValuesToBeOfType(column=TEXT_COL, type_="str"),
        gxe.ExpectColumnValueLengthsToBeBetween(column=TEXT_COL, min_value=1),
        gxe.ExpectColumnValuesToNotBeNull(column=LABEL_COL),
        gxe.ExpectColumnValuesToBeInSet(column=LABEL_COL, value_set=LABELS),
        gxe.ExpectColumnDistinctValuesToEqualSet(column=LABEL_COL, value_set=LABELS),
    ]


def configure_context():
    """Create or update the GX context, data source, suite, validation and checkpoint."""
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

    data_source = context.data_sources.add_or_update_pandas_filesystem(
        name=DATA_SOURCE_NAME, base_directory=RAW_DIR
    )
    asset = data_source.add_csv_asset(name=ASSET_NAME)
    batch_definition = asset.add_batch_definition_path(name=BATCH_NAME, path=RAW_FILE)

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
    logger.info("Validating %s", RAW_DIR / RAW_FILE)
    result = checkpoint.run()

    for validation_result in result.run_results.values():
        for res in validation_result.results:
            config = res.expectation_config
            status = "OK  " if res.success else "FAIL"
            logger.info("%s %s %s", status, config.type, config.kwargs.get("column", ""))

    if not result.success:
        logger.error("The raw dataset does not meet the expectations")
        sys.exit(1)
    logger.info("The raw dataset meets all the expectations")


if __name__ == "__main__":
    main()
