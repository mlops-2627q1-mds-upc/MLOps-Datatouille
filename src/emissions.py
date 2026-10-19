"""
Energy and CO2 tracking for pipeline stages, built on CodeCarbon.
"""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

from codecarbon import OfflineEmissionsTracker
import mlflow

from src.config import (
    CODECARBON_COUNTRY_ISO_CODE,
    CODECARBON_MEASURE_POWER_SECS,
    CODECARBON_PROJECT_NAME,
    CODECARBON_TRACKING_MODE,
    EMISSIONS_DIR,
)

EMISSIONS_FILE = "emissions.csv"

logger = logging.getLogger(__name__)

# CodeCarbon CSV column -> name of the figure in summaries and MLflow metrics
SUMMARY_COLUMNS = {
    "duration": "duration_s",
    "energy_consumed": "energy_consumed_kwh",
    "emissions": "emissions_kg_co2eq",
}


def build_tracker(
    stage: str,
    output_dir: Path = EMISSIONS_DIR,
    measure_power_secs: float | None = None,
) -> OfflineEmissionsTracker:
    """
    Create an offline CodeCarbon tracker for one pipeline stage.

    Args:
        stage: Pipeline stage name (e.g. "features", "train", "predict").
            It is stored in the `project_name` column of the CSV as
            `<CODECARBON_PROJECT_NAME>-<stage>`.
        output_dir: Directory where `emissions.csv` is appended.
        measure_power_secs: Power sampling interval. Defaults to
            `CODECARBON_MEASURE_POWER_SECS`. Override (e.g. 1 s) for short
            experiments where the default interval yields too few readings.

    Returns:
        A configured, not yet started, tracker.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return OfflineEmissionsTracker(
        project_name=f"{CODECARBON_PROJECT_NAME}-{stage}",
        country_iso_code=CODECARBON_COUNTRY_ISO_CODE,
        measure_power_secs=CODECARBON_MEASURE_POWER_SECS
        if measure_power_secs is None
        else measure_power_secs,
        tracking_mode=CODECARBON_TRACKING_MODE,
        output_dir=str(output_dir),
        output_file=EMISSIONS_FILE,
        log_level="warning",
    )


def emissions_summary(tracker: OfflineEmissionsTracker) -> dict[str, float]:
    """
    Extract the key energy and CO2 figures from a stopped tracker.

    Args:
        tracker: A tracker on which `stop()` has already been called.

    Returns:
        Mapping with duration (s), energy (kWh) and emissions (kg CO2eq).
    """
    return total_emissions([emissions_record(tracker)])


def emissions_record(tracker: OfflineEmissionsTracker) -> dict[str, Any]:
    """
    Return the full measurement of a stopped tracker, as one row of the CodeCarbon CSV.

    Args:
        tracker: A tracker on which `stop()` has already been called.

    Returns:
        Column name to value, with the same columns as `emissions.csv`.
    """
    return dict(tracker.final_emissions_data.values)


def total_emissions(
    records: Iterable[Mapping[str, Any]], prefix: str | None = None
) -> dict[str, float]:
    """
    Sum the duration, energy and emissions of several measurements.

    Args:
        records: Rows as returned by `emissions_record`.
        prefix: Optional prefix of the returned names, e.g. "train" -> "train_duration_s".

    Returns:
        Mapping with total duration (s), energy (kWh) and emissions (kg CO2eq).
    """
    records = list(records)
    totals = {
        name: float(sum(record[column] for record in records))
        for column, name in SUMMARY_COLUMNS.items()
    }
    return {f"{prefix}_{name}" if prefix else name: value for name, value in totals.items()}


def _log_to_mlflow(prefix: str, summary: dict[str, float]) -> None:
    """
    Log the emission figures to the active MLflow run, if there is one.

    Args:
        prefix: Metric name prefix, e.g. "train_final" -> "train_final_duration_s".
        summary: Output of `emissions_summary`.
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_metrics({f"{prefix}_{key}": value for key, value in summary.items()})


@contextmanager
def track_emissions(
    stage: str,
    output_dir: Path = EMISSIONS_DIR,
    measure_power_secs: float | None = None,
    metric_prefix: str | None = None,
) -> Iterator[OfflineEmissionsTracker]:
    """
    Measure energy and CO2 of the wrapped block and record the result.

    The result is appended to `<output_dir>/emissions.csv` and, if an MLflow
    run is active, logged as `<metric_prefix>_*` metrics. The CSV is not uploaded
    to MLflow, since it accumulates every past measurement of the machine; use
    `emissions_record` to log only the rows of a run.

    Args:
        stage: Pipeline stage name (e.g. "features", "train", "predict").
        output_dir: Directory where `emissions.csv` is appended.
        measure_power_secs: Power sampling interval override. Defaults to
            `CODECARBON_MEASURE_POWER_SECS`.
        metric_prefix: Prefix of the MLflow metric names. Defaults to `stage`.
            Set it when `stage` holds run-specific parts (e.g. the model name)
            that would give each run different metric names, so runs could
            not be compared in the same MLflow plot.

    Yields:
        The running tracker (useful for `start_task` / `stop_task`).
    """
    tracker = build_tracker(stage, output_dir, measure_power_secs)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        summary = emissions_summary(tracker)
        logger.info(
            "[%s] %.3g kWh, %.3g kg CO2eq in %.1f s",
            stage,
            summary["energy_consumed_kwh"],
            summary["emissions_kg_co2eq"],
            summary["duration_s"],
        )
        _log_to_mlflow(metric_prefix or stage, summary)
