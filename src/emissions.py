"""Energy and CO2 tracking for pipeline stages, built on CodeCarbon."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from codecarbon import OfflineEmissionsTracker
from loguru import logger
import mlflow

from src.config import (
    CODECARBON_COUNTRY_ISO_CODE,
    CODECARBON_MEASURE_POWER_SECS,
    CODECARBON_PROJECT_NAME,
    CODECARBON_TRACKING_MODE,
    EMISSIONS_DIR,
)

EMISSIONS_FILE = "emissions.csv"

def build_tracker(stage: str, output_dir: Path = EMISSIONS_DIR) -> OfflineEmissionsTracker:
    """Create an offline CodeCarbon tracker for one pipeline stage.

    Args:
        stage: Pipeline stage name (e.g. "features", "train", "predict").
            It is stored in the `project_name` column of the CSV as
            `<CODECARBON_PROJECT_NAME>-<stage>`.
        output_dir: Directory where `emissions.csv` is appended.

    Returns:
        A configured, not yet started, tracker.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return OfflineEmissionsTracker(
        project_name=f"{CODECARBON_PROJECT_NAME}-{stage}",
        country_iso_code=CODECARBON_COUNTRY_ISO_CODE,
        measure_power_secs=CODECARBON_MEASURE_POWER_SECS,
        tracking_mode=CODECARBON_TRACKING_MODE,
        output_dir=str(output_dir),
        output_file=EMISSIONS_FILE,
        log_level="warning",
    )


def emissions_summary(tracker: OfflineEmissionsTracker) -> dict[str, float]:
    """Extract the key energy and CO2 figures from a stopped tracker.

    Args:
        tracker: A tracker on which `stop()` has already been called.

    Returns:
        Mapping with duration (s), energy (kWh) and emissions (kg CO2eq).
    """
    data = tracker.final_emissions_data
    return {
        "duration_s": float(data.duration),
        "energy_consumed_kwh": float(data.energy_consumed),
        "emissions_kg_co2eq": float(data.emissions),
    }


def _log_to_mlflow(stage: str, summary: dict[str, float], csv_path: Path) -> None:
    """Log the emission figures to the active MLflow run, if there is one.

    Args:
        stage: Pipeline stage name, used as metric prefix.
        summary: Output of `emissions_summary`.
        csv_path: Path to the CodeCarbon CSV, logged as an artifact.
    """
    if mlflow.active_run() is None:
        return
    mlflow.log_metrics({f"{stage}_{key}": value for key, value in summary.items()})
    mlflow.log_artifact(str(csv_path), artifact_path="emissions")


@contextmanager
def track_emissions(stage: str, output_dir: Path = EMISSIONS_DIR) -> Iterator[OfflineEmissionsTracker]:
    """Measure energy and CO2 of the wrapped block and record the result.

    The result is appended to `<output_dir>/emissions.csv` and, if an MLflow
    run is active, logged as `<stage>_*` metrics plus the CSV as artifact.

    Args:
        stage: Pipeline stage name (e.g. "features", "train", "predict").
        output_dir: Directory where `emissions.csv` is appended.

    Yields:
        The running tracker (useful for `start_task` / `stop_task`).
    """
    tracker = build_tracker(stage, output_dir)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        summary = emissions_summary(tracker)
        logger.info(
            f"[{stage}] {summary['energy_consumed_kwh']:.3g} kWh, "
            f"{summary['emissions_kg_co2eq']:.3g} kg CO2eq in {summary['duration_s']:.1f} s"
        )
        _log_to_mlflow(stage, summary, output_dir / EMISSIONS_FILE)
