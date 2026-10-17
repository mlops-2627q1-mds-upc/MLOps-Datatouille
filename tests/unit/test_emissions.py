"""Unit tests for the CodeCarbon wrapper in src/emissions.py."""

import csv
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.emissions import EMISSIONS_FILE, track_emissions

SUMMARY_KEYS = {"duration_s", "energy_consumed_kwh", "emissions_kg_co2eq"}


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the CodeCarbon CSV as a list of row dicts."""
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_track_emissions_writes_one_row_per_run(tmp_path: Path) -> None:
    """Each tracked block appends exactly one row with the stage name."""
    with track_emissions("unit-test", output_dir=tmp_path):
        sum(i * i for i in range(100_000))

    rows = _read_rows(tmp_path / EMISSIONS_FILE)
    assert len(rows) == 1
    assert rows[0]["project_name"] == "mlops-datatouille-unit-test"
    assert rows[0]["country_iso_code"] == "ESP"
    assert float(rows[0]["emissions"]) >= 0
    assert float(rows[0]["energy_consumed"]) >= 0


def test_track_emissions_records_even_if_block_fails(tmp_path: Path) -> None:
    """A crash inside the block still stops the tracker and saves the row."""
    try:
        with track_emissions("failing-stage", output_dir=tmp_path):
            raise RuntimeError("simulated training crash")
    except RuntimeError:
        pass

    rows = _read_rows(tmp_path / EMISSIONS_FILE)
    assert rows[0]["project_name"] == "mlops-datatouille-failing-stage"


def _logged_metrics(tracking_uri: str, stage: str, tmp_path: Path, **kwargs: str) -> set[str]:
    """Track an empty block inside an MLflow run and return the metric names logged."""
    mlflow.set_experiment("test-emissions")
    with mlflow.start_run() as run, track_emissions(stage, output_dir=tmp_path, **kwargs):
        pass
    return set(MlflowClient(tracking_uri).get_run(run.info.run_id).data.metrics)


def test_metric_prefix_keeps_model_name_out_of_mlflow(tracking_uri: str, tmp_path: Path) -> None:
    """MLflow metric names use the prefix, while the CSV keeps the full stage label."""
    metrics = _logged_metrics(
        tracking_uri, "train_final_logreg", tmp_path, metric_prefix="train_final"
    )
    assert metrics == {f"train_final_{key}" for key in SUMMARY_KEYS}
    assert _read_rows(tmp_path / EMISSIONS_FILE)[0]["project_name"].endswith("train_final_logreg")


def test_metric_prefix_defaults_to_stage(tracking_uri: str, tmp_path: Path) -> None:
    """Without a prefix, metric names keep using the stage label."""
    metrics = _logged_metrics(tracking_uri, "features", tmp_path)
    assert metrics == {f"features_{key}" for key in SUMMARY_KEYS}
