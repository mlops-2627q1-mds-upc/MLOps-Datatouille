"""Unit tests for the CodeCarbon wrapper in src/emissions.py."""

import csv
from pathlib import Path

from src.emissions import EMISSIONS_FILE, track_emissions


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
