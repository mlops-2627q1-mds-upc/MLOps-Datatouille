"""CO2 experiment (issue #66): train each model N times under CodeCarbon.

Measures training energy and emissions per model with 5-fold CV plus final
fit, repeating each model `--reps` times and reporting the mean. Runs are
round-robin interleaved with a cooldown sleep to limit thermal bias.

Usage (from the root of the repo):
    python -m src.modeling.measure_co2 --reps 5 --train data/dataset/train.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import codecarbon
import pandas as pd
import sklearn

from src.config import (
    CODECARBON_COUNTRY_ISO_CODE,
    CODECARBON_MEASURE_POWER_SECS,
    CODECARBON_TRACKING_MODE,
)
from src.emissions import emissions_summary, track_emissions
from src.modeling.phase_3_training import load_xy, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["logreg", "naive_bayes", "linear_svm", "random_forest"]
SUMMARY_KEYS = ["duration_s", "energy_consumed_kwh", "emissions_kg_co2eq"]


def _cpu_model() -> str:
    """Return the CPU model name, or a fallback identifier.

    Returns:
        CPU model string from /proc/cpuinfo on Linux, else platform info.
    """
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_total_gb() -> float | None:
    """Return total RAM in GB, or None if it cannot be determined.

    Returns:
        Total physical memory in gigabytes, or None on unsupported platforms.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1e9, 2)
    except (AttributeError, ValueError, OSError):
        return None


def _gpu_model() -> str:
    """Return the GPU model string, or 'unknown' if none is detected.

    Returns:
        First GPU name reported by nvidia-smi, else 'unknown'.
    """
    if shutil.which("nvidia-smi") is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.splitlines()[0].strip()
    except (subprocess.SubprocessError, IndexError):
        return "unknown"


def collect_hardware(measure_power_secs: float) -> dict[str, str | float | int | None]:
    """Capture the machine and tracking setup behind a measurement run.

    Called once per experiment; the result is embedded in every run record so
    numbers always travel with the hardware that produced them.

    Args:
        measure_power_secs: Power sampling interval used for this experiment.

    Returns:
        Mapping with machine, OS, CPU, RAM, GPU, library versions and
        CodeCarbon tracking settings.
    """
    return {
        "machine": platform.node(),
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": _ram_total_gb(),
        "gpu_model": _gpu_model(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "codecarbon": codecarbon.__version__,
        "tracking_mode": CODECARBON_TRACKING_MODE,
        "country_iso_code": CODECARBON_COUNTRY_ISO_CODE,
        "measure_power_secs": measure_power_secs,
        "repo_default_measure_power_secs": CODECARBON_MEASURE_POWER_SECS,
    }


def run_rep(
    model: str,
    rep: int,
    X: pd.DataFrame,
    y: pd.Series,
    count_columns: list[str],
    max_features: int,
    cv_folds: int,
    random_state: int,
    measure_power_secs: float,
    hardware: dict,
) -> dict:
    """Train one model once under emissions tracking.

    Args:
        model: Model name accepted by `train_and_evaluate`.
        rep: Repetition index (1-based), for record keeping.
        X: Feature frame.
        y: Target series.
        count_columns: Numeric count feature names.
        max_features: TF-IDF vocabulary size.
        cv_folds: Cross-validation folds.
        random_state: Fixed seed so reps differ only by measurement noise.
        measure_power_secs: CodeCarbon power sampling interval.
        hardware: Machine descriptor from `collect_hardware`, embedded so each
            record carries the hardware that produced it.

    Returns:
        Record with model, rep, quality metrics, emission figures and hardware.
    """
    with tempfile.TemporaryDirectory(prefix="codecarbon-") as tmp:
        with track_emissions(
            f"train-{model}-rep{rep}",
            output_dir=Path(tmp),
            measure_power_secs=measure_power_secs,
        ) as tracker:
            _, metrics = train_and_evaluate(
                model, X, y, count_columns, max_features, cv_folds, random_state
            )
        summary = emissions_summary(tracker)
    return {"model": model, "rep": rep, **metrics, **summary, "hardware": hardware}


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean and std of emission figures per model.

    Args:
        records: Per-rep records as returned by `run_rep`.

    Returns:
        Mapping of model name to mean/std of duration, energy and emissions.
    """
    summary: dict[str, dict[str, float]] = {}
    df = pd.DataFrame(records)
    for model, group in df.groupby("model"):
        stats: dict[str, float] = {"n": int(len(group))}
        for key in SUMMARY_KEYS:
            stats[f"{key}_mean"] = float(group[key].mean())
            stats[f"{key}_std"] = float(group[key].std(ddof=1)) if len(group) > 1 else 0.0
            stats[f"{key}_min"] = float(group[key].min())
            stats[f"{key}_max"] = float(group[key].max())
        stats["cv_f1_macro_mean"] = float(group["cv_f1_macro"].mean())
        summary[str(model)] = stats
    return summary


def main() -> None:
    """Run the CO2 measurement protocol and write results."""
    parser = argparse.ArgumentParser(description="Measure training CO2 per model.")
    parser.add_argument("--train", required=True, help="Train CSV")
    parser.add_argument("--out", required=True, help="Output JSONL with per-rep records")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--measure-power-secs", type=float, default=1.0)
    parser.add_argument("--cooldown", type=float, default=10.0, help="Sleep between runs (s)")
    args = parser.parse_args()

    X, y, count_columns = load_xy(args.train, args.label_col)
    logger.info("Loaded %d rows for CO2 experiment", len(X))
    hardware = collect_hardware(args.measure_power_secs)
    logger.info("Hardware: %s", json.dumps(hardware))

    records: list[dict] = []
    for rep in range(1, args.reps + 1):
        for model in args.models:  # round-robin: interleave models across reps
            logger.info("Rep %d/%d model %s", rep, args.reps, model)
            records.append(
                run_rep(
                    model,
                    rep,
                    X,
                    y,
                    count_columns,
                    args.max_features,
                    args.cv_folds,
                    args.random_state,
                    args.measure_power_secs,
                    hardware,
                )
            )
            time.sleep(args.cooldown)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d records to %s", len(records), out)
    logger.info("Summary: %s", json.dumps(summarize(records)))


if __name__ == "__main__":
    main()
