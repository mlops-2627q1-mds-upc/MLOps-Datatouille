"""
Unit tests for the CO2 helpers in src/modeling/measure_co2.py.
"""

from src.modeling.measure_co2 import collect_hardware, summarize


def _record(model: str, rep: int, emissions: float) -> dict:
    """
    Build a minimal per-rep record for testing.
    """
    return {
        "model": model,
        "rep": rep,
        "cv_f1_macro": 0.99,
        "duration_s": 10.0 * rep,
        "energy_consumed_kwh": emissions * 2,
        "emissions_kg_co2eq": emissions,
    }


def test_summarize_reports_mean_and_std_per_model() -> None:
    """Means aggregate reps; std is zero for a single rep."""
    records = [_record("logreg", 1, 0.001), _record("logreg", 2, 0.003)]
    summary = summarize(records)
    assert summary["logreg"]["n"] == 2
    assert summary["logreg"]["emissions_kg_co2eq_mean"] == 0.002
    assert summary["logreg"]["emissions_kg_co2eq_std"] > 0

    single = summarize([_record("naive_bayes", 1, 0.005)])
    assert single["naive_bayes"]["emissions_kg_co2eq_mean"] == 0.005
    assert single["naive_bayes"]["emissions_kg_co2eq_std"] == 0.0


def test_collect_hardware_returns_machine_descriptor() -> None:
    """Hardware descriptor carries machine, OS, CPU and tracking settings."""
    hardware = collect_hardware(measure_power_secs=1.0)
    assert hardware["machine"]
    assert hardware["os"]
    assert hardware["cpu_model"]
    assert hardware["measure_power_secs"] == 1.0
    assert hardware["country_iso_code"] == "ESP"
