"""Stage 1B literature benchmarks: tolerance tests.

Two layers:

* **Recorded results** (``benchmarks/*/results/comparison.csv``, produced by the
  full runs and committed alongside ``benchmarks/REPORT.md``). Every trend the
  papers state must PASS; a quantitative value outside tolerance is a
  DEVIATION and is surfaced as a :class:`~benchmarks.common.BenchmarkDeviation`
  warning with its stated cause --- never a failure (project specification,
  Stage 1B).
* **Fresh quick runs** of the trends, so a code change that breaks one is
  caught here and not only when someone reruns the full benchmark.

References: Li et al. 2015, doi:10.1073/pnas.1504484112; Zhang et al. 2023,
doi:10.3390/ijms24043338.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.benchmark_01_tassaw import cases as li
from benchmarks.benchmark_02_alternating_baw import cases as zhang
from benchmarks.common import (
    CALIBRATION,
    DEVIATION,
    FAIL,
    PASS,
    BenchmarkDeviation,
    Check,
    warn_deviations,
)

ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
RECORDED = {
    "benchmark_01_tassaw": ROOT / "benchmark_01_tassaw" / "results",
    "benchmark_02_alternating_baw": ROOT / "benchmark_02_alternating_baw" / "results",
}


# -- the comparison record itself ----------------------------------------------


def test_a_deviation_must_state_its_cause():
    with pytest.raises(ValueError, match="likely cause"):
        Check("x", "q", "quantity", 50.0, 10.0, "%", tolerance=5.0)
    ok = Check("x", "q", "quantity", 12.0, 10.0, "%", tolerance=5.0)
    assert ok.status == PASS and ok.deviation == pytest.approx(2.0)
    rng = Check("x", "r", "range", 97.0, (83.0, 96.0), "%", cause="because")
    assert rng.status == DEVIATION and rng.deviation == pytest.approx(1.0)
    assert Check("x", "t", "trend", "a", "b", passed=False).status == FAIL
    assert Check("x", "c", "calibration", 5.0, 5.0).status == CALIBRATION


def test_deviations_warn_rather_than_fail():
    dev = Check("x", "q", "quantity", 50.0, 10.0, "%", tolerance=5.0, cause="unknown SI value")
    with pytest.warns(BenchmarkDeviation, match="unknown SI value"):
        warn_deviations([dev])


# -- configs are runnable and their calibrated numbers are the ones derived ----


def test_benchmark_configs_validate():
    assert li.make_params().mode == "tassaw"
    assert zhang.make_params().mode == "alternating_baw"


def test_zhang_energy_densities_are_the_calibrated_ones():
    """config.yaml carries the E_ac values the stated rules imply (to 0.1 %)."""
    cal = zhang.calibration()
    e1, e3 = (p.reference_energy_density for p in zhang.make_params().switching.phases)
    assert e1 == pytest.approx(cal["E1_J_m3"], rel=1e-3)
    assert e3 == pytest.approx(cal["E3_J_m3"], rel=1e-3)
    assert cal["rule"]["feasible"]


def test_li_reference_pressure_is_inside_the_recorded_calibration_bracket():
    summary = RECORDED["benchmark_01_tassaw"] / "summary.json"
    if not summary.exists():
        pytest.skip("full benchmark_01 has not been run")
    cal = json.loads(summary.read_text())["calibration"]
    if "bracket_Pa" not in cal:
        pytest.skip("recorded run reused the configured value")
    lo, hi = cal["bracket_Pa"]
    p = li.make_params().power_drive.reference_pressure
    assert lo <= p <= hi, f"config.yaml {p:.0f} Pa outside the fitted bracket {lo:.0f}-{hi:.0f}"


# -- recorded full results -----------------------------------------------------


@pytest.mark.parametrize("name", list(RECORDED))
def test_recorded_benchmark(name):
    path = RECORDED[name] / "comparison.csv"
    if not path.exists():
        pytest.skip(f"{name} has not been run: python -m benchmarks.{name}.run")
    df = pd.read_csv(path)
    failed = df[df["status"] == FAIL]
    assert failed.empty, "stated trends the model gets wrong:\n" + failed[
        ["case", "quantity", "ours"]].to_string()
    deviations = df[df["status"] == DEVIATION]
    assert deviations["cause"].fillna("").str.len().gt(0).all()
    for _, row in deviations.iterrows():
        warnings.warn(f"{name} {row['case']} {row['quantity']}: ours {row['ours']} vs paper "
                      f"{row['paper']} {row['unit']} — {row['cause']}", BenchmarkDeviation,
                      stacklevel=1)


def test_report_cites_both_papers_and_every_check():
    report = ROOT / "REPORT.md"
    if not report.exists():
        pytest.skip("benchmarks/REPORT.md not generated yet")
    text = report.read_text(encoding="utf-8")
    assert "10.1073/pnas.1504484112" in text and "10.3390/ijms24043338" in text
    for results in RECORDED.values():
        csv = results / "comparison.csv"
        if csv.exists():
            for quantity in pd.read_csv(csv)["quantity"]:
                assert quantity in text, f"REPORT.md is stale: missing {quantity!r}"


# -- fresh quick runs of the stated trends -------------------------------------


@pytest.fixture(scope="module")
def li_pressure():
    return float(li.make_params().power_drive.reference_pressure)


def _passed(checks, needle):
    (c,) = [c for c in checks if needle in c.quantity]
    return c


def test_li_optimum_tilt_falls_as_flow_rises(li_pressure):
    _, checks = li.case_a(li_pressure, n=6, flows=(25.0, 75.0, 125.0),
                          tilts=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 35.0))
    c = _passed(checks, "optimum tilt falls")
    assert c.status == PASS, c.ours


def test_li_idt_length_has_an_interior_optimum(li_pressure):
    _, checks = li.case_b(li_pressure, n=6, lengths=(2, 5, 8, 10, 12, 20, 30))
    assert _passed(checks, "interior maximum").status == PASS


def test_li_power_trades_recovery_for_removal(li_pressure):
    _, checks = li.case_c(li_pressure, powers=(31.0, 33.0, 35.0, 37.0, 39.0), count=80)
    assert _passed(checks, "recovery rises").status == PASS
    assert _passed(checks, "removal falls").status == PASS


def _small_zhang(monkeypatch):
    """Fewer cells for the quick runs: the trends, not the percentages, are tested."""
    base = zhang.base_config

    def smaller():
        cfg = base()
        cfg["populations"] = [dict(p, count=60 if p.get("target") else 120)
                              for p in cfg["populations"]]
        return cfg

    monkeypatch.setattr(zhang, "base_config", smaller)


def test_zhang_contamination_falls_with_the_3mhz_duration(monkeypatch):
    _small_zhang(monkeypatch)
    df, checks = zhang.case_a((0.2, 1.4, 3.0))
    assert _passed(checks, "contamination falls").status == PASS
    assert _passed(checks, "capture changes little").status == PASS


def test_zhang_1mhz_amplitude_raises_capture_more_than_contamination(monkeypatch):
    _small_zhang(monkeypatch)
    _, checks, _ = zhang.case_b(t1=(0.4, 0.8), v1=(6.0, 8.0, 9.0))
    assert _passed(checks, "both metrics rise with 1 MHz V1").status == PASS
    assert _passed(checks, "rises much less than capture (up to the best V1)").status == PASS


def test_counterfactual_follows_gorkov():
    """Compressibility moves separation in the direction Gor'kov theory says.

    The paper's claim that the measured compressibility difference rescues
    similar-sized cells is a separate, recorded claim; what is asserted here is
    that the MODEL is self-consistent: less compressible CTCs separate better,
    more compressible ones worse, than size alone.
    """
    _, checks, _ = zhang.counterfactual(v1=(5.0, 7.0, 8.0, 9.0, 12.0))
    assert _passed(checks, "model follows Gor'kov").status == PASS
