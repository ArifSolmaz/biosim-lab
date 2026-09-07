"""Command-line interface."""

import pytest
from typer.testing import CliRunner

from biosim_lab import __version__
from biosim_lab.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_list_shows_all_instruments_and_solvers():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    for name in ("saw_sorter", "impedance_rtca", "cell_counter", "cell_tracker"):
        assert name.replace("_", "") in result.stdout.replace("_", "").replace("\n", "")


def test_doctor_reports_missing_backends_without_failing():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "instrument(s) discovered" in result.stdout


def test_init_writes_a_config_that_runs(tmp_path):
    cfg = tmp_path / "c.yaml"
    result = runner.invoke(app, ["init", "saw_sorter", "-o", str(cfg)])
    assert result.exit_code == 0, result.stdout
    assert cfg.exists()

    result = runner.invoke(app, ["run", str(cfg), "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "out" / "ctc_vs_rbc.nc").exists()
    assert (tmp_path / "out" / "ctc_vs_rbc_metrics.csv").exists()
    assert "efficiency_percent" in result.stdout


def test_init_of_an_unknown_instrument_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["init", "nope", "-o", str(tmp_path / "c.yaml")])
    assert result.exit_code != 0


def test_materials_lists_the_assumptions():
    result = runner.invoke(app, ["materials"])
    assert result.exit_code == 0
    assert "ASSUMPTION-flagged" in result.stdout


@pytest.mark.slow
def test_sweep_writes_a_table(tmp_path):
    cfg = tmp_path / "c.yaml"
    runner.invoke(app, ["init", "saw_sorter", "-o", str(cfg)])
    # Trim the population so the sweep is quick.
    text = cfg.read_text().replace("count: 300", "count: 20")
    cfg.write_text(text)

    result = runner.invoke(
        app,
        ["sweep", str(cfg), "-p", "voltage_pp=5 V,15 V", "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "out" / "ctc_vs_rbc_sweep.csv").exists()
    assert (tmp_path / "out" / "ctc_vs_rbc_sweep.nc").exists()
