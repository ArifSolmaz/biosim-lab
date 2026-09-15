"""Stage 2: impedance model, Cell Index and IC50."""

import numpy as np
import pytest

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.instruments.impedance_rtca import ImpedanceRTCA
from biosim_lab.instruments.impedance_rtca.instrument import logistic_coverage
from biosim_lab.instruments.impedance_rtca.physics import (
    ElectrodeGeometry,
    alpha_parameter,
    cell_index,
    fit_dose_response,
    four_parameter_logistic,
    giaever_keese_impedance,
    naked_electrode_impedance,
    solution_resistance,
    well_impedance,
)

FREQ = np.logspace(2, 7, 60)
GEOM = ElectrodeGeometry()


def test_blank_electrode_impedance_is_physically_plausible():
    """A ~1 mm^2 gold electrode in saline reads hundreds of ohms at 10 kHz."""
    z = naked_electrode_impedance(np.array([1e4]), area_cm2=GEOM.area_cm2,
                                  conductivity=1.4)[0]
    assert 50.0 < abs(z) < 5000.0
    assert z.imag < 0, "a capacitive interface must have negative reactance"


def test_impedance_falls_monotonically_with_frequency():
    z = naked_electrode_impedance(FREQ, area_cm2=GEOM.area_cm2, conductivity=1.4)
    assert np.all(np.diff(np.abs(z)) < 0)


def test_spreading_resistance_matches_the_disc_formula():
    area_cm2 = 1e-2
    a_m = np.sqrt(area_cm2 * 1e-4 / np.pi)
    assert solution_resistance(area_cm2, conductivity=1.5) == pytest.approx(
        1 / (4 * 1.5 * a_m)
    )


def test_cell_coverage_raises_impedance_monotonically():
    coverages = [0.0, 0.25, 0.5, 0.75, 1.0]
    z = [abs(well_impedance(np.array([1e4]), coverage=c, geometry=GEOM)[0])
         for c in coverages]
    assert all(b > a for a, b in zip(z, z[1:])), z


def test_cell_index_is_zero_for_a_blank_well_and_positive_when_covered():
    z0 = well_impedance(np.array([1e4]), coverage=0.0, geometry=GEOM)[0]
    z1 = well_impedance(np.array([1e4]), coverage=0.9, geometry=GEOM)[0]
    assert float(cell_index(z0, z0)) == pytest.approx(0.0, abs=1e-12)
    assert float(cell_index(z1, z0)) > 0


def test_tighter_junctions_raise_the_impedance():
    """Rb is the paracellular resistance: more of it means less leak current."""
    leaky = abs(giaever_keese_impedance(
        np.array([1e4]), z_naked_specific=np.array([1.0 + 0j]), rb=0.5, alpha=2.0,
        specific_capacitance=1e-6)[0])
    tight = abs(giaever_keese_impedance(
        np.array([1e4]), z_naked_specific=np.array([1.0 + 0j]), rb=20.0, alpha=2.0,
        specific_capacitance=1e-6)[0])
    assert tight > leaky


def test_alpha_parameter_scales_as_expected():
    """alpha = r * sqrt(rho/h): halving the gap raises alpha by sqrt(2)."""
    a1 = alpha_parameter(8e-6, 100e-9, 1 / 1.4)
    a2 = alpha_parameter(8e-6, 50e-9, 1 / 1.4)
    assert a2 / a1 == pytest.approx(np.sqrt(2.0))


def test_logistic_growth_respects_lag_and_capacity():
    t = np.linspace(0, 2e5, 200)
    cov = logistic_coverage(t, seeding=0.05, capacity=0.9, doubling_time=7e4, lag=7200.0)
    assert cov[0] == pytest.approx(0.05)
    assert np.all(cov[t < 7200] == pytest.approx(0.05))
    assert np.all(np.diff(cov) >= -1e-12), "growth must be monotone"
    assert cov[-1] < 0.9 + 1e-9


def test_four_parameter_logistic_shape():
    ic50 = 2.0
    assert four_parameter_logistic(ic50, 0.0, 1.0, ic50, 1.0) == pytest.approx(0.5)
    assert four_parameter_logistic(1e-6, 0.0, 1.0, ic50, 1.0) == pytest.approx(1.0, abs=1e-3)
    assert four_parameter_logistic(1e6, 0.0, 1.0, ic50, 1.0) == pytest.approx(0.0, abs=1e-3)


def test_ic50_is_recovered_from_noiseless_data():
    c = np.logspace(-3, 2, 12)
    y = four_parameter_logistic(c, 0.05, 1.0, 1.3, 1.4)
    fit = fit_dose_response(c, y)
    assert fit.ic50 == pytest.approx(1.3, rel=1e-3)
    assert fit.hill == pytest.approx(1.4, rel=1e-3)
    assert fit.r_squared > 0.999


def test_ic50_fit_needs_enough_points():
    with pytest.raises(ValueError, match="at least 4"):
        fit_dose_response(np.array([1.0, 2.0]), np.array([1.0, 0.5]))


def test_instrument_end_to_end_recovers_the_planted_ic50():
    cfg = ExperimentConfig.model_validate(ImpedanceRTCA.example_config())
    inst = ImpedanceRTCA(cfg)
    result = inst.run()
    m = result.metrics
    assert m["max_cell_index"] > 1.0
    assert m["fit_r_squared"] > 0.95
    # The apparent IC50 from a fixed endpoint is exposure-time dependent, so a
    # factor-of-two window is the honest tolerance here, not a tight one.
    assert 0.5 < m["ic50"] / 1.0 < 2.0
    assert "cell_index" in result.fields
    assert "impedance_real" in result.fields
    assert result.table is not None and "normalised_response" in result.table


def test_instrument_reads_a_real_rtca_export(tmp_path):
    path = tmp_path / "export.csv"
    lines = ["RTCA export", "Time (h),A1,A2"]
    for i in range(12):
        lines.append(f"{i * 0.5},{1 + 0.2 * i},{1 + 0.1 * i}")
    path.write_text("\n".join(lines), encoding="utf-8")

    cfg = ExperimentConfig(
        name="measured", instrument="impedance_rtca",
        params={"source_file": str(path)},
    )
    result = ImpedanceRTCA(cfg).run()
    assert result.metrics["n_wells"] == 2
    assert result.meta["source"] == str(path)
    assert result.metrics["duration_h"] == pytest.approx(5.5)


def _config(**params):
    cfg = ExperimentConfig.model_validate(ImpedanceRTCA.example_config())
    return cfg.model_copy(update={"params": {**cfg.params, **params}})


def test_the_default_electrode_is_interdigitated_and_reports_its_constants():
    m = ImpedanceRTCA(_config()).run().metrics
    assert m["electrode"] == "interdigitated"
    assert m["cell_constant_1_per_m"] > 0
    assert m["solution_resistance_ohm"] < m["blank_impedance_ohm"]


def test_fem_and_lumped_agree_at_the_readout_frequency():
    """At 10 kHz the interface dominates, so the Cell Index is model-independent."""
    lumped = ImpedanceRTCA(_config(spectrum_frequencies=12)).run()
    fem = ImpedanceRTCA(_config(field_model="fem", spectrum_frequencies=12)).run()
    assert fem.metrics["readout_lumped_vs_fem_max_percent"] < 1.0
    assert fem.metrics["max_cell_index"] == pytest.approx(
        lumped.metrics["max_cell_index"], rel=0.02)
    assert "impedance_lumped_real" in fem.fields
    assert fem.metrics["spectrum_lumped_vs_fem_max_percent"] > 1.0


def test_fem_needs_the_interdigitated_electrode():
    with pytest.raises(ValueError, match="interdigitated"):
        ImpedanceRTCA(_config(field_model="fem", electrode="disc")).run()


def test_disc_electrode_path_matches_the_scalar_model():
    """The vectorised coverage path reproduces well_impedance exactly."""
    inst = ImpedanceRTCA(_config(electrode="disc"))
    inst.setup()
    cov = np.array([0.0, 0.3, 0.9])
    fast = np.abs(inst._lumped(np.array([1e4]), cov))
    slow = [abs(well_impedance(np.array([1e4]), coverage=c, geometry=inst.params.geometry)[0])
            for c in cov]
    np.testing.assert_allclose(fast, slow, rtol=1e-12)


def test_normalized_cell_index_is_one_at_the_normalisation_time():
    result = ImpedanceRTCA(_config()).run()
    nci = result.fields["normalized_cell_index"]
    t_norm = nci.attrs["normalised_at_s"]
    np.testing.assert_allclose(nci.sel(time=t_norm).values, 1.0)


def test_a_measured_plate_with_a_layout_gets_an_ic50(tmp_path):
    """Export a simulated plate the way an instrument would, read it back, fit it."""
    sim = ImpedanceRTCA(_config(noise_cv=0.0)).run()
    ci = sim.fields["cell_index"]
    wells = [str(w) for w in ci["well"].values]
    export = tmp_path / "plate.csv"
    header = "Time (Hour)," + ",".join(f"{w[0]}{int(w[1:]):02d}" for w in wells)  # A01 style
    rows = [header] + [
        f"{t / 3600:.4f}," + ",".join(f"{v:.6f}" for v in ci.sel(time=t).values)
        for t in ci["time"].values
    ]
    export.write_text("\n".join(rows), encoding="utf-8")
    layout = tmp_path / "layout.csv"
    table = sim.table[["well", "concentration"]]
    table.to_csv(layout, index=False)

    measured = ImpedanceRTCA(ExperimentConfig(
        name="measured", instrument="impedance_rtca",
        params={"source_file": str(export), "plate_layout_file": str(layout),
                "treatment_time": "24 h"},
    )).run()
    assert measured.metrics["ic50"] == pytest.approx(sim.metrics["ic50"], rel=1e-3)
    assert "normalized_cell_index" in measured.fields


def test_a_measured_plate_without_a_layout_has_no_ic50(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("Time (h),A1,A2\n0,0.1,0.1\n1,0.5,0.4\n2,1.0,0.8\n", encoding="utf-8")
    m = ImpedanceRTCA(ExperimentConfig(
        name="m", instrument="impedance_rtca", params={"source_file": str(path)})).run().metrics
    assert "ic50" not in m
