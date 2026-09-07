"""End-to-end behaviour of the Stage 1 instrument."""

import warnings

import numpy as np
import pytest

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter import SAWSorter, SAWSorterParams, parameter_sweep
from biosim_lab.instruments.saw_sorter.fem_model import (
    PRESSURE_PER_VOLT_ASSUMPTION,
    SAWFieldModel,
    pressure_from_voltage,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterSimulation


def _run(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return SAWSorterSimulation(params), SAWSorterSimulation(params).run()


def test_large_cells_are_separated_from_erythrocytes(sorter_params):
    _, outcome = _run(sorter_params)
    m = outcome.metrics
    assert m["efficiency_percent"] > 90, "MCF-7 must reach the pressure node"
    assert m["purity_percent"] > 75, "most RBCs must miss the collection outlet"
    assert m["enrichment_fold"] > 1.4
    per_pop = m["per_population"]
    assert per_pop["mcf7"]["collected_fraction"] > per_pop["rbc"]["collected_fraction"]


def test_every_cell_leaves_the_channel(sorter_params):
    """Mass conservation: no cell may be lost or stuck inside the device."""
    _, outcome = _run(sorter_params)
    assert outcome.metrics["all_cells_exited"]
    assert len(outcome.cells) == sum(p.count for p in sorter_params.populations)
    assert outcome.cells["x_outlet_m"].between(
        0.0, sorter_params.channel_width
    ).all()


def test_turning_the_field_off_stops_separation(sorter_params):
    off = sorter_params.model_copy(
        update={"pressure_amplitude": 1.0, "voltage_pp": None}
    )
    _, outcome = _run(off)
    per_pop = outcome.metrics["per_population"]
    assert per_pop["mcf7"]["mean_abs_displacement_um"] < 1.0
    assert per_pop["rbc"]["mean_abs_displacement_um"] < 1.0
    # Sheath-focused cells start at the walls; with no field none of them can
    # reach the central collection outlet, so nothing is collected at all.
    assert outcome.metrics["n_collected"] == 0


def test_separation_improves_with_drive_voltage(sorter_params):
    low = sorter_params.model_copy(update={"voltage_pp": 4.0})
    high = sorter_params.model_copy(update={"voltage_pp": 15.0})
    _, out_low = _run(low)
    _, out_high = _run(high)
    assert (out_high.metrics["per_population"]["mcf7"]["collected_fraction"]
            > out_low.metrics["per_population"]["mcf7"]["collected_fraction"])


def test_faster_flow_reduces_capture(sorter_params):
    slow = sorter_params.model_copy(update={"flow_rate": 5e-9 / 60})
    fast = sorter_params.model_copy(update={"flow_rate": 60e-9 / 60})
    _, out_slow = _run(slow)
    _, out_fast = _run(fast)
    assert (out_fast.metrics["per_population"]["mcf7"]["collected_fraction"]
            <= out_slow.metrics["per_population"]["mcf7"]["collected_fraction"])


def test_multi_node_operating_point_warns(sorter_params):
    """20 MHz in a 300 um channel gives three nodes; the user must be told."""
    params = sorter_params.model_copy(update={"frequency": 20e6})
    with pytest.warns(RegimeWarning, match="pressure nodes"):
        SAWSorterSimulation(params).run()


def test_config_requires_a_target_population():
    with pytest.raises(ValueError, match="target=True"):
        SAWSorterParams(populations=[{"cell_type": "rbc", "count": 10, "target": False}])


def test_config_requires_a_drive_level():
    with pytest.raises(ValueError, match="pressure_amplitude or voltage_pp"):
        SAWSorterParams(voltage_pp=None, pressure_amplitude=None)


def test_voltage_calibration_is_the_documented_assumption():
    assert pressure_from_voltage(15.0) == pytest.approx(0.45e6)
    assert pytest.approx(0.45e6 / 15.0) == PRESSURE_PER_VOLT_ASSUMPTION


def test_instrument_contract_end_to_end(sorter_config):
    inst = SAWSorter(sorter_config)
    inst.setup()
    inst.setup()  # idempotent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        result = inst.run()
    assert result is inst.results()
    assert "efficiency_percent" in result.metrics
    assert "position" in result.fields
    assert result.table is not None and len(result.table) == 120
    assert result.meta["instrument"] == "saw_sorter"
    assert result.meta["config_hash"] == sorter_config.hash


@pytest.mark.slow
def test_fem_and_analytic_modes_agree_on_the_ordering_not_the_magnitude(sorter_params):
    """The two modes must rank the populations the same way.

    They are not expected to give the same numbers: the closed-form model applies
    the substrate-plane force at every height, while the FEM field decays with
    distance from the transducer, so the analytic mode is a best-case estimate.
    ``examples/04_validate_analytic_vs_fem.py`` quantifies the gap. What must
    hold in both is the physics that makes the device work — large cells focus,
    small ones do not — and the FEM mode must never be the *more* optimistic of
    the two.
    """
    analytic = sorter_params.model_copy(update={"mode": "analytic"})
    fem = sorter_params.model_copy(
        update={"mode": "fem", "fem_resolution": 32, "fem_grid": (161, 25)}
    )
    _, out_a = _run(analytic)
    _, out_f = _run(fem)
    for out, name in ((out_a, "analytic"), (out_f, "fem")):
        pop = out.metrics["per_population"]
        assert pop["mcf7"]["collected_fraction"] > pop["rbc"]["collected_fraction"] + 0.3, (
            f"{name} mode failed to separate the populations"
        )
        assert out.metrics["all_cells_exited"], f"{name} mode stranded cells in the channel"
    assert (out_f.metrics["efficiency_percent"]
            <= out_a.metrics["efficiency_percent"] + 1e-9), (
        "the FEM model resolves the vertical decay of the field, so it cannot predict "
        "a higher recovery than the substrate-plane closed form"
    )


@pytest.mark.slow
def test_vertical_fem_force_is_off_by_default_because_it_strands_cells(sorter_params):
    """Enabling the vertical Gor'kov force without a lift force traps cells at a wall."""
    fem = sorter_params.model_copy(
        update={"mode": "fem", "fem_resolution": 32, "fem_grid": (161, 25)}
    )
    assert fem.enable_vertical_arf is False
    _, baseline = _run(fem)
    assert baseline.metrics["all_cells_exited"]

    with_vertical = fem.model_copy(update={"enable_vertical_arf": True})
    _, out = _run(with_vertical)
    y_out = out.cells["y_outlet_m"].to_numpy()
    assert np.median(y_out) > 0.8 * sorter_params.channel_height, (
        "the vertical force should drive cells towards the low-pressure ceiling"
    )


@pytest.mark.slow
def test_parameter_sweep_produces_a_gridded_dataset(sorter_params):
    small = SAWSorterParams.model_validate(
        {**sorter_params.model_dump(),
         "populations": [
             {"cell_type": "mcf7", "count": 25, "target": True},
             {"cell_type": "rbc", "count": 25, "target": False},
         ],
         "n_time_samples": 21}
    )
    df, ds = parameter_sweep(
        small, {"voltage_pp": [5.0, 15.0], "flow_rate": [5e-9 / 60, 20e-9 / 60]}
    )
    assert len(df) == 4
    assert set(ds["efficiency_percent"].dims) == {"voltage_pp", "flow_rate"}
    assert ds["efficiency_percent"].shape == (2, 2)
    assert np.isfinite(ds["efficiency_percent"].values).all()


def test_field_model_reports_node_geometry():
    from biosim_lab.core.materials import LINBO3_128YX, WATER

    model = SAWFieldModel(
        frequency=6.632e6, channel_width=300e-6, channel_height=50e-6,
        fluid=WATER, substrate=LINBO3_128YX, pressure_amplitude=4.5e5, resolution=24,
    )
    assert model.wavelength == pytest.approx(3979.0 / 6.632e6)
    assert model.n_nodes_in_channel == 1
    assert np.degrees(model.rayleigh_angle_rad) == pytest.approx(22.1, abs=0.3)


@pytest.mark.slow
def test_fem_field_puts_the_pressure_node_where_it_was_asked_to():
    from biosim_lab.core.materials import LINBO3_128YX, WATER

    model = SAWFieldModel(
        frequency=6.632e6, channel_width=300e-6, channel_height=50e-6,
        fluid=WATER, substrate=LINBO3_128YX, pressure_amplitude=4.5e5, resolution=40,
    )
    result = model.solve(nx=201, ny=25)
    grid = result.grid
    p = np.hypot(grid["p_real"].values, grid["p_imag"].values)
    profile = p[:, p.shape[1] // 2]
    x_min = grid["x"].values[np.argmin(profile)]
    assert x_min == pytest.approx(150e-6, abs=8e-6)
    assert result.diagnostics["n_pressure_nodes"] == 1
