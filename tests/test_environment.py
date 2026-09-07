"""Temperature-dependent fluid properties and the thermal budget."""

import numpy as np
import pytest

from biosim_lab.core import environment as env
from biosim_lab.core.materials import CELL_CULTURE_MEDIUM, WATER
from biosim_lab.core.plugin import RegimeWarning


def test_correlations_reproduce_the_library_values_at_25c():
    """The transcription check: these are the numbers materials.py already quotes."""
    assert env.water_density(25.0) == pytest.approx(997.05, abs=0.02)
    assert env.water_sound_speed(25.0) == pytest.approx(1496.7, abs=0.1)
    assert env.water_viscosity(25.0) == pytest.approx(0.890e-3, rel=0.005)
    assert env.water_compressibility(25.0) == pytest.approx(4.477e-10, rel=0.002)


def test_correlations_reproduce_body_temperature_reference_values():
    assert env.water_density(37.0) == pytest.approx(993.3, abs=0.1)
    assert env.water_sound_speed(37.0) == pytest.approx(1523.7, abs=0.5)
    assert env.water_viscosity(37.0) == pytest.approx(0.6913e-3, rel=0.01)


def test_water_density_peaks_at_four_degrees():
    """The famous anomaly — a strong check that the polynomial is right."""
    temperatures = np.linspace(0.0, 15.0, 151)
    densities = np.array([env.water_density(t) for t in temperatures])
    assert temperatures[np.argmax(densities)] == pytest.approx(4.0, abs=0.2)


def test_viscosity_falls_and_sound_speed_rises_with_temperature():
    for t in range(5, 90, 10):
        assert env.water_viscosity(t + 5) < env.water_viscosity(t)
    # Sound speed in water is non-monotonic: it peaks near 74 C.
    assert env.water_sound_speed(74.0) > env.water_sound_speed(25.0)
    assert env.water_sound_speed(74.0) > env.water_sound_speed(95.0)


def test_extrapolating_outside_the_fitted_range_warns():
    with pytest.warns(RegimeWarning, match="outside the range"):
        env.water_viscosity(150.0)


def test_fluid_at_reference_temperature_is_a_no_op():
    same = env.fluid_at(WATER, WATER.temperature_K - env.KELVIN)
    assert same.rho == pytest.approx(WATER.rho, rel=1e-9)
    assert same.c == pytest.approx(WATER.c, rel=1e-9)
    assert same.mu == pytest.approx(WATER.mu, rel=1e-9)


def test_fluid_at_preserves_the_solute_offset():
    """A medium is not water; moving its temperature must not turn it into water.

    The invariant `fluid_at` actually guarantees is that the *ratio* to water is
    unchanged, so that is what is checked. Asserting a sign instead would be
    wrong: PBS carries a clear positive sound-speed offset from its salt, while
    the DMEM entry is essentially "water at 37 C" and carries none.
    """
    from biosim_lab.core.materials import PBS

    for fluid in (PBS, CELL_CULTURE_MEDIUM):
        reference_c = fluid.temperature_K - env.KELVIN
        offset_rho = fluid.rho / env.water_density(reference_c)
        offset_c = fluid.c / env.water_sound_speed(reference_c)
        offset_mu = fluid.mu / env.water_viscosity(reference_c)

        for target in (4.0, 25.0, 37.0):
            moved = env.fluid_at(fluid, target)
            assert moved.rho / env.water_density(target) == pytest.approx(offset_rho)
            assert moved.c / env.water_sound_speed(target) == pytest.approx(offset_c)
            assert moved.mu / env.water_viscosity(target) == pytest.approx(offset_mu)

    # PBS really is saltier than water, and that must survive the move.
    assert env.fluid_at(PBS, 4.0).rho > env.water_density(4.0)
    assert env.fluid_at(PBS, 4.0).c > env.water_sound_speed(4.0)


def test_derived_fluids_are_flagged_as_derived():
    warm = env.fluid_at(WATER, 37.0)
    assert warm.density.prov.is_assumption
    assert "derived" in warm.density.prov.assumption


def test_warming_to_body_temperature_speeds_migration_by_a_quarter():
    """The headline consequence: a device tuned on the bench is not the same
    device in an incubator."""
    effect = env.describe_temperature_effect(WATER, 25.0, 37.0)
    assert effect["viscosity_ratio"] == pytest.approx(0.777, rel=0.02)
    assert effect["acoustophoretic_speed_ratio"] == pytest.approx(1.25, rel=0.03)


def test_absorption_is_quadratic_in_frequency():
    a1 = env.absorption_coefficient(10e6)
    a2 = env.absorption_coefficient(20e6)
    assert a2 / a1 == pytest.approx(4.0, rel=1e-6)
    # ~10 Np/m at 20 MHz: 0.88 dB/cm.
    assert a2 == pytest.approx(10.1, rel=0.05)


def test_water_absorbs_a_negligible_fraction_across_the_channel():
    """Justifies treating the fluid as lossless in the Helmholtz solver."""
    alpha = env.absorption_coefficient(6.632e6)
    assert alpha * 300e-6 < 1e-3


def test_bulk_heating_is_negligible_and_transducer_heating_is_not():
    budget = env.thermal_budget(
        pressure_amplitude=0.45e6, frequency=6.632e6, fluid=WATER,
        channel_width=300e-6, channel_height=50e-6, channel_length=2e-3,
        flow_rate=5e-9 / 60, rf_power=0.25,
    )
    assert budget["bulk_absorption_K"] < 0.05, "sound absorbed by the water is tiny"
    assert budget["transducer_K"] > 100 * budget["bulk_absorption_K"], (
        "the transducer is the dominant heat source, which is the point"
    )
    assert budget["total_rise_K"] == pytest.approx(
        budget["bulk_absorption_K"] + budget["transducer_K"]
    )


def test_heating_scales_with_pressure_squared():
    common = dict(
        frequency=6.632e6, fluid=WATER, channel_width=300e-6, channel_height=50e-6,
        channel_length=2e-3, flow_rate=5e-9 / 60,
    )
    low = env.thermal_budget(pressure_amplitude=0.2e6, **common)
    high = env.thermal_budget(pressure_amplitude=0.4e6, **common)
    assert high["bulk_absorption_K"] / low["bulk_absorption_K"] == pytest.approx(4.0, rel=1e-6)


def test_faster_flow_carries_the_heat_away():
    common = dict(
        pressure_amplitude=0.45e6, frequency=6.632e6, fluid=WATER,
        channel_width=300e-6, channel_height=50e-6, channel_length=2e-3,
    )
    slow = env.thermal_budget(flow_rate=5e-9 / 60, **common)
    fast = env.thermal_budget(flow_rate=50e-9 / 60, **common)
    assert fast["bulk_absorption_K"] == pytest.approx(slow["bulk_absorption_K"] / 10, rel=1e-6)


def test_celsius_configuration_strings_are_accepted():
    from biosim_lab.core.units import to_si

    assert to_si("37 degC", "K") == pytest.approx(310.15)
    assert to_si("4 degC", "K") == pytest.approx(277.15)
    assert to_si("98.6 degF", "K") == pytest.approx(310.15, abs=0.1)
