"""Cell damage: thermal dose, shear and cavitation."""

import numpy as np
import pytest

from biosim_lab.core.materials import WATER
from biosim_lab.instruments.saw_sorter import viability as vb
from biosim_lab.instruments.saw_sorter.flow import RectangularPoiseuille


def test_cem43_is_one_minute_at_43_degrees():
    """The definition: the scale is calibrated so 43 C accumulates 1:1."""
    assert float(vb.cem43(43.0, 60.0)) == pytest.approx(1.0)


def test_cem43_halves_per_degree_above_43_and_quarters_below():
    at_43 = float(vb.cem43(43.0, 60.0))
    assert float(vb.cem43(44.0, 60.0)) == pytest.approx(at_43 / 0.5)
    assert float(vb.cem43(42.0, 60.0)) == pytest.approx(at_43 * 0.25)
    assert float(vb.cem43(41.0, 60.0)) == pytest.approx(at_43 * 0.25**2)


def test_cem43_is_linear_in_time():
    assert float(vb.cem43(45.0, 120.0)) == pytest.approx(2 * float(vb.cem43(45.0, 60.0)))


def test_thermal_survival_halves_at_the_threshold():
    assert float(vb.thermal_survival(vb.CEM43_DAMAGE_THRESHOLD_MIN)) == pytest.approx(0.5)
    assert float(vb.thermal_survival(0.0)) == pytest.approx(1.0)
    assert float(vb.thermal_survival(1e4)) == pytest.approx(0.0, abs=1e-9)


def test_a_short_transit_is_harmless_even_when_hot():
    """The key result for a flow-through device: exposure is 0.36 s, not minutes.

    A trap that holds cells for ten minutes at the same temperature is a
    completely different proposition, and this is why.
    """
    flow_through = float(vb.thermal_survival(vb.cem43(50.0, 0.36)))
    trapped = float(vb.thermal_survival(vb.cem43(50.0, 600.0)))
    assert flow_through > 0.95
    assert trapped < 0.01


def test_wall_shear_in_the_default_channel_is_far_below_lysis():
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, 5e-9 / 60)
    shear = vb.wall_shear_stress(flow)
    assert 0.1 < shear["peak_wall_shear_Pa"] < 5.0
    margin = vb.SHEAR_LYSIS_THRESHOLD_PA / shear["peak_wall_shear_Pa"]
    assert margin > 50, f"expected a large safety margin, got {margin:.0f}x"


def test_shear_scales_with_flow_rate():
    slow = vb.wall_shear_stress(RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, 5e-9 / 60))
    fast = vb.wall_shear_stress(RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, 50e-9 / 60))
    assert fast["peak_wall_shear_Pa"] / slow["peak_wall_shear_Pa"] == pytest.approx(
        10.0, rel=0.02
    )


def test_shear_is_lowest_where_the_cells_are_focused():
    """A focused cell sits at the centre, which is the gentlest place in the channel."""
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, 5e-9 / 60)
    centre = vb.shear_at(flow, np.array([150e-6]), np.array([25e-6]))
    near_wall = vb.shear_at(flow, np.array([150e-6]), np.array([2e-6]))
    assert centre[0] < 0.05 * near_wall[0]


def test_shear_survival_halves_at_the_reference_exposure():
    s = vb.shear_survival(vb.SHEAR_LYSIS_THRESHOLD_PA, 120.0)
    assert float(s) == pytest.approx(0.5)


def test_mechanical_index_definition_and_margin():
    mi = vb.mechanical_index(0.45e6, 6.632e6)
    assert mi == pytest.approx(0.45 / np.sqrt(6.632), rel=1e-9)
    assert mi < vb.MECHANICAL_INDEX_LIMIT
    # It rises with pressure and falls with frequency.
    assert vb.mechanical_index(0.9e6, 6.632e6) == pytest.approx(2 * mi)
    assert vb.mechanical_index(0.45e6, 26.528e6) == pytest.approx(mi / 2)


def test_acoustic_intensity_scales_with_pressure_squared():
    a = vb.acoustic_intensity(0.2e6, WATER)
    b = vb.acoustic_intensity(0.4e6, WATER)
    assert b / a == pytest.approx(4.0)


def _report(temperature_c: float, seconds: float, shear: float, inlet: float = 0.95):
    rng = np.random.default_rng(0)
    n = 4000
    return vb.assess(
        temperature_c=temperature_c,
        residence_time_s=np.full(n, seconds),
        shear_stress_pa=np.full(n, shear),
        pressure_amplitude=0.45e6,
        frequency=6.632e6,
        fluid=WATER,
        inlet_viability=inlet,
        rng=rng,
    )


def test_the_device_kills_nobody_at_the_default_operating_point():
    r = _report(25.0, 0.36, 0.66)
    assert r.viability_in_percent == pytest.approx(95.0, abs=2.0)
    assert r.killed_by_device_percent == pytest.approx(0.0, abs=0.5)
    assert r.indicators["thermal_margin"] > 1e3
    assert r.indicators["shear_margin"] > 50
    assert r.indicators["cavitation_margin"] > 5


def test_inlet_viability_is_carried_through_not_invented():
    """A 70 %-viable sample must not come out looking 100 % viable."""
    r = _report(25.0, 0.36, 0.66, inlet=0.70)
    assert r.viability_in_percent == pytest.approx(70.0, abs=2.0)
    assert r.viability_out_percent <= r.viability_in_percent


def test_a_hot_slow_device_does_kill_cells():
    r = _report(52.0, 60.0, 0.66)
    assert r.killed_by_device_percent > 20.0
    assert r.indicators["thermal_margin"] < 1.0


def test_dead_cells_are_never_resurrected():
    r = _report(25.0, 0.36, 0.66, inlet=0.5)
    assert not np.any(r.alive_at_outlet & ~r.alive_at_inlet)


def test_cells_that_never_exited_are_charged_the_longest_transit():
    """A NaN residence time must not become a free pass."""
    rng = np.random.default_rng(1)
    residence = np.array([1.0, 2.0, np.nan, 3.0])
    r = vb.assess(
        temperature_c=50.0, residence_time_s=residence,
        shear_stress_pa=np.full(4, 1.0), pressure_amplitude=0.45e6,
        frequency=6.632e6, fluid=WATER, inlet_viability=1.0, rng=rng,
    )
    assert np.isfinite(r.thermal_dose_cem43).all()
    assert r.thermal_dose_cem43[2] == pytest.approx(r.thermal_dose_cem43[3])


def test_dead_cell_acoustics_assumption_is_documented():
    assert "assumption" in vb.DEAD_CELL_ACOUSTICS_ASSUMPTION.lower() or len(
        vb.DEAD_CELL_ACOUSTICS_ASSUMPTION
    ) > 80
