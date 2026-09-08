"""Measuring the acoustic field from bead trajectories.

Two things are being checked here. One is that the calibration recovers a
planted field strength, which makes it usable on real data. The other is more
valuable: the closed-form trajectory is an **independent analytic solution** of
the same equation the numerical tracker integrates, so comparing them checks the
whole trajectory pipeline over its entire path rather than at a single instant.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core.materials import POLYSTYRENE_BEAD, WATER
from biosim_lab.core.particles import ForceRegistry, LagrangianTracker, make_state
from biosim_lab.instruments.saw_sorter import calibration as cal
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    contrast_factor,
    primary_radiation_force_1d,
)

LAMBDA = 600e-6
NODE = 150e-6
P0 = 0.45e6
PHI = float(contrast_factor(
    POLYSTYRENE_BEAD.rho, WATER.rho, POLYSTYRENE_BEAD.kappa, WATER.kappa
))
E_AC = cal.energy_density(P0, WATER.kappa)
U_MAX = cal.acoustophoretic_velocity(
    radius=POLYSTYRENE_BEAD.r, phi=PHI, energy_density_value=E_AC,
    wavelength=LAMBDA, viscosity=WATER.mu,
)


def test_energy_density_round_trips_through_pressure():
    assert cal.pressure_amplitude(cal.energy_density(P0, WATER.kappa), WATER.kappa) == (
        pytest.approx(P0)
    )


def test_velocity_formula_equals_peak_force_over_stokes_drag():
    """u_max = F_max / (6 pi mu a) — two routes to the same number.

    The closed-form prefactor 2/9 is only correct if it is consistent with the
    force machinery used everywhere else.
    """
    volume = 4 / 3 * np.pi * POLYSTYRENE_BEAD.r**3
    x_peak = NODE + LAMBDA / 8  # where sin(2k(x - node)) = 1
    force = abs(float(primary_radiation_force_1d(
        np.array([x_peak]), p0=P0, volume=volume, kappa_f=WATER.kappa,
        wavelength=LAMBDA, phi=PHI, node_offset=NODE,
    )[0]))
    from_force = force / (6 * np.pi * WATER.mu * POLYSTYRENE_BEAD.r)
    assert from_force == pytest.approx(U_MAX, rel=1e-12)


def test_velocity_scales_with_the_square_of_the_radius():
    common = dict(phi=PHI, energy_density_value=E_AC, wavelength=LAMBDA,
                  viscosity=WATER.mu)
    small = cal.acoustophoretic_velocity(radius=2e-6, **common)
    large = cal.acoustophoretic_velocity(radius=4e-6, **common)
    assert large / small == pytest.approx(4.0)


def test_closed_form_starts_where_it_is_told_and_ends_at_the_node():
    times = np.linspace(0.0, 20.0, 200)
    path = cal.analytic_trajectory(
        times, x_initial=40e-6, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    assert path[0] == pytest.approx(40e-6)
    assert path[-1] == pytest.approx(NODE, abs=1e-9)
    assert np.all(np.diff(path) > -1e-12), "must approach the node monotonically"


def test_closed_form_works_on_both_sides_of_the_node():
    times = np.linspace(0.0, 20.0, 120)
    for start in (40e-6, 260e-6):
        path = cal.analytic_trajectory(
            times, x_initial=start, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
        )
        assert path[-1] == pytest.approx(NODE, abs=1e-9)


def test_a_particle_starting_on_the_node_stays_there():
    times = np.linspace(0.0, 5.0, 20)
    path = cal.analytic_trajectory(
        times, x_initial=NODE, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    assert np.allclose(path, NODE)


def test_closed_form_matches_the_numerical_integrator():
    """The verification that matters: two independent routes, whole trajectory.

    ``analytic_trajectory`` solves the ODE exactly with no discretisation;
    ``LagrangianTracker`` integrates it numerically through the same force
    kernel the sorter uses. Agreement over the entire path exercises the force
    formula, the mobility, the time integration and the state plumbing at once
    — far more than a single-instant comparison could.
    """
    volume = 4 / 3 * np.pi * POLYSTYRENE_BEAD.r**3

    def acoustic(state):
        out = np.zeros_like(state.x)
        out[:, 0] = primary_radiation_force_1d(
            state.x[:, 0], p0=P0, volume=volume, kappa_f=WATER.kappa,
            wavelength=LAMBDA, phi=PHI, node_offset=NODE,
        )
        return out

    registry = ForceRegistry()
    registry.register("acoustic", acoustic)
    x0 = 40e-6
    state = make_state(
        np.array([[x0, 25e-6]]), POLYSTYRENE_BEAD.r, POLYSTYRENE_BEAD.rho,
        POLYSTYRENE_BEAD.kappa, ["bead"],
    )
    tracker = LagrangianTracker(registry, lambda t, x: np.zeros_like(x), WATER.mu)
    result = tracker.run(state, (0.0, 0.5), n_samples=51, rtol=1e-11, atol=1e-16)

    times = result.trajectories["time"].values
    numeric = result.trajectories["position"].values[0, :, 0]
    exact = cal.analytic_trajectory(
        times, x_initial=x0, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    error = np.abs(numeric - exact).max() / LAMBDA
    assert error < 1e-7, f"relative path error {error:.2e}"


def test_fit_recovers_a_planted_field_strength():
    times = np.linspace(0.0, 1.5, 60)
    path = cal.analytic_trajectory(
        times, x_initial=40e-6, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    fit = cal.fit_energy_density(
        times, path, radius=POLYSTYRENE_BEAD.r, phi=PHI, wavelength=LAMBDA,
        kappa_f=WATER.kappa, viscosity=WATER.mu, node_offset=NODE,
    )
    assert fit.energy_density == pytest.approx(E_AC, rel=1e-9)
    assert fit.pressure_amplitude == pytest.approx(P0, rel=1e-9)
    assert fit.r_squared > 0.999999


def test_fit_survives_realistic_tracking_noise():
    """Sub-pixel tracking noise is the real limit on a cal."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 1.2, 40)
    path = cal.analytic_trajectory(
        times, x_initial=40e-6, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    noisy = path + rng.normal(0.0, 0.3e-6, path.shape)  # 0.3 um, ~half a pixel
    fit = cal.fit_energy_density(
        times, noisy, radius=POLYSTYRENE_BEAD.r, phi=PHI, wavelength=LAMBDA,
        kappa_f=WATER.kappa, viscosity=WATER.mu, node_offset=NODE,
    )
    assert fit.energy_density == pytest.approx(E_AC, rel=0.15)
    assert fit.u_max_stderr > 0, "a noisy fit must report a non-zero uncertainty"


def test_fit_refuses_a_track_that_carries_no_information():
    """A bead sitting on the node never moves, so its speed is unconstrained."""
    times = np.linspace(0.0, 1.0, 20)
    stuck = np.full_like(times, NODE)
    with pytest.raises(ValueError, match="unconstrained"):
        cal.fit_energy_density(
            times, stuck, radius=POLYSTYRENE_BEAD.r, phi=PHI, wavelength=LAMBDA,
            kappa_f=WATER.kappa, viscosity=WATER.mu, node_offset=NODE,
        )


def test_fit_rejects_mismatched_or_tiny_inputs():
    with pytest.raises(ValueError, match="same length"):
        cal.fit_energy_density(
            np.zeros(5), np.zeros(4), radius=1e-6, phi=0.3, wavelength=LAMBDA,
            kappa_f=WATER.kappa, viscosity=WATER.mu,
        )
    with pytest.raises(ValueError, match="at least three"):
        cal.fit_energy_density(
            np.zeros(2), np.zeros(2), radius=1e-6, phi=0.3, wavelength=LAMBDA,
            kappa_f=WATER.kappa, viscosity=WATER.mu,
        )


def test_calibrate_from_a_track_table_reports_spread_across_beads():
    """One bead looks far more precise than the calibration really is."""
    import pandas as pd

    rng = np.random.default_rng(1)
    times = np.linspace(0.0, 1.2, 30)
    rows = []
    for particle in range(6):
        path = cal.analytic_trajectory(
            times, x_initial=40e-6 + particle * 5e-6, u_max=U_MAX,
            wavelength=LAMBDA, node_offset=NODE,
        ) + rng.normal(0.0, 0.3e-6, times.size)
        rows.append(pd.DataFrame({"particle": particle, "time": times, "x": path}))
    tracks = pd.concat(rows, ignore_index=True)

    table = cal.calibrate_from_tracks(
        tracks, radius=POLYSTYRENE_BEAD.r, phi=PHI, wavelength=LAMBDA,
        kappa_f=WATER.kappa, viscosity=WATER.mu, node_offset=NODE,
    )
    assert len(table) == 6
    assert table["pressure_amplitude_MPa"].mean() == pytest.approx(P0 / 1e6, rel=0.15)
    assert table["pressure_amplitude_MPa"].std() > 0


def test_calibration_depends_on_viscosity_and_therefore_on_temperature():
    """A calibration quoted without its temperature is incomplete.

    E_ac is proportional to the viscosity used in the fit, and water's viscosity
    moves 22 % between a bench and an incubator.
    """
    from biosim_lab.core.environment import water_viscosity

    times = np.linspace(0.0, 1.5, 50)
    path = cal.analytic_trajectory(
        times, x_initial=40e-6, u_max=U_MAX, wavelength=LAMBDA, node_offset=NODE
    )
    common = dict(radius=POLYSTYRENE_BEAD.r, phi=PHI, wavelength=LAMBDA,
                  kappa_f=WATER.kappa, node_offset=NODE)
    cold = cal.fit_energy_density(times, path, viscosity=water_viscosity(25.0), **common)
    warm = cal.fit_energy_density(times, path, viscosity=water_viscosity(37.0), **common)
    ratio = warm.energy_density / cold.energy_density
    assert ratio == pytest.approx(water_viscosity(37.0) / water_viscosity(25.0), rel=1e-9)
    assert ratio < 0.8, "using the wrong temperature misreads the field by >20 %"


def test_a_bead_moving_the_wrong_way_is_rejected_not_reported_as_nan() -> None:
    """A failed fit must raise, never quietly produce a nan pressure.

    Before this was checked, heavy tracking noise could drive the fitted speed
    negative; ``sqrt`` of the resulting negative energy density returned ``nan``
    and the calibration reported a blank instead of a failure.
    """
    times = np.linspace(0.0, 1.0, 20)
    # A bead drifting away from the node at x = 0.
    positions = 10e-6 + 40e-6 * times

    with pytest.raises(ValueError, match="does not move toward"):
        cal.fit_energy_density(
            times, positions, radius=5e-6, phi=0.17, wavelength=200e-6,
            kappa_f=4.5e-10, viscosity=1.0e-3,
        )


def test_pressure_amplitude_refuses_a_negative_energy_density() -> None:
    with pytest.raises(ValueError, match="negative"):
        cal.pressure_amplitude(-1.0, 4.5e-10)


def test_the_track_table_reports_how_many_beads_were_rejected() -> None:
    """Two usable beads must not look like fifty. The reject count travels."""
    pd = pytest.importorskip("pandas")

    e_ac = cal.energy_density(0.4e6, 4.5e-10)
    k = 2.0 * np.pi / 200e-6
    u_max = (2.0 / 9.0) * (5e-6) ** 2 * 0.17 * e_ac * k / 1.0e-3
    times = np.linspace(0.0, 0.6, 25)

    frames = []
    # Both start off the antinode at -50 um, which is an equilibrium: a bead
    # parked there never moves and carries no information about the field.
    for pid, x0 in enumerate((-70e-6, -30e-6)):
        x = cal.analytic_trajectory(
            times, x_initial=x0, u_max=u_max, wavelength=200e-6
        )
        frames.append(pd.DataFrame({"particle": pid, "time": times, "x": x}))
    # A third bead that never moves toward the node: unusable.
    frames.append(pd.DataFrame({"particle": 2, "time": times, "x": 10e-6 + 40e-6 * times}))

    table = cal.calibrate_from_tracks(
        pd.concat(frames), radius=5e-6, phi=0.17, wavelength=200e-6,
        kappa_f=4.5e-10, viscosity=1.0e-3,
    )

    assert table.attrs["n_accepted"] == 2
    assert table.attrs["n_rejected"] == 1
