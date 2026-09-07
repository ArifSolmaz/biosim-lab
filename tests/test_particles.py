"""Force registry and Lagrangian integration."""

import numpy as np
import pytest

from biosim_lab.core.particles import (
    ForceRegistry,
    LagrangianTracker,
    make_state,
)
from biosim_lab.core.plugin import RegimeWarning


def _zero_flow(t, x):
    return np.zeros_like(x)


def test_registry_sums_only_enabled_forces():
    reg = ForceRegistry()
    reg.register("a", lambda s: np.full_like(s.x, 1e-12))
    reg.register("b", lambda s: np.full_like(s.x, 2e-12))
    state = make_state(np.zeros((3, 2)), 5e-6, 1050.0, 4e-10, "x")
    assert np.allclose(reg.total(state), 3e-12)
    reg.disable("b")
    assert reg.active == ["a"]
    assert np.allclose(reg.total(state), 1e-12)
    with pytest.raises(KeyError):
        reg.enable("missing")


def test_particle_number_is_conserved():
    reg = ForceRegistry()
    reg.register("push", lambda s: np.full_like(s.x, 1e-13))
    state = make_state(
        np.column_stack([np.linspace(10e-6, 290e-6, 50), np.full(50, 25e-6)]),
        5e-6, 1050.0, 4e-10, ["a"] * 50,
    )
    tracker = LagrangianTracker(reg, _zero_flow, 1e-3,
                                bounds=((0.0, 300e-6), (0.0, 50e-6)))
    result = tracker.run(state, (0.0, 1.0), n_samples=11)
    assert result.trajectories.sizes["particle"] == 50
    assert len(result.final) == 50
    assert np.isfinite(result.trajectories["position"].values).all()


def test_particles_stay_inside_the_channel():
    reg = ForceRegistry()
    reg.register("wall_ward", lambda s: np.full_like(s.x, 5e-12))
    state = make_state(np.array([[290e-6, 45e-6]]), 5e-6, 1050.0, 4e-10, ["a"])
    tracker = LagrangianTracker(reg, _zero_flow, 1e-3,
                                bounds=((0.0, 300e-6), (0.0, 50e-6)))
    pos = tracker.run(state, (0.0, 5.0), n_samples=51).trajectories["position"].values
    assert pos[..., 0].min() >= -1e-12
    assert pos[..., 0].max() <= 300e-6 + 1e-12
    assert pos[..., 1].min() >= -1e-12
    assert pos[..., 1].max() <= 50e-6 + 1e-12


def test_overdamped_particle_reaches_terminal_velocity():
    """v = F / (6 pi mu r) — the defining property of the overdamped limit."""
    force = 1e-13
    radius, mu = 5e-6, 1e-3
    reg = ForceRegistry()
    reg.register("const", lambda s: np.tile([force, 0.0], (s.n, 1)))
    state = make_state(np.array([[0.0, 0.0]]), radius, 1050.0, 4e-10, ["a"])
    tracker = LagrangianTracker(reg, _zero_flow, mu)
    result = tracker.run(state, (0.0, 1.0), n_samples=3)
    displacement = result.final["x_final_m"][0] - result.final["x_initial_m"][0]
    assert displacement == pytest.approx(force / (6 * np.pi * mu * radius), rel=1e-4)


def test_inertial_mode_agrees_with_overdamped_for_tiny_stokes_number():
    force = 1e-13
    reg = ForceRegistry()
    reg.register("const", lambda s: np.tile([force, 0.0], (s.n, 1)))
    state = make_state(np.array([[0.0, 0.0]]), 5e-6, 1050.0, 4e-10, ["a"])
    over = LagrangianTracker(reg, _zero_flow, 1e-3, mode="overdamped")
    inert = LagrangianTracker(reg, _zero_flow, 1e-3, mode="inertial")
    a = over.run(state, (0.0, 1.0), n_samples=3).final["x_final_m"][0]
    b = inert.run(state, (0.0, 1.0), n_samples=3).final["x_final_m"][0]
    assert b == pytest.approx(a, rel=1e-3)


def test_regime_check_warns_for_a_heavy_slow_ensemble():
    reg = ForceRegistry()
    reg.register("none", lambda s: np.zeros_like(s.x))
    state = make_state(np.array([[0.0, 0.0]]), 500e-6, 8000.0, 1e-11, ["a"])
    tracker = LagrangianTracker(reg, _zero_flow, 1e-3)
    with pytest.warns(RegimeWarning, match="Stokes number"):
        tracker.check_regime(state, transit_time=1e-4)


def test_relaxation_time_formula():
    reg = ForceRegistry()
    state = make_state(np.array([[0.0, 0.0]]), 9e-6, 1068.0, 4e-10, ["a"])
    tracker = LagrangianTracker(reg, _zero_flow, 0.89e-3)
    tau = tracker.relaxation_time(state)[0]
    assert tau == pytest.approx(2 * 1068.0 * 9e-6**2 / (9 * 0.89e-3))


def test_rejects_a_backwards_time_span():
    reg = ForceRegistry()
    state = make_state(np.array([[0.0, 0.0]]), 5e-6, 1050.0, 4e-10, ["a"])
    tracker = LagrangianTracker(reg, _zero_flow, 1e-3)
    with pytest.raises(ValueError, match="increasing"):
        tracker.run(state, (1.0, 0.0))
