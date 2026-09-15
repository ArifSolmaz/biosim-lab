"""Alternating-frequency BAW mode: physics, integrators, switching and the config contract.

References: Bruus 2012 (doi:10.1039/c2lc21068a) for the force, Barnkob et al.
2010 (doi:10.1039/b920376a) for the closed-form trajectory, Zhang et al. 2023
(doi:10.3390/ijms24043338) for the device and its stated rules.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from biosim_lab.core.materials import get_cell, get_fluid
from biosim_lab.core.particles import ForceRegistry, LagrangianTracker, make_state
from biosim_lab.core.plugin import ConfigurationError, RegimeWarning
from biosim_lab.instruments.saw_sorter.design import (
    operating_window,
    return_energy,
    separation_rule_energy,
)
from biosim_lab.instruments.saw_sorter.physics import baw
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    bruus_phi,
    contrast_factor,
    primary_radiation_force_1d,
)
from biosim_lab.instruments.saw_sorter.simulate import (
    Population,
    SAWSorterParams,
    SAWSorterSimulation,
)
from biosim_lab.instruments.saw_sorter.switching import Schedule

W = 737e-6
MU = 0.93e-3


def _phase(freq, dur, volts, energy):
    return dict(frequency=freq, duration=dur, voltage_pp=volts, reference_voltage_pp=volts,
                reference_energy_density=energy, energy_source="test value")


def _params(**kw):
    base = dict(
        mode="alternating_baw", channel_width=W, channel_height="50 um",
        channel_length="20 mm", flow_rate="150 uL/h", fluid="pbs", substrate="silicon",
        inlet="side", sheath_ratio=2.0, track_viability=False, seed=1,
        switching=dict(phases=[_phase("1 MHz", 0.8, "9 V", 41.0),
                               _phase("3 MHz", 1.4, "110 V", 20.0)], time_step=5e-3),
        populations=[dict(cell_type="mcf7", count=20, target=True),
                     dict(cell_type="pbmc", count=20)],
    )
    base.update(kw)
    return SAWSorterParams.model_validate(base)


# -- physics ------------------------------------------------------------------


def test_baw_force_is_the_specification_formula_with_the_origin_at_the_wall():
    """4 pi Phi_B a^3 k E sin(2ky) == -(pi p0^2 V beta / 2 lambda) Phi sin(2k(y - y_node))."""
    fluid = get_fluid("pbs")
    cell = get_cell("mcf7")
    p0 = 0.2e6
    e_ac = baw.energy_density_from_pressure(p0, fluid.rho, fluid.c)
    y = np.linspace(0, W, 57)
    for n in (1, 3):
        ours = baw.radiation_force(y, n=n, width=W, energy_density=e_ac, radius=cell.r,
                                   phi_bruus=bruus_phi(cell.rho, fluid.rho, cell.kappa,
                                                       fluid.kappa))
        spec = primary_radiation_force_1d(
            y, p0=p0, volume=4 / 3 * np.pi * cell.r**3, kappa_f=fluid.kappa,
            wavelength=2 * W / n,
            phi=contrast_factor(cell.rho, fluid.rho, cell.kappa, fluid.kappa),
            node_offset=W / (2 * n),
        )
        np.testing.assert_allclose(ours, spec, rtol=1e-10, atol=1e-25)


def test_modes_and_nodes_of_the_zhang_channel():
    c = get_fluid("pbs").c
    assert baw.mode_number(1e6, W, c) == 1
    assert baw.mode_number(3e6, W, c) == 3
    np.testing.assert_allclose(baw.node_positions(W, 3), [W / 6, W / 2, 5 * W / 6])
    np.testing.assert_allclose(baw.node_positions(W, 1), [W / 2])


def test_off_resonance_drive_warns():
    with pytest.warns(RegimeWarning, match="half-wave resonance"):
        baw.mode_number(1.5e6, W, 1500.0)  # 2 W f / c = 1.47: halfway between n = 1 and 2


def test_energy_density_scales_with_voltage_squared():
    e = baw.energy_density_from_voltage(18.0, reference_voltage_pp=9.0,
                                        reference_energy_density=10.0)
    assert e == pytest.approx(40.0)


def test_analytic_trajectory_converges_on_the_node_and_respects_antinodes():
    kw = dict(n=3, width=W, energy_density=20.0, radius=4e-6, phi_bruus=0.05, viscosity=MU)
    y = baw.analytic_position(np.array([0.05 * W, 0.30 * W, W / 3, 0.40 * W]), 30.0, **kw)
    assert y[0] == pytest.approx(W / 6, rel=1e-3)   # below the node: rises onto it
    assert y[1] == pytest.approx(W / 6, rel=1e-3)   # above it: falls onto it
    assert y[2] == pytest.approx(W / 3)             # antinode: unstable equilibrium
    assert y[3] == pytest.approx(W / 2, rel=1e-3)   # next basin


# -- integrators ---------------------------------------------------------------


def _single_mode_tracker(n=1, e_ac=30.0):
    fluid = get_fluid("pbs")
    cell = get_cell("mcf7")
    phi = float(bruus_phi(cell.rho, fluid.rho, cell.kappa, fluid.kappa))
    reg = ForceRegistry()
    reg.register("acoustic_radiation", lambda s: np.column_stack([
        baw.radiation_force(s.x[:, 0], n=n, width=W, energy_density=e_ac,
                            radius=s.radius, phi_bruus=phi), np.zeros(s.n)]))
    tracker = LagrangianTracker(reg, lambda t, x: np.zeros_like(x), fluid.mu)
    y0 = np.array([0.03, 0.10, 0.20, 0.30]) * W
    state = make_state(np.column_stack([y0, np.zeros(4)]), cell.r, cell.rho, cell.kappa, "mcf7")
    return tracker, state, phi, fluid, cell


def test_rk4_matches_the_closed_form_trajectory():
    tracker, state, phi, fluid, cell = _single_mode_tracker()
    t = np.linspace(0, 2.0, 11)
    rk = tracker.run(state, (0, 2.0), t_eval=t, integrator="rk4", dt=5e-3, check_regime=False)
    exact = baw.analytic_position(state.x[:, 0][:, None], t[None, :], n=1, width=W,
                                  energy_density=30.0, radius=cell.r, phi_bruus=phi,
                                  viscosity=fluid.mu)
    got = rk.trajectories["position"].sel(axis="x").values
    assert np.max(np.abs(got - exact)) < 1e-9  # sub-nanometre at a 5 ms step


def test_rk4_and_solve_ivp_agree_on_a_switched_run():
    """The paper's method (fixed-step RK4) against SciPy's adaptive LSODA."""
    base = dict(switching=dict(phases=[_phase("1 MHz", 0.8, "9 V", 41.0),
                                       _phase("3 MHz", 1.4, "110 V", 20.0)],
                               entry="fixed", entry_time_in_cycle=0.3, time_step=2e-3))
    rk = SAWSorterSimulation(_params(**base)).run()
    base["switching"]["integrator"] = "solve_ivp"
    iv = SAWSorterSimulation(_params(**base)).run()
    assert rk.diagnostics["integrator"] == "rk4"
    assert iv.diagnostics["integrator"] == "solve_ivp"
    dx = np.abs(rk.cells["x_outlet_m"].to_numpy() - iv.cells["x_outlet_m"].to_numpy())
    assert dx.max() < 1e-6, f"outlet positions differ by up to {dx.max() * 1e6:.3f} um"
    assert (rk.cells["outlet"] == iv.cells["outlet"]).all()


def test_rk4_step_is_converged_at_the_configured_value():
    runs = [SAWSorterSimulation(_params(switching=dict(
        phases=[_phase("1 MHz", 0.8, "9 V", 41.0), _phase("3 MHz", 1.4, "110 V", 20.0)],
        time_step=dt))).run() for dt in (1e-3, 5e-3)]
    dx = np.abs(runs[0].cells["x_outlet_m"].to_numpy() - runs[1].cells["x_outlet_m"].to_numpy())
    assert dx.max() < 0.5e-6
    assert runs[0].metrics["capture_efficiency_percent"] == \
        runs[1].metrics["capture_efficiency_percent"]


def test_breakpoint_on_a_sample_time_is_reported_once():
    tracker, state, *_ = _single_mode_tracker()
    t = np.array([0.0, 0.5, 1.0, 1.5])
    for integrator, kw in (("solve_ivp", {}), ("rk4", {"dt": 1e-2})):
        tr = tracker.run(state, (0, 1.5), t_eval=t, breakpoints=[0.5, 1.0],
                         integrator=integrator, check_regime=False, **kw)
        np.testing.assert_allclose(tr.trajectories["time"].values, t)


# -- the design rule (case f of benchmark_02) ----------------------------------


def test_design_rule_holds_in_the_integrated_trajectory():
    """During 1 MHz, MCF-7 moves more than W/6 and a PBMC less (doi:10.3390/ijms24043338)."""
    fluid = get_fluid("pbs")
    mcf7 = Population(cell_type="mcf7", target=True, compressibility=4.3e-10,
                      override_source="test").resolved_cell()
    pbmc = get_cell("pbmc")
    cal = separation_rule_energy(mcf7, pbmc, fluid, width=W, duration=0.8)
    assert cal.feasible and cal.target_minimum < cal.chosen < cal.background_maximum
    for cell, bigger in ((mcf7, True), (pbmc, False)):
        phi = float(bruus_phi(cell.rho, fluid.rho, cell.kappa, fluid.kappa))
        reg = ForceRegistry()
        reg.register("arf", lambda s, phi=phi: np.column_stack([
            baw.radiation_force(s.x[:, 0], n=1, width=W, energy_density=cal.chosen,
                                radius=s.radius, phi_bruus=phi), np.zeros(s.n)]))
        tr = LagrangianTracker(reg, lambda t, x: np.zeros_like(x), fluid.mu).run(
            make_state([[W / 6, 0.0]], cell.r, cell.rho, cell.kappa, cell.key),
            (0.0, 0.8), t_eval=np.array([0.0, 0.8]), integrator="rk4", dt=2e-3,
            check_regime=False)
        moved = float(np.ptp(tr.trajectories["position"].sel(axis="x").values))
        assert (moved > W / 6) is bigger, f"{cell.key} moved {moved * 1e6:.0f} um"


def test_return_energy_brings_a_cell_back_in_the_stated_time():
    fluid = get_fluid("pbs")
    pbmc = get_cell("pbmc")
    e3 = return_energy(pbmc, fluid, width=W, duration=1.0)
    phi = float(bruus_phi(pbmc.rho, fluid.rho, pbmc.kappa, fluid.kappa))
    half = W / 6
    y0 = W / 3 - 0.1 * half          # 90 % of the way from the W/6 node to the antinode
    y1 = baw.analytic_position(y0, 1.0, n=3, width=W, energy_density=e3, radius=pbmc.r,
                               phi_bruus=phi, viscosity=fluid.mu)
    assert abs(y1 - W / 6) == pytest.approx(0.1 * half, rel=1e-6)


def test_operating_window_reports_empty_honestly():
    import pandas as pd

    df = pd.DataFrame({"V1": [5, 7, 9], "capture_efficiency_percent": [10, 60, 99],
                       "contamination_rate_percent": [2, 8, 30]})
    win = operating_window(df, "V1")
    assert win["window"] is None
    assert win["best_setting"] == 9 and win["best_youden_percent"] == 69


# -- switching schedule --------------------------------------------------------


def test_schedule_phases_switch_times_and_timeline():
    s = Schedule(durations=np.array([0.8, 1.4]), mode_numbers=np.array([1, 3]),
                 energy_densities=np.array([40.0, 20.0]), frequencies=np.array([1e6, 3e6]),
                 labels=("1 MHz", "3 MHz"))
    assert s.period == pytest.approx(2.2)
    np.testing.assert_array_equal(s.phase_index([0.0, 0.79, 0.81, 2.19, 2.21]), [0, 0, 1, 1, 0])
    np.testing.assert_allclose(s.switch_times(0.0, 4.5), [0.8, 2.2, 3.0, 4.4])
    np.testing.assert_allclose(s.switch_times(0.0, 3.0, offsets=0.3), [0.5, 1.9, 2.7])
    tl = s.timeline(0.0, 2.2)
    assert [b["label"] for b in tl] == ["1 MHz", "3 MHz"]


def test_two_switching_cycles_are_enforced():
    """A run whose fastest cell sees < 2 cycles is refused with the three remedies."""
    fast = _params(flow_rate="900 uL/h")
    with pytest.raises(ConfigurationError, match="at least 2 are required"):
        SAWSorterSimulation(fast).run()


def test_each_cell_follows_its_own_clock():
    out = SAWSorterSimulation(_params()).run()
    sw = out.diagnostics["switching"]
    assert sw["cycles_seen"]["fastest_cell"] >= 2.0
    assert [p["mode_number"] for p in sw["phases"]] == [1, 3]
    assert out.metrics["all_cells_exited"]


def test_force_history_drag_mirrors_the_radiation_force():
    out = SAWSorterSimulation(_params(record_forces=True, switching=dict(
        phases=[_phase("1 MHz", 0.8, "9 V", 41.0), _phase("3 MHz", 1.4, "110 V", 20.0)],
        entry="fixed", time_step=5e-3))).run()
    f = out.forces["F"]
    np.testing.assert_allclose(f.sel(force="stokes_drag").values,
                               -f.sel(force="acoustic_radiation").values, atol=1e-25)


# -- config contract -----------------------------------------------------------


def test_legacy_mode_spellings_still_load():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = SAWSorterParams(voltage_pp="15 V", mode="analytic")
        q = SAWSorterParams(voltage_pp="15 V", tilt_angle_deg=-5.0)
    assert (p.mode, p.field_model) == ("ssaw", "analytic")
    assert q.mode == "tassaw"


@pytest.mark.parametrize("kw, match", [
    ({"mode": "ssaw", "tilt_angle_deg": 5.0}, "mode='tassaw'"),
    ({"mode": "tassaw"}, "non-zero tilt"),
    ({"mode": "alternating_baw"}, "switching"),
])
def test_contradictory_modes_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        SAWSorterParams(voltage_pp="15 V", **kw)


def test_overrides_need_a_source_and_carry_it():
    with pytest.raises(ValueError, match="override_source"):
        Population(cell_type="wbc", diameter="12 um")
    cell = Population(cell_type="wbc", diameter="12 um", override_doi="10.1073/pnas.1504484112",
                      override_source="Li 2015 p. 4975").resolved_cell()
    assert cell.r == pytest.approx(6e-6)
    assert cell.radius_mean.prov.doi == "10.1073/pnas.1504484112"
    guess = Population(cell_type="wbc", diameter_cv=0.0, override_source="monodisperse")
    assert guess.resolved_cell().radius_cv.prov.is_assumption


def test_sheath_ratio_gives_the_stated_sample_stream():
    """1:2 sample:sheath in the Zhang channel puts the sample in y0 < W/3 (Sec. 4.2)."""
    sim = SAWSorterSimulation(_params())
    ((lo, hi),) = sim.inlet_band_bounds()
    assert lo == 0.0
    assert hi == pytest.approx(W / 3, rel=0.04)
    assert hi > W / 3  # slower flow at the side wall widens a wall-hugging stream


def test_baw_dashboard_renders(tmp_path):
    """The mode-aware Panel dashboard evaluates every bound view without error."""
    pytest.importorskip("panel")
    from biosim_lab.instruments.saw_sorter.dashboard import build_dashboard

    view = build_dashboard(_params(populations=[dict(cell_type="mcf7", count=10, target=True),
                                                dict(cell_type="pbmc", count=10)]))
    out = tmp_path / "baw.html"
    view.save(str(out), embed=False)
    text = out.read_text()
    assert "Alternating-frequency BAW" in text and "W/6 rule" in text
