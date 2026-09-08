"""Tilted-angle SSAW, which is a different mechanism rather than a different layout.

A conventional device puts the standing wave straight across the channel: node
planes lie parallel to the flow, a cell migrates sideways to a node and stops,
and its displacement is capped by the node spacing however long the channel is.

Tilting the IDTs makes the node planes cross the flow. A cell held in one is
then dragged across the channel as it travels downstream, so displacement grows
with channel length, and the separation turns on whether a cell can be *held* at
all rather than on how fast it migrates (doi:10.1073/pnas.1413325111).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from biosim_lab.core.materials import get_cell
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    cutoff_radius,
    max_trappable_tilt,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

BASE = dict(
    frequency="6.632 MHz", voltage_pp="15 V", channel_width="600 um",
    channel_height="50 um", channel_length="4 mm", flow_rate="5 uL/min",
    fluid="water", substrate="linbo3_128yx", inlet="side", inlet_side="left",
    mode="analytic", seed=20260907,
)
POPS = [{"cell_type": "mcf7", "count": 80, "target": True}]


def _sim(**over):
    cfg = {**BASE, **over}
    pops = cfg.pop("populations", POPS)
    return SAWSorterSimulation(SAWSorterParams(populations=pops, **cfg))


def _run(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return sim.run()


# -- the untilted case must be untouched ----------------------------------


def test_zero_tilt_is_the_conventional_device_exactly() -> None:
    """The default must keep producing bit-identical results."""
    a = _run(_sim()).cells["x_outlet_m"].to_numpy()
    b = _run(_sim(tilt_angle_deg=0.0)).cells["x_outlet_m"].to_numpy()
    assert np.array_equal(a, b)


def test_the_force_is_purely_lateral_without_tilt() -> None:
    sim = _sim()
    state = sim.build_ensemble()
    force = sim._analytic_arf(state)
    assert np.allclose(force[:, 2], 0.0), "an untilted wave must not push along the flow"


# -- the force geometry ---------------------------------------------------


@pytest.mark.parametrize("tilt", [5.0, 15.0, 30.0])
def test_the_force_points_along_the_rotated_wave_vector(tilt: float) -> None:
    """Rotating the IDTs rotates the force with them: Fz/Fx = tan(theta)."""
    sim = _sim(tilt_angle_deg=tilt)
    state = sim.build_ensemble()
    force = sim._analytic_arf(state)

    strong = np.abs(force[:, 0]) > 1e-18
    ratio = force[strong, 2] / force[strong, 0]
    assert np.allclose(ratio, np.tan(np.deg2rad(tilt)), rtol=1e-9)


def test_tilting_widens_the_node_spacing_seen_across_the_channel() -> None:
    """The wave keeps its wavelength; the channel cuts the planes obliquely."""
    straight = _sim().lateral_node_spacing
    tilted = _sim(tilt_angle_deg=30.0).lateral_node_spacing
    assert tilted == pytest.approx(straight / np.cos(np.deg2rad(30.0)))


# -- the mechanism --------------------------------------------------------


@pytest.mark.slow
def test_displacement_saturates_without_tilt_and_grows_with_it() -> None:
    """The signature of the mechanism, and the reason to tilt at all."""
    straight = [
        _run(_sim(channel_length=f"{mm} mm")).metrics["per_population"]["mcf7"][
            "mean_outlet_x_um"] for mm in (2, 4, 8)
    ]
    tilted = [
        _run(_sim(channel_length=f"{mm} mm", tilt_angle_deg=-5.0)).metrics[
            "per_population"]["mcf7"]["mean_outlet_x_um"] for mm in (2, 4, 8)
    ]
    # Straight: parked on a node, so doubling the channel changes nothing.
    assert straight[2] == pytest.approx(straight[1], rel=0.02)
    # Tilted: still moving after the straight device has stopped.
    assert tilted[1] > tilted[0] + 50.0
    assert tilted[2] > tilted[1] + 50.0


# -- the trapping criterion ------------------------------------------------


def _criterion_kwargs(sim):
    p = sim.params
    return {
        "p0": p.p0, "kappa_f": sim.fluid.kappa, "wavelength": sim.wavelength,
        "viscosity": sim.fluid.mu,
        "flow_speed": p.flow_rate / (p.channel_width * p.channel_height),
    }


def test_the_tilt_limit_and_the_cutoff_size_are_inverses() -> None:
    sim = _sim()
    kw = _criterion_kwargs(sim)
    phi = sim.phi_for(get_cell("mcf7"))
    tilt = max_trappable_tilt(9e-6, phi=phi, **kw)
    assert cutoff_radius(float(tilt), phi=phi, **kw) == pytest.approx(9e-6, rel=1e-6)


def test_bigger_cells_tolerate_more_tilt() -> None:
    """R scales with a^2, so the small cells lose their grip first.

    That asymmetry is the separation: any angle between the two limits deflects
    one population and lets the other flow straight through.
    """
    sim = _sim()
    kw = _criterion_kwargs(sim)
    limits = {
        name: float(np.degrees(max_trappable_tilt(
            float(get_cell(name).r), phi=sim.phi_for(get_cell(name)), **kw)))
        for name in ("mcf7", "rbc", "platelet")
    }
    assert limits["mcf7"] > limits["rbc"] > limits["platelet"]


@pytest.mark.slow
def test_a_tilt_past_the_limit_lets_cells_slip_straight_through() -> None:
    """Past the limit the device silently does nothing, which is worth testing.

    The average force over a slipping trajectory cancels, so the cell flows on
    almost undeflected --- a failure that produces plausible-looking output.
    """
    sim = _sim()
    kw = _criterion_kwargs(sim)
    limit = float(np.degrees(max_trappable_tilt(
        float(get_cell("mcf7").r), phi=sim.phi_for(get_cell("mcf7")), **kw)))

    held = _run(_sim(tilt_angle_deg=-0.5 * limit)).metrics["per_population"]["mcf7"]
    slipped = _run(_sim(tilt_angle_deg=-(limit + 25.0))).metrics["per_population"]["mcf7"]
    assert held["mean_outlet_x_um"] > 3.0 * slipped["mean_outlet_x_um"]


def test_the_design_window_is_reported() -> None:
    report = _sim(tilt_angle_deg=-10.0).tilt_report()
    assert report["per_population"]["mcf7"]["max_trappable_tilt_deg"] > 10.0
    assert report["cutoff_diameter_um"] > 0.0
    assert report["geometric_drift_um"] == pytest.approx(
        4e-3 * np.tan(np.deg2rad(10.0)) * 1e6
    )


# -- the two ways a tilted run is quietly wrong ----------------------------


def test_fem_mode_refuses_a_tilt_rather_than_ignoring_it() -> None:
    """The FEM field does not vary along the flow, so it cannot hold a tilt."""
    with pytest.raises(ValueError, match="cannot represent a tilted pattern"):
        SAWSorterParams(voltage_pp="15 V", mode="fem", tilt_angle_deg=10.0)


def test_a_tilt_into_the_inlet_wall_is_flagged() -> None:
    """Wrong sign: every cell is pressed into the wall it started against."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sim(tilt_angle_deg=+10.0, inlet_side="left").run()
    assert any("deflects cells toward the left wall" in str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sim(tilt_angle_deg=-10.0, inlet_side="left").run()
    assert not any("deflects cells toward" in str(w.message) for w in caught)
