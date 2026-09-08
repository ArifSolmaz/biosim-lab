"""Two-outlet lateral splitting: large cells to one side, everything else to the other.

The original model only described a three-outlet chip --- a collection band
centred on the pressure node, creaming large cells off the middle. A great many
real devices are two-outlet instead: the sample enters along one wall, large
cells cross the channel toward the node, small ones do not, and a single divider
separates them. That topology could not be expressed at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

BASE = dict(
    frequency="6.632 MHz", voltage_pp="15 V", channel_width="300 um",
    channel_height="50 um", channel_length="2 mm", flow_rate="5 uL/min",
    fluid="water", substrate="linbo3_128yx", mode="analytic", seed=20260907,
)
POPS = [
    {"cell_type": "mcf7", "count": 150, "target": True},
    {"cell_type": "rbc", "count": 150, "target": False},
]


def _sim(**over: object) -> SAWSorterSimulation:
    cfg = {**BASE, **over}
    pops = cfg.pop("populations", POPS)
    return SAWSorterSimulation(SAWSorterParams(populations=pops, **cfg))  # type: ignore[arg-type]


def _run(sim: SAWSorterSimulation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return sim.run()


# -- the collection region ------------------------------------------------


def test_the_default_layout_is_unchanged() -> None:
    """The three-outlet chip must keep behaving exactly as before."""
    sim = _sim()
    half = 0.5 * (1 / 3) * 300e-6
    assert sim.params.outlet_layout == "centre_band"
    lo, hi = sim.collection_bounds
    assert lo == pytest.approx(sim.node_offset - half)
    assert hi == pytest.approx(sim.node_offset + half)


def test_a_lateral_split_collects_one_whole_side() -> None:
    right = _sim(outlet_layout="lateral_split", split_position=0.4,
                 collect_side="right").collection_bounds
    assert right == pytest.approx((120e-6, 300e-6))

    left = _sim(outlet_layout="lateral_split", split_position=0.4,
                collect_side="left").collection_bounds
    assert left == pytest.approx((0.0, 120e-6))


def test_the_split_ignores_the_band_width_that_does_not_apply_to_it() -> None:
    """collection_fraction describes the centre band and must not leak across."""
    a = _sim(outlet_layout="lateral_split", split_position=0.4,
             collection_fraction=0.1).collection_bounds
    b = _sim(outlet_layout="lateral_split", split_position=0.4,
             collection_fraction=0.8).collection_bounds
    assert a == pytest.approx(b)


# -- the one-side inlet ---------------------------------------------------


@pytest.mark.parametrize("side,expected", [("left", 45e-6), ("right", 255e-6)])
def test_a_side_inlet_puts_the_whole_sample_against_one_wall(
    side: str, expected: float
) -> None:
    """Every cell must start in the band at that wall, giving it the full width."""
    state = _sim(inlet="side", inlet_side=side).build_ensemble()
    x = np.asarray(state.x)[:, 0]
    assert len(x) == 300
    if side == "left":
        assert x.max() < expected, "cells strayed past the inlet band"
    else:
        assert x.min() > expected


# -- the separation this exists to describe -------------------------------


@pytest.mark.slow
def test_large_cells_leave_on_one_side_and_small_cells_on_the_other() -> None:
    """The claim, end to end: a divider between them separates the two.

    Large cells cross to the node; small ones stay near the wall they entered
    on. With the divider between the two landing positions the split is nearly
    complete.
    """
    outcome = _run(_sim(inlet="side", inlet_side="left",
                        outlet_layout="lateral_split", split_position=0.4,
                        collect_side="right"))
    per_pop = outcome.metrics["per_population"]

    assert per_pop["mcf7"]["mean_outlet_x_um"] > 130.0, "large cells did not cross"
    assert per_pop["rbc"]["mean_outlet_x_um"] < 80.0, "small cells should not have"
    assert outcome.metrics["efficiency_percent"] > 90.0
    assert outcome.metrics["purity_percent"] > 90.0


@pytest.mark.slow
def test_swapping_the_collection_side_swaps_what_is_collected() -> None:
    """A sanity check on orientation: the other outlet holds the remainder."""
    kwargs = dict(inlet="side", inlet_side="left",
                  outlet_layout="lateral_split", split_position=0.4)
    right = _run(_sim(collect_side="right", **kwargs)).metrics
    left = _run(_sim(collect_side="left", **kwargs)).metrics

    assert right["n_collected"] + left["n_collected"] == right["n_cells"]
    # Large cells are on the right, so collecting the left inverts the purity.
    assert right["purity_percent"] > 90.0
    assert left["purity_percent"] < 20.0


@pytest.mark.slow
def test_a_divider_on_the_node_catches_nothing_and_says_why() -> None:
    """Cells stop short of the node, so 'at the node' is already too far.

    Without the warning this is a bare 0 % recovery and a nan purity, which
    reads as a broken run rather than a divider in the wrong place.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = _sim(inlet="side", outlet_layout="lateral_split",
                       split_position=0.5, collect_side="right").run()

    assert outcome.metrics["n_collected"] == 0
    messages = [str(w.message) for w in caught if issubclass(w.category, RegimeWarning)]
    assert any("caught no cells" in m for m in messages), messages
    assert any("stopped" in m and "short" in m for m in messages)
