"""The three shipped failures found by exercising settings nobody had clicked.

Each of these was reachable from the web app with two clicks and crashed it.
None was reachable from the default widget positions, which is why rendering
each page once had reported everything healthy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import ConfigurationError, RegimeWarning
from biosim_lab.instruments.cell_tracker.instrument import CellTracker
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

BASE = dict(
    frequency="6.632 MHz", voltage_pp="15 V", channel_width="300 um",
    channel_height="50 um", channel_length="2 mm", flow_rate="5 uL/min",
    fluid="water", substrate="linbo3_128yx", inlet="sheath_sides",
    collection_fraction=1 / 3, mode="analytic", seed=1,
)
POPS = [
    {"cell_type": "mcf7", "count": 60, "target": True},
    {"cell_type": "rbc", "count": 60, "target": False},
]


def _sim(**over: object) -> SAWSorterSimulation:
    cfg = {**BASE, **over}
    pops = cfg.pop("populations", POPS)
    return SAWSorterSimulation(SAWSorterParams(populations=pops, **cfg))  # type: ignore[arg-type]


# -- 1. the centre inlet returned one position for the whole population ----


@pytest.mark.parametrize("inlet", ["sheath_sides", "uniform", "centre"])
def test_every_inlet_places_one_cell_per_cell(inlet: str) -> None:
    """``rng.uniform`` without ``size=`` returns a scalar, not an array.

    Both bounds are scalars in the centre branch, so it silently produced a
    single float for the entire population and the caller then tried to
    concatenate zero-dimensional arrays.
    """
    state = _sim(inlet=inlet).build_ensemble()
    assert len(state.x) == 120
    assert np.asarray(state.x).ndim == 2


def test_the_centre_inlet_actually_focuses_at_the_centre() -> None:
    """Shape alone is not enough; the positions must also mean something."""
    state = _sim(inlet="centre").build_ensemble()
    x_um = np.asarray(state.x)[:, 0] * 1e6
    assert x_um.min() > 100.0 and x_um.max() < 200.0, "not focused near the 150 um midline"


# -- 2. cells taller than the channel made the wall clearance ill-posed ---


def test_cells_too_tall_for_the_channel_are_excluded_and_reported() -> None:
    """A 20 um channel rejects ~16 % of an MCF-7 population; that is not a tail.

    Sampling a wall clearance for such a cell asks for uniform(low, high) with
    low > high. Excluding them is the physical answer --- the cell never enters
    the device --- but it must be said out loud, because it changes what the
    metrics describe.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = _sim(channel_height="20 um").build_ensemble()

    messages = [str(w.message) for w in caught if issubclass(w.category, RegimeWarning)]
    assert any("taller than the 20 um channel" in m for m in messages), messages
    assert len(state.x) < 120, "some cells should have been excluded"
    # Everything that survived must actually fit.
    assert np.all(2.0 * np.asarray(state.radius) <= 20e-6 + 1e-15)


def test_a_channel_no_cell_fits_is_refused_with_a_usable_message() -> None:
    with pytest.raises(ConfigurationError, match="no mcf7 cell fits"):
        _sim(channel_height="5 um",
             populations=[{"cell_type": "mcf7", "count": 40, "target": True}]).run()


def test_a_roomy_channel_excludes_nobody() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = _sim(channel_height="50 um").build_ensemble()
    assert len(state.x) == 120
    assert not [w for w in caught if "taller than" in str(w.message)]


# -- 3. an over-wide linking radius crashed inside trackpy ----------------


def _tracker(**params: object) -> CellTracker:
    defaults = {"n_frames": 20, "n_cells": 60, "image_size": 256, "seed": 2}
    return CellTracker(ExperimentConfig.model_validate(
        {"instrument": "cell_tracker", "params": {**defaults, **params}}
    ))


@pytest.mark.slow
def test_an_ambiguous_linking_radius_is_explained_not_raised_from_trackpy() -> None:
    """trackpy refuses with SubnetOversizeException, naming an internal detail.

    Refusing is correct --- once the search radius reaches typical cell spacing
    there is no unique assignment. But the user needs to be told what to change,
    and the app needs an error type it can distinguish from a bug.
    """
    with pytest.raises(ConfigurationError, match="too large for this cell density"):
        _tracker(search_range_px=30.0).run()


@pytest.mark.slow
def test_a_sane_linking_radius_still_tracks() -> None:
    assert _tracker(search_range_px=12.0).run().metrics["n_tracks"] > 0


def test_configuration_error_is_a_value_error() -> None:
    """Existing `except ValueError` handlers must keep working."""
    assert issubclass(ConfigurationError, ValueError)
