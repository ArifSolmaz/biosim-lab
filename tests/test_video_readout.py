"""Counting a sort from a synthetic video of its outlets (biosim_lab.video_readout).

The end-to-end tests film a small version of the Zhang et al. (2023,
doi:10.3390/ijms24043338) alternating-BAW device once and check that the
count made from the pictures alone lands on the simulation's own numbers,
and that it cannot have peeked at them.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from benchmarks.benchmark_02_alternating_baw import cases
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation
from biosim_lab.video_readout import (
    CameraSpec,
    _arrivals,
    _calibrate,
    count_film,
    film_sorter,
    otsu_split,
)

pytest.importorskip("trackpy")


# -- pieces --------------------------------------------------------------------


def test_otsu_split_cuts_in_the_middle_of_the_gap():
    low = 0.05 + np.linspace(-0.001, 0.0007, 23)
    high = 0.58 + np.linspace(-0.01, 0.02, 12)
    cut = otsu_split(np.concatenate([low, high]))
    # skimage's 256-bin version put this at the top edge of the low cluster
    assert low.max() < cut < high.min()
    assert cut == pytest.approx(0.5 * (low.max() + high.min()))


def test_otsu_split_degenerate_inputs():
    assert np.isnan(otsu_split(np.array([1.0])))
    assert np.isnan(otsu_split(np.full(5, 2.0)))
    assert otsu_split(np.array([1.0, 3.0])) == pytest.approx(2.0)


def test_arrivals_keep_each_cell_on_its_drive_phase():
    rng = np.random.default_rng(0)
    n, period = 200, 2.2
    t_in = rng.uniform(10.0, 40.0, n)
    phase = rng.integers(0, 440, n) * period / 440
    arrival = _arrivals(t_in, rate=30.0, rng=rng, period=period, phase=phase)
    offset = np.mod(arrival - phase, period)
    assert np.allclose(np.minimum(offset, period - offset), 0.0, atol=1e-9)
    # ...and still reach the window as a stream a few seconds long, not the
    # 30 s the transit times alone would spread them over.
    reach = arrival + t_in
    assert np.ptp(reach) < n / 30.0 + period + 1.0


def test_calibration_threshold_ignores_empty_frames():
    rng = np.random.default_rng(1)
    frames = [0.25 + rng.normal(0.0, 0.02, (120, 90)) for _ in range(20)]
    background, level = _calibrate(frames)
    assert background.shape == (120, 90)
    from biosim_lab.core.imaging import segment

    blank = 0.25 + rng.normal(0.0, 0.02, (120, 90))
    found = segment(blank, threshold=level, background=background, min_radius_px=1.5)
    assert found.n_objects == 0


def test_alternating_sorter_records_entry_phase():
    cfg = cases.base_config()
    cfg["populations"] = [dict(p, count=3) for p in cfg["populations"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        out = SAWSorterSimulation(SAWSorterParams(**cfg)).run()
    period = sum(p.duration for p in SAWSorterParams(**cfg).switching.phases)
    offsets = out.cells["entry_offset_s"].to_numpy()
    assert offsets.shape == (6,) and np.all((offsets >= 0) & (offsets < period))


# -- end to end ----------------------------------------------------------------


@pytest.fixture(scope="module")
def film():
    cfg = cases.base_config()
    cfg["populations"] = [dict(p, count=10 if p.get("target") else 20)
                          for p in cfg["populations"]]
    return film_sorter(SAWSorterParams(**cfg), seed=3, keep_frames=4,
                       camera=CameraSpec(pixel_size=1.5e-6))


@pytest.mark.slow
def test_frame_rate_keeps_linking_unambiguous(film):
    g = film.geometry
    r_min_px = film.outcome.tracks.trajectories["radius"].values[film.in_view].min() \
        / g["pixel_size_m"]
    assert g["search_range_px"] < 2.0 * r_min_px
    assert film.sample_frames.shape == (4, g["rows"], g["cols"])


@pytest.mark.slow
@pytest.mark.parametrize("by", ["size", "fluorescence"])
def test_video_count_agrees_with_the_simulation(film, by):
    r = count_film(film, by)
    e = r.error_budget
    assert e["counted_fraction_percent"] >= 85.0
    assert e["classification_accuracy_percent"] >= 95.0
    assert e["outlet_agreement_percent"] >= 95.0
    assert e["missed_cells"] == e["missed_occluded"] + e["missed_other"]
    v, t = r.video_metrics, r.truth_metrics
    assert abs(v["capture_efficiency_percent"] - t["capture_efficiency_percent"]) <= 15.0
    assert abs(v["contamination_rate_percent"] - t["contamination_rate_percent"]) <= 10.0
    assert v["capture_ci_low"] <= v["capture_efficiency_percent"] <= v["capture_ci_high"]


@pytest.mark.slow
def test_the_count_never_sees_the_simulation_labels(film):
    """Scramble the truth: the video's numbers must not move, only its score."""
    honest = count_film(film, "size")
    lied = copy.copy(film)
    lied.is_target = ~film.is_target
    lied.outcome = copy.copy(film.outcome)
    lied.outcome.cells = film.outcome.cells.assign(is_target=~film.outcome.cells["is_target"])
    fooled = count_film(lied, "size")
    for key in ("capture_efficiency_percent", "contamination_rate_percent", "n_target"):
        assert fooled.video_metrics[key] == honest.video_metrics[key]
    assert fooled.error_budget["classification_accuracy_percent"] <= 5.0
