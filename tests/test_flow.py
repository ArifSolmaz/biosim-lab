"""Rectangular Poiseuille flow: exactness and limits."""

import numpy as np
import pytest

from biosim_lab.instruments.saw_sorter.flow import RectangularPoiseuille

Q = 5e-9 / 60  # 5 uL/min


def test_flow_rate_is_conserved_by_the_series():
    """Independent check: integrating the profile must return the requested Q."""
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    assert flow.verify_flow_rate() < 1e-3


def test_mean_velocity_equals_q_over_area():
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    assert flow.mean_velocity == pytest.approx(Q / (300e-6 * 50e-6), rel=1e-9)


def test_wide_channel_recovers_the_parallel_plate_ratio():
    """For W >> H the centreline/mean velocity ratio must approach 3/2."""
    flow = RectangularPoiseuille(2e-2, 50e-6, 0.89e-3, Q)
    assert flow.max_velocity / flow.mean_velocity == pytest.approx(1.5, rel=0.01)


def test_no_slip_on_all_four_walls():
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    x = np.linspace(0, 300e-6, 25)
    y = np.linspace(0, 50e-6, 9)
    assert np.allclose(flow.velocity(np.zeros_like(y), y), 0.0, atol=1e-12)
    assert np.allclose(flow.velocity(np.full_like(y, 300e-6), y), 0.0, atol=1e-12)
    assert np.allclose(flow.velocity(x, np.zeros_like(x)), 0.0, atol=1e-12)
    assert np.allclose(flow.velocity(x, np.full_like(x, 50e-6)), 0.0, atol=1e-12)


def test_velocity_is_positive_and_peaks_at_the_centre():
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    xs, ys, u = flow.profile(61, 21)
    interior = u[1:-1, 1:-1]
    assert (interior > 0).all()
    i, j = np.unravel_index(np.argmax(u), u.shape)
    assert xs[i] == pytest.approx(150e-6, abs=6e-6)
    assert ys[j] == pytest.approx(25e-6, abs=2e-6)


def test_flow_rate_scales_linearly_with_pressure_gradient():
    a = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    b = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, 2 * Q)
    assert b.pressure_gradient / a.pressure_gradient == pytest.approx(2.0)


def test_transit_time():
    flow = RectangularPoiseuille(300e-6, 50e-6, 0.89e-3, Q)
    assert flow.transit_time(2e-3) == pytest.approx(2e-3 / flow.mean_velocity)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        RectangularPoiseuille(0.0, 50e-6, 1e-3, Q)
    with pytest.raises(ValueError):
        RectangularPoiseuille(300e-6, 50e-6, -1e-3, Q)
