"""Stokes drag and regime validation."""

import numpy as np
import pytest

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.physics import drag


def test_stokes_drag_formula():
    u_f = np.array([[1e-3, 0.0]])
    u_p = np.array([[0.0, 0.0]])
    r = np.array([5e-6])
    f = drag.stokes_drag(u_f, u_p, r, 1e-3)
    assert f[0, 0] == pytest.approx(6 * np.pi * 1e-3 * 5e-6 * 1e-3)
    assert f[0, 1] == 0.0


def test_drag_vanishes_when_particle_matches_the_fluid():
    u = np.array([[2e-3, 1e-3]])
    assert np.allclose(drag.stokes_drag(u, u, np.array([5e-6]), 1e-3), 0.0)


def test_mobility_is_the_inverse_of_the_drag_coefficient():
    r = np.array([3e-6, 9e-6])
    assert np.allclose(drag.mobility(r, 1e-3), 1 / (6 * np.pi * 1e-3 * r))


def test_hydraulic_diameter_of_a_square_duct_equals_its_side():
    assert drag.hydraulic_diameter(1e-4, 1e-4) == pytest.approx(1e-4)


def test_default_operating_point_is_deeply_in_the_stokes_regime():
    """No warning may fire at the shipped default operating point."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = drag.stokes_regime_report(
            mean_velocity=5.56e-3, width=300e-6, height=50e-6,
            radius=np.array([9e-6]), rho_f=997.0, viscosity=0.89e-3,
        )
    assert report["channel_reynolds"] < 1.0
    assert report["particle_reynolds"] < 1.0


def test_fast_flow_warns_about_inertia():
    with pytest.warns(RegimeWarning, match="Reynolds"):
        drag.stokes_regime_report(
            mean_velocity=5.0, width=300e-6, height=50e-6,
            radius=np.array([9e-6]), rho_f=997.0, viscosity=0.89e-3,
        )


def test_faxen_correction_increases_drag_near_a_wall_and_is_bounded():
    far = drag.faxen_wall_correction(np.array([1e-6]), np.array([1e-3]))
    near = drag.faxen_wall_correction(np.array([1e-6]), np.array([2e-6]))
    contact = drag.faxen_wall_correction(np.array([1e-6]), np.array([1e-9]))
    assert far == pytest.approx(1.0, rel=1e-3)
    assert near > far
    assert np.isfinite(contact).all(), "the series must be clipped, not divergent"
