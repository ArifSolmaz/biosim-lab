"""Built-in FEM solvers, checked against closed-form solutions."""

import numpy as np
import pytest

from biosim_lab.core.fem.electroquasistatic import ElectroQuasistaticSolver
from biosim_lab.core.fem.helmholtz import HelmholtzSolver, analytic_standing_wave
from biosim_lab.core.fem.stokes import StokesSolver
from biosim_lab.core.geometry import straight_channel_2d
from biosim_lab.core.materials import WATER
from biosim_lab.core.solver import BoundaryCondition


def test_electroquasistatic_matches_the_parallel_plate_formula():
    """Z = L / (sigma* * A) exactly, for a uniform field between two plates."""
    width, height, depth = 100e-6, 50e-6, 1e-3
    mesh = straight_channel_2d(width, height, resolution=12)
    solver = ElectroQuasistaticSolver(1e5, conductivity=1.5, permittivity_rel=78.0)
    solver.setup(
        mesh,
        [
            BoundaryCondition("bottom", "dirichlet", 1.0),
            BoundaryCondition("top", "dirichlet", 0.0),
            BoundaryCondition("left", "insulating", 0),
            BoundaryCondition("right", "insulating", 0),
        ],
    )
    solver.run()
    z = solver.impedance(depth=depth)
    expected = height / (solver.sigma_star * width * depth)
    assert abs(z - expected) / abs(expected) < 1e-9


def test_electroquasistatic_impedance_scales_with_geometry():
    def impedance(height):
        mesh = straight_channel_2d(100e-6, height, resolution=10)
        s = ElectroQuasistaticSolver(1e4, conductivity=1.5, permittivity_rel=78.0)
        s.setup(mesh, [
            BoundaryCondition("bottom", "dirichlet", 1.0),
            BoundaryCondition("top", "dirichlet", 0.0),
            BoundaryCondition("left", "insulating", 0),
            BoundaryCondition("right", "insulating", 0),
        ])
        s.run()
        return abs(s.impedance(depth=1e-3))

    assert impedance(100e-6) / impedance(50e-6) == pytest.approx(2.0, rel=1e-6)


def test_helmholtz_reproduces_a_one_dimensional_standing_wave():
    """A rigid box driven at a cavity mode must give the analytic mode shape."""
    f = 20e6
    c = WATER.c
    lam = c / f
    width, height = lam / 2, lam / 8
    mesh = straight_channel_2d(width, height, resolution=48)
    solver = HelmholtzSolver(f, WATER.rho, c, damping=0.02, element_order=2)
    k = 2 * np.pi / lam
    omega = 2 * np.pi * f
    solver.setup(
        mesh,
        [BoundaryCondition("bottom", "velocity", lambda x: -1j * omega * 1e-9
                           * np.cos(k * x[0]))],
    )
    solver.run()
    grid = solver.sample_on_grid(nx=101, ny=9)
    p = grid["p_real"].values + 1j * grid["p_imag"].values
    profile = np.abs(p[:, 4])
    profile = profile / profile.max()
    expected = np.abs(analytic_standing_wave(grid["x"].values, 1.0, lam))
    assert np.sqrt(np.mean((profile - expected) ** 2)) < 0.02


def test_helmholtz_pressure_release_wall_gives_zero_pressure():
    mesh = straight_channel_2d(100e-6, 50e-6, resolution=12)
    solver = HelmholtzSolver(5e6, WATER.rho, WATER.c)
    solver.setup(
        mesh,
        [
            BoundaryCondition("bottom", "velocity", 1e-3),
            BoundaryCondition("top", "pressure", 0.0),
        ],
    )
    solver.run()
    ds = solver.fields()
    top = ds["p_abs"].values[ds["y"].values > 50e-6 - 1e-9]
    assert np.allclose(top, 0.0, atol=1e-9)


def test_helmholtz_rejects_unknown_boundary_kinds():
    mesh = straight_channel_2d(100e-6, 50e-6, resolution=6)
    solver = HelmholtzSolver(1e6, WATER.rho, WATER.c)
    with pytest.raises(ValueError, match="unknown acoustic BC"):
        solver.setup(mesh, [BoundaryCondition("bottom", "nonsense", 1.0)])
    with pytest.raises(KeyError, match="not in mesh"):
        solver.setup(mesh, [BoundaryCondition("nowhere", "rigid", 0.0)])


def test_helmholtz_requires_setup_before_run():
    solver = HelmholtzSolver(1e6, WATER.rho, WATER.c)
    with pytest.raises(RuntimeError, match="setup"):
        solver.run()


def test_stokes_enforces_no_slip_and_drives_recirculation():
    mesh = straight_channel_2d(100e-6, 50e-6, resolution=10)
    solver = StokesSolver(1e-3)
    lid = 1e-3
    solver.setup(
        mesh,
        [
            BoundaryCondition("bottom", "no_slip", 0),
            BoundaryCondition("left", "no_slip", 0),
            BoundaryCondition("right", "no_slip", 0),
            BoundaryCondition("top", "velocity", (lid, 0.0)),
        ],
    )
    solver.run()
    fields = solver.fields()
    ux = fields["u_x"].values
    y = fields["y"].values
    assert np.allclose(ux[y < 1e-9], 0.0, atol=1e-9), "no-slip on the floor"
    assert ux.max() == pytest.approx(lid, rel=1e-6), "the lid sets the maximum"
    assert ux.min() < 0, "a closed cavity must recirculate"


def test_stokes_rejects_a_negative_viscosity():
    with pytest.raises(ValueError):
        StokesSolver(-1.0)
