"""Verification: does the code solve the equations it claims to solve?

The rest of the suite is *validation* --- it checks that answers match a
hand-derived special case, an identity, or a conservation law. Those checks can
be passed by a scheme that is subtly wrong, because a single special case is a
single point.

Verification asks the stronger question: as the discretisation is refined, does
the error fall at the **rate the method promises**? A solver that agrees with an
exact answer at one mesh size might be lucky. One that converges at exactly
second order over four refinements is implementing second-order elements.

The technique is the *method of manufactured solutions* (MMS): choose an exact
solution, differentiate it to find the source term that would produce it, feed
that source to the solver, and measure the error against the chosen solution.
It works on any domain and needs no analytic solution of the real problem.

Reference: Roache, *Code Verification by the Method of Manufactured Solutions*,
J. Fluids Eng. 124:4 (2002), doi:10.1115/1.1436090; Salari & Knupp,
*Code Verification by the Method of Manufactured Solutions*, SAND2000-1444,
doi:10.2172/759450.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core.fem.helmholtz import HelmholtzSolver
from biosim_lab.core.geometry import straight_channel_2d
from biosim_lab.core.particles import ForceRegistry, LagrangianTracker, make_state
from biosim_lab.core.solver import BoundaryCondition


def observed_order(sizes: list[float], errors: list[float]) -> float:
    """Slope of log(error) against log(h) --- the observed convergence order."""
    return float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])


# ---------------------------------------------------------------------------
# the Helmholtz solver
# ---------------------------------------------------------------------------

# Manufactured solution on the unit square, zero on every boundary:
#     p(x, y) = sin(pi x) sin(pi y)
#     laplacian(p) = -2 pi^2 p
#     laplacian(p) + k^2 p = -f   =>   f = (2 pi^2 - k^2) p
_K = 2.0


def _exact(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def _source(x: np.ndarray) -> np.ndarray:
    return (2.0 * np.pi**2 - _K**2) * _exact(x)


def _solve_manufactured(resolution: int, order: int) -> float:
    """Solve the manufactured problem at one mesh size; return the L2 error."""
    mesh = straight_channel_2d(1.0, 1.0, resolution=resolution)
    # omega / c must equal _K for the solver's wavenumber to match the source.
    solver = HelmholtzSolver(
        frequency=_K / (2.0 * np.pi), density=1.0, speed_of_sound=1.0,
        element_order=order,
    )
    assert solver.wavenumber == pytest.approx(_K)
    solver.setup(
        mesh,
        [BoundaryCondition(side, "pressure", 0.0)
         for side in ("left", "right", "top", "bottom")],
        source=_source,
    )
    solver.run()
    return solver.l2_error(_exact)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("order", "theoretical"), [(1, 2.0), (2, 3.0)], ids=["P1", "P2"]
)
def test_helmholtz_converges_at_the_theoretical_order(order: int, theoretical: float):
    """L2 error must fall as h^(k+1) for degree-k Lagrange elements.

    P1 gives second order, P2 gives third. Getting the *rate* right is what
    distinguishes a correct implementation from one that merely happens to be
    close on a particular mesh.
    """
    resolutions = [8, 16, 32, 64]
    errors = [_solve_manufactured(r, order) for r in resolutions]
    sizes = [1.0 / r for r in resolutions]

    assert all(b < a for a, b in zip(errors, errors[1:])), (
        f"error must fall under refinement, got {errors}"
    )
    measured = observed_order(sizes, errors)
    assert measured == pytest.approx(theoretical, abs=0.15), (
        f"P{order}: expected order {theoretical}, measured {measured:.3f} "
        f"from errors {errors}"
    )


def test_helmholtz_recovers_the_manufactured_solution_at_all():
    """A cheap version of the above, so the fast suite still checks the source term."""
    assert _solve_manufactured(24, 2) < 1e-4


def test_helmholtz_without_a_source_is_unchanged():
    """The source term is optional and must not perturb the homogeneous problem."""
    mesh = straight_channel_2d(1.0, 1.0, resolution=12)
    results = []
    for source in (None, 0.0):
        solver = HelmholtzSolver(1e5, 997.0, 1497.0, element_order=1)
        solver.setup(
            mesh,
            [BoundaryCondition("bottom", "velocity", 1e-3),
             BoundaryCondition("top", "pressure", 0.0)],
            source=source,
        )
        solver.run()
        results.append(solver.solution())
    assert np.allclose(results[0], results[1])


# ---------------------------------------------------------------------------
# the trajectory integrator
# ---------------------------------------------------------------------------

_MU = 1e-3
_RADIUS = 5e-6
_DRAG = 6.0 * np.pi * _MU * _RADIUS


def _track(force, t_end: float, rtol: float, x0: float = 0.0) -> float:
    """Integrate one particle under *force* and return its final x."""
    registry = ForceRegistry()
    registry.register("manufactured", force)
    state = make_state(np.array([[x0, 0.0]]), _RADIUS, 1050.0, 4e-10, ["a"])
    tracker = LagrangianTracker(registry, lambda t, x: np.zeros_like(x), _MU)
    result = tracker.run(state, (0.0, t_end), n_samples=3, rtol=rtol, atol=1e-16)
    return float(result.final["x_final_m"][0])


def test_integrator_reproduces_an_exact_exponential_decay():
    """F = -kappa x in the overdamped limit gives x(t) = x0 exp(-kappa t / drag).

    A closed-form solution of the actual ODE being integrated, so any error is
    the integrator's.
    """
    kappa = 1e-7
    x0, t_end = 100e-6, 2.0

    def spring(state):
        out = np.zeros_like(state.x)
        out[:, 0] = -kappa * state.x[:, 0]
        return out

    expected = x0 * np.exp(-kappa * t_end / _DRAG)
    assert _track(spring, t_end, rtol=1e-10, x0=x0) == pytest.approx(expected, rel=1e-6)


def test_integrator_reproduces_a_manufactured_time_dependent_trajectory():
    """Manufacture x(t) = A sin(omega t) by supplying the force that produces it.

    In the overdamped limit ``dx/dt = F / drag``, so ``F(t) = drag * A * omega *
    cos(omega t)`` yields exactly that trajectory. Nothing about the force
    depends on position, so the test isolates the time integration.
    """
    amplitude, omega, t_end = 40e-6, 3.0, 1.7

    def driving(state):
        out = np.zeros_like(state.x)
        out[:, 0] = _DRAG * amplitude * omega * np.cos(omega * state.t)
        return out

    expected = amplitude * np.sin(omega * t_end)
    assert _track(driving, t_end, rtol=1e-11) == pytest.approx(expected, rel=1e-5)


def test_integrator_error_falls_as_the_tolerance_is_tightened():
    """Loosening rtol must degrade the answer, and tightening must recover it.

    An integrator that ignores its tolerance would return the same number for
    all of these, which is a failure mode that no single-tolerance check sees.
    """
    amplitude, omega, t_end = 40e-6, 3.0, 1.7
    exact = amplitude * np.sin(omega * t_end)

    def driving(state):
        out = np.zeros_like(state.x)
        out[:, 0] = _DRAG * amplitude * omega * np.cos(omega * state.t)
        return out

    errors = [
        abs(_track(driving, t_end, rtol=tol) - exact) for tol in (1e-4, 1e-7, 1e-10)
    ]
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] < 1e-10 * amplitude * 1e3


# ---------------------------------------------------------------------------
# the Gor'kov post-processing chain
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gorkov_gradient_chain_is_second_order_in_the_sampling_grid():
    """The force is central differences on an interpolated field.

    That composition has never had its order measured --- only its value at one
    grid size. Central differences are second order, and the whole chain must
    inherit that, or the interpolation is contaminating the derivative.
    """
    import xarray as xr

    from biosim_lab.core.materials import MCF7, WATER
    from biosim_lab.instruments.saw_sorter.physics.acoustics import (
        effective_contrast_factor,
        gorkov_force_on_grid,
        primary_radiation_force_1d,
    )

    p0, lam = 4.5e5, 600e-6
    frequency = 3979.0 / lam
    kx = 2 * np.pi / lam
    kf = 2 * np.pi * frequency / WATER.c
    ky = np.sqrt(kf**2 - kx**2)
    volume = 4 / 3 * np.pi * MCF7.r**3
    phi = effective_contrast_factor(
        MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa, k_transverse=kx, k_fluid=kf
    )

    sizes, errors = [], []
    for n in (51, 101, 201, 401):
        xs = np.linspace(0.0, 300e-6, n)
        ys = np.linspace(0.0, 50e-6, 41)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        field = p0 * np.sin(kx * (X - 150e-6)) * np.cos(ky * Y)
        grid = xr.Dataset(
            {"p_real": (("x", "y"), field), "p_imag": (("x", "y"), np.zeros_like(field))},
            coords={"x": xs, "y": ys},
        )
        numeric = gorkov_force_on_grid(
            grid, volume=volume, rho_f=WATER.rho, c_f=WATER.c,
            rho_p=MCF7.rho, kappa_p=MCF7.kappa, frequency=frequency,
        )["F_x"].values[:, 0]
        analytic = primary_radiation_force_1d(
            xs, p0=p0, volume=volume, kappa_f=WATER.kappa, wavelength=lam,
            phi=phi, node_offset=150e-6,
        )
        # Interior only: one-sided differences at the edges are first order by
        # construction and would mask the interior rate.
        interior = slice(2, -2)
        errors.append(float(np.abs(numeric[interior] - analytic[interior]).max()))
        sizes.append(xs[1] - xs[0])

    measured = observed_order(sizes, errors)
    assert measured == pytest.approx(2.0, abs=0.3), (
        f"expected second order from central differences, measured {measured:.3f} "
        f"from errors {errors}"
    )
