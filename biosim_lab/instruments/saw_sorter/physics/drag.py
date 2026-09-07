"""Viscous drag and validity checks for the creeping-flow regime.

Stokes' law
-----------
    F_d = 6 * pi * mu * r * (u_fluid - u_particle)      [N]

valid for particle Reynolds number ``Re_p = rho_f * |u_rel| * 2r / mu << 1``.
In a 300 x 50 um channel at 5 uL/min the mean velocity is ~5.6 mm/s and
``Re_p`` for a 20 um cell is ~1e-1 * ... well below unity, so the linear law
holds; :func:`stokes_regime_report` makes that check explicit rather than
assumed.

Reference: Batchelor, *An Introduction to Fluid Dynamics*,
doi:10.1017/CBO9780511800955, section 4.9.

Wall corrections
----------------
Near a wall the drag increases.  :func:`faxen_wall_correction` implements the
first-order Faxen result for motion *parallel* to a plane wall,

    F_parallel = F_stokes / (1 - (9/16)*(r/h) + (1/8)*(r/h)^3
                              - (45/256)*(r/h)^4 - (1/16)*(r/h)^5)

Reference: Happel & Brenner, *Low Reynolds Number Hydrodynamics*,
doi:10.1007/978-94-009-8352-6, section 7.2.  It is **off by default** because
it diverges as ``r -> h`` and needs a lubrication cut-off to be usable in a
trajectory integration.
"""

from __future__ import annotations

import warnings

import numpy as np

from biosim_lab.core.particles import ParticleState
from biosim_lab.core.plugin import RegimeWarning


def stokes_drag(
    u_fluid: np.ndarray, u_particle: np.ndarray, radius: np.ndarray, viscosity: float
) -> np.ndarray:
    """Stokes drag ``F = 6*pi*mu*r*(u_f - u_p)`` [N].

    Parameters
    ----------
    u_fluid, u_particle:
        Velocities, shape ``(n, dim)`` [m/s].
    radius:
        Particle radii, shape ``(n,)`` [m].
    viscosity:
        Dynamic viscosity [Pa*s].
    """
    return (
        6.0
        * np.pi
        * float(viscosity)
        * np.asarray(radius, dtype=float)[:, None]
        * (np.asarray(u_fluid, dtype=float) - np.asarray(u_particle, dtype=float))
    )


def mobility(radius: np.ndarray, viscosity: float) -> np.ndarray:
    """Stokes mobility ``1/(6*pi*mu*r)`` [m/(N*s)]."""
    return 1.0 / (6.0 * np.pi * float(viscosity) * np.asarray(radius, dtype=float))


def particle_reynolds_number(
    u_rel: np.ndarray, radius: np.ndarray, rho_f: float, viscosity: float
) -> np.ndarray:
    """``Re_p = rho_f * |u_rel| * 2r / mu`` (dimensionless)."""
    speed = np.linalg.norm(np.atleast_2d(np.asarray(u_rel, dtype=float)), axis=-1)
    return rho_f * speed * 2.0 * np.asarray(radius, dtype=float) / float(viscosity)


def channel_reynolds_number(
    mean_velocity: float, hydraulic_diameter: float, rho_f: float, viscosity: float
) -> float:
    """``Re = rho_f * U * D_h / mu`` for the channel itself (dimensionless)."""
    return float(rho_f * mean_velocity * hydraulic_diameter / viscosity)


def hydraulic_diameter(width: float, height: float) -> float:
    """Hydraulic diameter of a rectangular duct, ``4A/P = 2wh/(w+h)`` [m]."""
    return 2.0 * width * height / (width + height)


def faxen_wall_correction(radius: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Drag enhancement factor for motion parallel to a plane wall (dimensionless).

    Returns the multiplier applied to the Stokes drag.  Values are clipped at a
    gap of ``0.51*r`` (i.e. ``r/h <= 0.51``) because the series diverges on
    contact; treat results near the clip as qualitative.

    doi:10.1007/978-94-009-8352-6, section 7.2.
    """
    beta = np.clip(np.asarray(radius, dtype=float) / np.asarray(distance, dtype=float),
                   0.0, 0.51)
    denom = (
        1.0
        - (9.0 / 16.0) * beta
        + (1.0 / 8.0) * beta**3
        - (45.0 / 256.0) * beta**4
        - (1.0 / 16.0) * beta**5
    )
    return 1.0 / denom


def stokes_regime_report(
    *,
    mean_velocity: float,
    width: float,
    height: float,
    radius: np.ndarray | float,
    rho_f: float,
    viscosity: float,
    warn: bool = True,
) -> dict[str, float]:
    """Check that the creeping-flow assumptions hold; warn when they do not.

    Returns a dictionary with ``channel_reynolds``, ``particle_reynolds`` and
    ``hydraulic_diameter_m``.  A channel Reynolds number above ~2000 means
    turbulence (never reached in these devices); above ~1 the Stokes solver
    is no longer the right model and the OpenFOAM back-end should be used.
    """
    d_h = hydraulic_diameter(width, height)
    re_channel = channel_reynolds_number(mean_velocity, d_h, rho_f, viscosity)
    re_particle = float(
        np.max(
            particle_reynolds_number(
                np.array([[mean_velocity, 0.0]]), np.atleast_1d(radius), rho_f, viscosity
            )
        )
    )
    if warn:
        if re_channel > 1.0:
            warnings.warn(
                f"channel Reynolds number {re_channel:.3g} > 1: inertial terms are no longer "
                "negligible; install the OpenFOAM back-end for a Navier-Stokes solution.",
                RegimeWarning,
                stacklevel=2,
            )
        if re_particle > 1.0:
            warnings.warn(
                f"particle Reynolds number {re_particle:.3g} > 1: Stokes' law over-predicts "
                "drag; apply a Schiller-Naumann correction.",
                RegimeWarning,
                stacklevel=2,
            )
    return {
        "channel_reynolds": re_channel,
        "particle_reynolds": re_particle,
        "hydraulic_diameter_m": d_h,
    }


def make_drag_force(
    fluid_velocity, viscosity: float, *, wall_correction: bool = False, wall_gap: float = 0.0
):
    """Build a drag :class:`~biosim_lab.core.particles.ForceModel`.

    Only needed in ``mode="inertial"``; the overdamped tracker folds drag into
    the mobility and must **not** also be given this force, or drag would be
    counted twice.
    """

    def drag(state: ParticleState) -> np.ndarray:
        u = np.asarray(fluid_velocity(state.t, state.x), dtype=float)
        force = stokes_drag(u, state.v, state.radius, viscosity)
        if wall_correction and wall_gap > 0:
            gap = np.minimum(state.x[:, 1], wall_gap - state.x[:, 1])
            force = force * faxen_wall_correction(state.radius, np.maximum(gap, 1e-12))[:, None]
        return force

    drag.name = "stokes_drag"  # type: ignore[attr-defined]
    return drag


__all__ = [
    "stokes_drag",
    "mobility",
    "particle_reynolds_number",
    "channel_reynolds_number",
    "hydraulic_diameter",
    "faxen_wall_correction",
    "stokes_regime_report",
    "make_drag_force",
]
