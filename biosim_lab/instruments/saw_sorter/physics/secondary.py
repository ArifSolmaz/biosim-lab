"""Optional secondary effects, **disabled by default**.

Each of these is real but either small in the default operating point or
expensive, so the platform makes them opt-in and states the cost:

* :func:`gravity_buoyancy` — sedimentation. For a 9 um MCF-7 cell in water the
  Stokes settling speed is ~4 um/s versus a 5.6 mm/s through-flow, i.e. a
  ~1 um drop over a 1 cm channel. Not negligible at very low flow rates.
* :func:`wall_repulsion` — a short-range soft repulsion that keeps trajectories
  off the walls. This is a **numerical device**, not a physical force
  (lubrication and electrostatic double-layer repulsion are the real
  mechanisms), and is labelled as such.
* :func:`secondary_bjerknes_force` — particle-particle acoustic interaction.
  Real and responsible for the cell chains seen in acoustophoresis videos, but
  ``O(N^2)`` and only relevant above ~1e6 cells/mL.
"""

from __future__ import annotations

import numpy as np

from biosim_lab.core.particles import ParticleState

G_EARTH = 9.80665
"""Standard gravity [m/s^2]. CGPM (1901) definition."""


def gravity_buoyancy(
    rho_f: float, *, axis: int = 1, gravity: float = G_EARTH, sign: float = -1.0
):
    """Net gravitational + buoyant force ``(rho_p - rho_f) * V * g`` [N].

    Parameters
    ----------
    rho_f:
        Fluid density [kg/m^3].
    axis:
        Index of the vertical coordinate (1 = ``y`` in a 2-D channel).
    sign:
        ``-1`` if ``+axis`` points up (the default), ``+1`` if it points down.
    """

    def force(state: ParticleState) -> np.ndarray:
        out = np.zeros_like(state.x)
        out[:, axis] = sign * (state.density - rho_f) * state.volume * gravity
        return out

    force.name = "gravity_buoyancy"  # type: ignore[attr-defined]
    return force


def wall_repulsion(
    bounds: tuple[tuple[float, float] | None, ...],
    *,
    stiffness: float = 1e-6,
    range_factor: float = 1.0,
):
    """Soft short-range wall repulsion [N] — a numerical regulariser.

    The force switches on when the surface-to-wall gap falls below
    ``range_factor * radius`` and grows linearly to ``stiffness * radius`` on
    contact.  It exists to keep the integrator from pushing particle centres
    through walls; it does **not** represent a measured physical interaction,
    and results should not depend on *stiffness* (vary it to confirm).

    Parameters
    ----------
    bounds:
        Per-axis ``(lo, hi)`` limits [m]; ``None`` for unbounded axes.
    stiffness:
        Force per unit overlap [N/m].
    """

    def force(state: ParticleState) -> np.ndarray:
        out = np.zeros_like(state.x)
        for axis, limits in enumerate(bounds):
            if limits is None or axis >= state.dim:
                continue
            lo, hi = limits
            cutoff = range_factor * state.radius
            gap_lo = state.x[:, axis] - lo - state.radius
            gap_hi = hi - state.x[:, axis] - state.radius
            out[:, axis] += stiffness * np.clip(cutoff - gap_lo, 0.0, None)
            out[:, axis] -= stiffness * np.clip(cutoff - gap_hi, 0.0, None)
        return out

    force.name = "wall_repulsion"  # type: ignore[attr-defined]
    return force


def secondary_bjerknes_force(
    *,
    p0: float,
    wavelength: float,
    rho_f: float,
    kappa_f: float,
    node_offset: float = 0.0,
    cutoff_radii: float = 20.0,
):
    """Secondary (inter-particle) Bjerknes force [N].

    Two scatterers in the same field re-scatter onto each other.  For particles
    small compared with both the wavelength and their separation ``d``, the
    time-averaged interaction along the line of centres is

        F_12 = 4*pi*a1^3*a2^3 / d^2
               * [ (rho_p - rho_f)^2 * (3*cos^2(theta) - 1) / (3*rho_f) * <v1 v2>
                   - omega^2 * rho_f * kappa_f^2 / 9 * <p1 p2> ]

    Reference: Crum (1975), *Bjerknes forces on bubbles in a stationary sound
    field*, JASA 57:1363, doi:10.1121/1.380614; applied to acoustophoresis of
    solid particles by Barnkob et al., doi:10.1103/PhysRevE.86.056307.

    Implementation notes
    --------------------
    * Cost is ``O(N^2)``; pairs beyond ``cutoff_radii`` particle radii are
      skipped.
    * The monopole (pressure) term is retained in full; the dipole term uses
      the local standing-wave velocity, which is exact for the 1-D SSAW field
      used elsewhere in this instrument.
    * This is an approximation valid for ``a << d << lambda``. It is **not**
      validated against experiment in this project and is therefore off by
      default.
    """
    k = 2.0 * np.pi / wavelength
    omega_over_c = k  # omega/c = k

    def force(state: ParticleState) -> np.ndarray:
        n = state.n
        out = np.zeros_like(state.x)
        if n < 2:
            return out
        # Pairwise separations.
        diff = state.x[:, None, :] - state.x[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        max_r = float(np.max(state.radius))
        near = dist < cutoff_radii * max_r
        if not near.any():
            return out

        a3 = state.radius**3
        # Local field amplitudes at each particle (1-D SSAW along axis 0).
        phase = k * (state.x[:, 0] - node_offset)
        p_loc = p0 * np.sin(phase)
        v_loc = p0 * np.cos(phase) / (rho_f * (omega_over_c / k) * 1.0) * kappa_f * 0.0
        # Velocity amplitude from  v = grad(p)/(rho*omega) = p0*k*cos(kx)/(rho*omega)
        v_loc = p0 * np.cos(phase) * np.sqrt(kappa_f / rho_f)

        drho = state.density - rho_f
        unit = np.zeros_like(diff)
        np.divide(diff, dist[:, :, None], out=unit, where=np.isfinite(dist[:, :, None]))
        cos_theta = unit[:, :, 0]

        pref = 4.0 * np.pi * a3[:, None] * a3[None, :] / np.where(near, dist**2, np.inf)
        dipole = (
            drho[:, None] * drho[None, :] * (3.0 * cos_theta**2 - 1.0)
            / (3.0 * rho_f)
            * 0.5 * v_loc[:, None] * v_loc[None, :]
        )
        monopole = -(k**2) * rho_f * kappa_f**2 / 9.0 * 0.5 * p_loc[:, None] * p_loc[None, :]
        magnitude = pref * (dipole + monopole)
        out = np.einsum("ij,ijk->ik", magnitude, unit)
        return out

    force.name = "secondary_bjerknes"  # type: ignore[attr-defined]
    return force


__all__ = ["gravity_buoyancy", "wall_repulsion", "secondary_bjerknes_force", "G_EARTH"]
