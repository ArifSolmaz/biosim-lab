"""Analytic pressure-driven flow in a rectangular microchannel.

Why analytic
------------
The channel is straight and the Reynolds number is ~1e-2, so the fully
developed velocity profile has an exact Fourier-series solution.  Meshing and
solving Stokes here would be slower *and* less accurate, so the sorter uses the
series by default; :class:`~biosim_lab.core.fem.stokes.StokesSolver` remains
available for non-trivial geometries.

Solution
--------
For a duct ``|x| <= a``, ``|y| <= b`` (``a = W/2``, ``b = H/2``) with axial
pressure gradient ``G = -dp/dz``:

    u_z(x, y) = (16 b^2 G) / (mu * pi^3)
                * sum_{n odd} (-1)^((n-1)/2) / n^3
                  * [ 1 - cosh(n*pi*x / (2b)) / cosh(n*pi*a / (2b)) ]
                  * cos(n*pi*y / (2b))

with the flow rate obtained term by term:

    Q = (64 b^3 G) / (mu * pi^4)
        * sum_{n odd} [ 2a - (4b / (n*pi)) * tanh(n*pi*a / (2b)) ] / n^4

Reference: White, *Viscous Fluid Flow*, 3rd ed., section 3-3.3
(ISBN 978-0072402315); the same series appears as eq. 2 in Bruus,
*Theoretical Microfluidics*, doi:10.1093/oso/9780199235094.001.0001, ch. 3.

Coordinate convention used by the sorter
----------------------------------------
``x`` is the acoustic (transverse) axis across the channel **width** ``W``,
``y`` is the channel **height** ``H``, and ``z`` is the flow direction.  Cells
are tracked in the ``(x, y)`` cross-section while being advected along ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _odd_terms(n_terms: int) -> np.ndarray:
    return np.arange(1, 2 * int(n_terms), 2, dtype=float)


@dataclass
class RectangularPoiseuille:
    """Fully developed flow in a rectangular duct of size ``width x height``.

    Parameters
    ----------
    width, height:
        Cross-section [m]. ``width`` lies along the acoustic axis ``x``.
    viscosity:
        Dynamic viscosity [Pa*s].
    flow_rate:
        Volumetric flow rate [m^3/s]. The pressure gradient is back-solved from
        it, which is what a syringe pump actually controls.
    n_terms:
        Number of odd Fourier terms. 30 terms give better than 1e-6 relative
        accuracy for aspect ratios up to 20:1.
    """

    width: float
    height: float
    viscosity: float
    flow_rate: float
    n_terms: int = 30

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.viscosity) <= 0:
            raise ValueError("width, height and viscosity must be positive")
        if self.n_terms < 1:
            raise ValueError("n_terms must be >= 1")
        self._a = 0.5 * self.width
        self._b = 0.5 * self.height
        self._n = _odd_terms(self.n_terms)
        self._pressure_gradient = self._solve_pressure_gradient()

    # -- derived quantities ----------------------------------------------
    def _shape_sum_for_q(self) -> float:
        n, a, b = self._n, self._a, self._b
        return float(
            np.sum((2.0 * a - (4.0 * b / (n * np.pi)) * np.tanh(n * np.pi * a / (2.0 * b)))
                   / n**4)
        )

    def _solve_pressure_gradient(self) -> float:
        """Axial pressure gradient ``G = -dp/dz`` [Pa/m] giving *flow_rate*."""
        q_per_g = 64.0 * self._b**3 / (self.viscosity * np.pi**4) * self._shape_sum_for_q()
        if q_per_g <= 0:  # pragma: no cover - guarded by __post_init__
            raise RuntimeError("degenerate channel geometry")
        return float(self.flow_rate / q_per_g)

    @property
    def pressure_gradient(self) -> float:
        """``-dp/dz`` [Pa/m]."""
        return self._pressure_gradient

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area [m^2]."""
        return self.width * self.height

    @property
    def mean_velocity(self) -> float:
        """Bulk mean velocity ``Q / A`` [m/s]."""
        return self.flow_rate / self.cross_section_area

    @property
    def max_velocity(self) -> float:
        """Centreline velocity [m/s]."""
        return float(
            self.velocity(np.array([0.5 * self.width]), np.array([0.5 * self.height]))[0]
        )

    def transit_time(self, length: float) -> float:
        """Mean residence time over a channel of *length* [s]."""
        return float(length) / self.mean_velocity

    # -- fields -----------------------------------------------------------
    def velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Axial velocity ``u_z`` [m/s] at channel coordinates ``(x, y)``.

        *x* and *y* are measured from the channel corner: ``0 <= x <= width``,
        ``0 <= y <= height`` (the same frame the particle tracker uses).
        """
        xc = np.asarray(x, dtype=float) - self._a
        yc = np.asarray(y, dtype=float) - self._b
        n = self._n
        a, b = self._a, self._b
        sign = np.where(((n - 1) / 2) % 2 == 0, 1.0, -1.0)

        shape = n[:, None] * np.pi / (2.0 * b)
        # cosh(u)/cosh(v) overflows for high aspect ratios (n*pi*a/2b can exceed 700),
        # so evaluate the ratio in log space:
        #   cosh(u)/cosh(v) = exp(u - v) * (1 + exp(-2u)) / (1 + exp(-2v))
        u = np.abs(shape * xc.ravel()[None, :])
        v = (n * np.pi * a / (2.0 * b))[:, None]
        cosh_ratio = np.exp(u - v) * (1.0 + np.exp(-2.0 * u)) / (1.0 + np.exp(-2.0 * v))
        series = (
            (sign / n**3)[:, None]
            * (1.0 - cosh_ratio)
            * np.cos(shape * yc.ravel()[None, :])
        )
        prefactor = 16.0 * b**2 * self._pressure_gradient / (self.viscosity * np.pi**3)
        u = prefactor * series.sum(0)
        return u.reshape(np.shape(x))

    def velocity_field(self, t: float, positions: np.ndarray) -> np.ndarray:
        """Fluid velocity for the tracker: ``(n, 2)`` array in the ``(x, y)`` plane.

        The cross-sectional flow is zero — the Poiseuille flow is entirely along
        ``z`` — so this returns zeros and the axial advection is handled
        separately by :meth:`axial_velocity`.  Acoustic streaming, when a
        streaming back-end is installed, is added here.
        """
        return np.zeros((np.asarray(positions).shape[0], 2), dtype=float)

    def axial_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Axial (``z``) velocity [m/s] for particles at ``(n, 2)`` positions."""
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        return self.velocity(pos[:, 0], pos[:, 1])

    def profile(self, nx: int = 121, ny: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the profile on a grid; returns ``(x, y, u)``."""
        xs = np.linspace(0.0, self.width, int(nx))
        ys = np.linspace(0.0, self.height, int(ny))
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return xs, ys, self.velocity(gx, gy)

    def verify_flow_rate(self, nx: int = 401, ny: int = 201) -> float:
        """Numerically integrate the profile and return the relative error in ``Q``.

        A mass-conservation self-check: the series pressure gradient is derived
        analytically, so this compares two independent routes to the same ``Q``.
        """
        xs, ys, u = self.profile(nx, ny)
        q = np.trapezoid(np.trapezoid(u, ys, axis=1), xs)
        return float(abs(q - self.flow_rate) / self.flow_rate)


__all__ = ["RectangularPoiseuille"]
