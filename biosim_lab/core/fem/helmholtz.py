"""Time-harmonic acoustic pressure: the Helmholtz equation on a 2-D mesh.

Governing equation
------------------
For a lossless, quiescent, linear fluid the first-order acoustic pressure
``p(x, y) exp(-i omega t)`` satisfies

    laplacian(p) + k^2 p = 0,      k = omega / c  [1/m]

Weak form used here (Galerkin, test function q):

    integral( grad(p) . grad(q) )  -  k^2 integral( p q )
        =  integral_Gamma( dp/dn * q )

Boundary conditions
-------------------
``rigid``
    ``dp/dn = 0`` — the natural condition, nothing is assembled.
``velocity``
    A prescribed normal velocity ``v_n`` (e.g. the surface displacement of a
    SAW substrate leaking into the fluid).  The linear-momentum equation
    ``rho dv/dt = -grad(p)`` gives, for the ``exp(-i omega t)`` convention,

        dp/dn = i * omega * rho * v_n

    Reference: Pierce, *Acoustics: An Introduction to Its Physical Principles
    and Applications*, 3rd ed., doi:10.1007/978-3-030-11214-1, ch. 1.
``pressure``
    Dirichlet ``p = p0``; ``p0 = 0`` is the pressure-release ("soft") wall, a
    good model for a fluid/air interface.
``impedance``
    Robin condition ``dp/dn = i * omega * rho * p / Z`` with specific acoustic
    impedance ``Z`` [Pa*s/m].  Setting ``Z = rho_wall * c_wall`` models a
    partially absorbing wall such as PDMS, whose impedance is close to water's
    and therefore leaks strongly.

Sign convention
---------------
Throughout biosim-lab the time factor is ``exp(-i omega t)``, so a
progressive wave in ``+x`` is ``exp(+i k x)``.  Mixing this with the
``exp(+i omega t)`` convention flips the sign of every imaginary part.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
import xarray as xr
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    FacetBasis,
    LinearForm,
    condense,
)
from skfem.helpers import dot, grad

from biosim_lab.core.geometry import MeshBundle
from biosim_lab.core.solver import BoundaryCondition, Solver


def _as_bc_list(bc: Any) -> list[BoundaryCondition]:
    if isinstance(bc, dict):
        return list(bc.values())
    if isinstance(bc, BoundaryCondition):
        return [bc]
    return list(bc)


def _evaluate(value: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate a BC value that may be a scalar or a callable of the coordinates."""
    out = value(x) if callable(value) else value
    return np.asarray(out, dtype=complex) * np.ones(x.shape[1:], dtype=complex)


class HelmholtzSolver(Solver):
    """Complex-valued Helmholtz solver on a triangular mesh.

    Parameters
    ----------
    frequency:
        Drive frequency [Hz].
    density, speed_of_sound:
        Fluid properties [kg/m^3], [m/s].
    damping:
        Dimensionless loss factor ``eta``; the wave number becomes
        ``k = omega / c * (1 + i*eta/2)``.  Classical thermoviscous absorption in
        water at 20 MHz over a 300 um channel is negligible (alpha ~ 0.09 Np/m
        at 20 MHz, doi:10.1121/1.1907120), so the default is 0 and any non-zero
        value should be justified as an effective device loss.
    element_order:
        1 (P1) or 2 (P2).  P2 is the default because the Gor'kov force needs a
        differentiable pressure field.
    """

    name = "builtin_helmholtz"
    kind = "acoustics"
    required_modules = ("skfem", "scipy")

    def __init__(
        self,
        frequency: float,
        density: float,
        speed_of_sound: float,
        *,
        damping: float = 0.0,
        element_order: int = 2,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        if element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")
        self.frequency = float(frequency)
        self.density = float(density)
        self.speed_of_sound = float(speed_of_sound)
        self.damping = float(damping)
        self.element_order = element_order

        self._bundle: MeshBundle | None = None
        self._basis: Basis | None = None
        self._A = None
        self._b = None
        self._dirichlet: dict[int, complex] | None = None
        self._p: np.ndarray | None = None

    # -- derived quantities ----------------------------------------------
    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def wavenumber(self) -> complex:
        """Complex wave number ``k`` [1/m]."""
        return self.omega / self.speed_of_sound * (1.0 + 0.5j * self.damping)

    @property
    def wavelength(self) -> float:
        """Acoustic wavelength in the fluid [m]."""
        return self.speed_of_sound / self.frequency

    # -- Solver contract --------------------------------------------------
    def setup(self, mesh: MeshBundle, bc: Any) -> None:
        """Assemble the Helmholtz system for *mesh* with boundary conditions *bc*."""
        if not isinstance(mesh, MeshBundle):
            raise TypeError("HelmholtzSolver expects a MeshBundle from core.geometry")
        self._bundle = mesh
        element = ElementTriP2() if self.element_order == 2 else ElementTriP1()
        basis = Basis(mesh.mesh, element)
        self._basis = basis

        k2 = self.wavenumber**2

        @BilinearForm(dtype=complex)
        def stiffness(u, v, _):
            return dot(grad(u), grad(v)) - k2 * u * v

        A = stiffness.assemble(basis)
        b = np.zeros(basis.N, dtype=complex)
        dirichlet: dict[int, complex] = {}

        for cond in _as_bc_list(bc):
            facets = mesh.boundaries.get(cond.where)
            if facets is None or len(facets) == 0:
                raise KeyError(
                    f"boundary group {cond.where!r} not in mesh "
                    f"(have: {sorted(mesh.boundaries)})"
                )
            kind = cond.kind.lower()
            if kind == "rigid":
                continue  # natural BC: dp/dn = 0
            fbasis = FacetBasis(mesh.mesh, element, facets=facets)

            if kind == "velocity":
                # dp/dn = i * omega * rho * v_n   (Pierce, doi:10.1007/978-3-030-11214-1)
                coeff = 1j * self.omega * self.density

                @LinearForm(dtype=complex)
                def load(v, w, _value=cond.value, _coeff=coeff):
                    return _coeff * _evaluate(_value, w.x) * v

                b += load.assemble(fbasis)

            elif kind == "impedance":
                impedance = complex(cond.value)
                if impedance == 0:
                    raise ValueError("impedance BC needs a non-zero Z")
                coeff = 1j * self.omega * self.density / impedance

                @BilinearForm(dtype=complex)
                def robin(u, v, _w, _coeff=coeff):
                    return -_coeff * u * v

                A = A + robin.assemble(fbasis)

            elif kind in ("pressure", "dirichlet"):
                dofs = basis.get_dofs(facets).all()
                x = basis.doflocs[:, dofs]
                values = _evaluate(cond.value, x)
                for dof, val in zip(dofs, np.atleast_1d(values)):
                    dirichlet[int(dof)] = complex(val)
            else:
                raise ValueError(
                    f"unknown acoustic BC kind {cond.kind!r}; expected one of "
                    "rigid, velocity, impedance, pressure"
                )

        self._A, self._b, self._dirichlet = A, b, dirichlet
        self._p = None

    def run(self) -> None:
        """Solve the assembled linear system."""
        if self._A is None or self._basis is None:
            raise RuntimeError("call setup() before run()")
        x = np.zeros(self._basis.N, dtype=complex)
        if self._dirichlet:
            dofs = np.fromiter(self._dirichlet.keys(), dtype=int)
            x[dofs] = np.fromiter(self._dirichlet.values(), dtype=complex)
            system = condense(self._A, self._b, x=x, D=dofs)
            # skfem's default solver is real-only; call SciPy directly for complex.
            K, f, u, I = system
            u = u.astype(complex)
            u[I] = spla.spsolve(K.tocsc(), f)
            self._p = u
        else:
            self._p = spla.spsolve(self._A.tocsc(), self._b)
        self._fields = None

    def fields(self) -> xr.Dataset:
        """Nodal complex pressure as an :class:`xarray.Dataset`.

        Variables
        ---------
        ``p_real``, ``p_imag``, ``p_abs`` on the DOF locations, with ``x``/``y``
        coordinates in metres.
        """
        if self._p is None:
            raise RuntimeError("call run() before fields()")
        assert self._basis is not None
        locs = self._basis.doflocs
        ds = xr.Dataset(
            {
                "p_real": ("dof", self._p.real),
                "p_imag": ("dof", self._p.imag),
                "p_abs": ("dof", np.abs(self._p)),
            },
            coords={"x": ("dof", locs[0]), "y": ("dof", locs[1])},
        )
        for name in ("p_real", "p_imag", "p_abs"):
            ds[name].attrs["units"] = "Pa"
        ds["x"].attrs["units"] = "m"
        ds["y"].attrs["units"] = "m"
        ds.attrs.update(
            solver=self.name,
            frequency_Hz=self.frequency,
            wavelength_m=self.wavelength,
            density_kg_m3=self.density,
            speed_of_sound_m_s=self.speed_of_sound,
            element_order=self.element_order,
        )
        self._fields = ds
        return ds

    # -- post-processing --------------------------------------------------
    def sample_on_grid(self, nx: int = 201, ny: int = 41) -> xr.Dataset:
        """Interpolate the solution onto a regular ``(x, y)`` grid.

        A structured grid is what the Gor'kov post-processing needs: the force
        is a gradient of a quadratic functional of ``p``, so it is evaluated by
        central differences on a grid rather than by differentiating P2 shape
        functions twice.
        """
        if self._p is None or self._basis is None or self._bundle is None:
            raise RuntimeError("call run() before sample_on_grid()")
        pts = self._bundle.mesh.p
        xs = np.linspace(pts[0].min(), pts[0].max(), int(nx))
        ys = np.linspace(pts[1].min(), pts[1].max(), int(ny))
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        query = np.vstack([gx.ravel(), gy.ravel()])

        interp = self._basis.interpolator(self._p)
        values = np.asarray(interp(query), dtype=complex).reshape(gx.shape)
        # Points marginally outside the mesh come back as NaN; fill from the
        # nearest valid column so the gradient stencil stays well defined.
        values = _fill_nan_nearest(values)

        ds = xr.Dataset(
            {"p_real": (("x", "y"), values.real), "p_imag": (("x", "y"), values.imag)},
            coords={"x": xs, "y": ys},
        )
        ds["p_real"].attrs["units"] = ds["p_imag"].attrs["units"] = "Pa"
        ds.attrs.update(self.fields().attrs)
        return ds


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest finite value along each axis (cheap fill)."""
    out = a.copy()
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return out[tuple(idx)]


def analytic_standing_wave(
    x: np.ndarray, p0: float, wavelength: float, phase: float = 0.0
) -> np.ndarray:
    """One-dimensional standing pressure wave ``p = p0 * cos(k x + phase)``.

    The reference solution used to validate the FEM path
    (``examples/04_validate_analytic_vs_fem.py``).
    """
    k = 2.0 * np.pi / wavelength
    return p0 * np.cos(k * np.asarray(x, dtype=float) + phase)


__all__ = ["HelmholtzSolver", "analytic_standing_wave"]
