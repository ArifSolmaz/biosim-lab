"""Electro-quasistatic (complex-conductivity) FEM for impedance sensing.

Governing equation
------------------
Below the wavelength limit (a 100 kHz - 10 MHz impedance measurement across a
100 um electrode gap is deeply quasistatic) the electric field is curl-free and
the complex potential ``phi`` satisfies the current-continuity equation

    div( sigma* grad(phi) ) = 0,     sigma* = sigma + i * omega * eps0 * eps_r

with ``sigma`` [S/m] the conductivity and ``eps_r`` the relative permittivity.
This is the standard formulation behind electric cell-substrate impedance
sensing (ECIS) and the RTCA / xCELLigence Cell Index.

Reference: Giaever & Keese (1991), *Micromotion of mammalian cells measured
electrically*, PNAS 88:7896, doi:10.1073/pnas.88.17.7896.

Boundary conditions
-------------------
``dirichlet``
    Prescribed potential on an electrode (the drive terminal at ``V``, the
    counter electrode at 0).
``insulating``
    ``dphi/dn = 0`` — the natural condition on the substrate between fingers.
``contact_impedance``
    Robin condition modelling a per-area electrode-electrolyte interface
    impedance ``z_s`` [Ohm*m^2] in series with the bulk:
    ``sigma* dphi/dn = (phi_electrode - phi) / z_s``.
    Needed because the double-layer capacitance dominates ECIS spectra below
    ~10 kHz.
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

EPS0 = 8.8541878128e-12
"""Vacuum permittivity [F/m]. CODATA 2018, doi:10.1103/RevModPhys.93.025010."""


def _as_bc_list(bc: Any) -> list[BoundaryCondition]:
    if isinstance(bc, dict):
        return list(bc.values())
    if isinstance(bc, BoundaryCondition):
        return [bc]
    return list(bc)


class ElectroQuasistaticSolver(Solver):
    """Complex-potential solver for interdigital-electrode impedance.

    Parameters
    ----------
    frequency:
        Measurement frequency [Hz].
    conductivity, permittivity_rel:
        Bulk electrolyte properties. When a cell layer is present, pass the
        *effective* values from the shell model (see
        ``instruments/impedance_rtca/physics.py``).
    element_order:
        1 or 2; P1 is enough because only the integrated current is reported.
    """

    name = "builtin_electroquasistatic"
    kind = "electro"
    required_modules = ("skfem", "scipy")

    def __init__(
        self,
        frequency: float,
        conductivity: float,
        permittivity_rel: float,
        *,
        element_order: int = 1,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.frequency = float(frequency)
        self.conductivity = float(conductivity)
        self.permittivity_rel = float(permittivity_rel)
        self.element_order = element_order

        self._bundle: MeshBundle | None = None
        self._basis: Basis | None = None
        self._A = None
        self._b = None
        self._dirichlet: dict[int, complex] = {}
        self._phi: np.ndarray | None = None
        self._drive: tuple[str, complex] | None = None

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2.0 * np.pi * self.frequency

    @property
    def sigma_star(self) -> complex:
        """Complex conductivity ``sigma + i*omega*eps0*eps_r`` [S/m]."""
        return self.conductivity + 1j * self.omega * EPS0 * self.permittivity_rel

    # -- Solver contract --------------------------------------------------
    def setup(self, mesh: MeshBundle, bc: Any) -> None:
        """Assemble ``div(sigma* grad phi) = 0``."""
        if not isinstance(mesh, MeshBundle):
            raise TypeError("ElectroQuasistaticSolver expects a MeshBundle")
        self._bundle = mesh
        element = ElementTriP2() if self.element_order == 2 else ElementTriP1()
        basis = Basis(mesh.mesh, element)
        self._basis = basis
        sigma = self.sigma_star

        @BilinearForm(dtype=complex)
        def laplace(u, v, _):
            return sigma * dot(grad(u), grad(v))

        A = laplace.assemble(basis)
        b = np.zeros(basis.N, dtype=complex)
        dirichlet: dict[int, complex] = {}
        drive: tuple[str, complex] | None = None

        for cond in _as_bc_list(bc):
            facets = mesh.boundaries.get(cond.where)
            if facets is None or len(facets) == 0:
                raise KeyError(
                    f"boundary group {cond.where!r} not in mesh "
                    f"(have: {sorted(mesh.boundaries)})"
                )
            kind = cond.kind.lower()
            if kind == "insulating":
                continue
            if kind in ("dirichlet", "potential"):
                dofs = basis.get_dofs(facets).all()
                value = complex(cond.value)
                for dof in dofs:
                    dirichlet[int(dof)] = value
                if value != 0 and drive is None:
                    drive = (cond.where, value)
            elif kind == "contact_impedance":
                z_s = complex(cond.meta.get("z_s", cond.value))
                phi_e = complex(cond.meta.get("potential", 0.0))
                if z_s == 0:
                    raise ValueError("contact_impedance needs a non-zero z_s [Ohm*m^2]")
                fbasis = FacetBasis(mesh.mesh, element, facets=facets)

                @BilinearForm(dtype=complex)
                def robin(u, v, _w, _g=1.0 / z_s):
                    return _g * u * v

                @LinearForm(dtype=complex)
                def load(v, _w, _g=1.0 / z_s, _phi=phi_e):
                    return _g * _phi * v

                A = A + robin.assemble(fbasis)
                b += load.assemble(fbasis)
                if phi_e != 0 and drive is None:
                    drive = (cond.where, phi_e)
            else:
                raise ValueError(
                    f"unknown electro BC kind {cond.kind!r}; expected dirichlet, "
                    "insulating or contact_impedance"
                )

        self._A, self._b, self._dirichlet, self._drive = A, b, dirichlet, drive
        self._phi = None

    def run(self) -> None:
        """Solve for the complex potential."""
        if self._A is None or self._basis is None:
            raise RuntimeError("call setup() before run()")
        if self._dirichlet:
            dofs = np.fromiter(self._dirichlet.keys(), dtype=int)
            x = np.zeros(self._basis.N, dtype=complex)
            x[dofs] = np.fromiter(self._dirichlet.values(), dtype=complex)
            K, f, u, I = condense(self._A, self._b, x=x, D=dofs)
            u = u.astype(complex)
            u[I] = spla.spsolve(K.tocsc(), f)
            self._phi = u
        else:
            self._phi = spla.spsolve(self._A.tocsc(), self._b)

    def fields(self) -> xr.Dataset:
        """Nodal complex potential [V]."""
        if self._phi is None or self._basis is None:
            raise RuntimeError("call run() before fields()")
        locs = self._basis.doflocs
        ds = xr.Dataset(
            {
                "phi_real": ("dof", self._phi.real),
                "phi_imag": ("dof", self._phi.imag),
                "phi_abs": ("dof", np.abs(self._phi)),
            },
            coords={"x": ("dof", locs[0]), "y": ("dof", locs[1])},
        )
        for name in ("phi_real", "phi_imag", "phi_abs"):
            ds[name].attrs["units"] = "V"
        ds.attrs.update(
            solver=self.name,
            frequency_Hz=self.frequency,
            conductivity_S_m=self.conductivity,
            permittivity_rel=self.permittivity_rel,
        )
        return ds

    # -- post-processing --------------------------------------------------
    def terminal_current(self, electrode: str, depth: float = 1.0) -> complex:
        """Total complex current [A] flowing out of *electrode*.

        Computed as ``I = integral( sigma* grad(phi) . n ) dS`` over the
        electrode facets, times *depth* (the out-of-plane electrode length,
        because the mesh is a 2-D cross-section).
        """
        if self._phi is None or self._basis is None or self._bundle is None:
            raise RuntimeError("call run() before terminal_current()")
        facets = self._bundle.boundaries[electrode]
        element = ElementTriP2() if self.element_order == 2 else ElementTriP1()
        fbasis = FacetBasis(self._bundle.mesh, element, facets=facets)

        @LinearForm(dtype=complex)
        def flux(v, w, _sigma=self.sigma_star):
            return _sigma * dot(grad(w["phi"]), w.n) * v

        current = flux.assemble(fbasis, phi=fbasis.interpolate(self._phi)).sum()
        return complex(current) * float(depth)

    def impedance(self, depth: float = 1.0) -> complex:
        """Complex impedance ``Z = V / I`` [Ohm] seen by the drive electrode."""
        if self._drive is None:
            raise RuntimeError(
                "no driven electrode: give one boundary a non-zero dirichlet potential"
            )
        electrode, voltage = self._drive
        current = self.terminal_current(electrode, depth=depth)
        if current == 0:
            raise ZeroDivisionError("terminal current is exactly zero")
        return voltage / current


__all__ = ["ElectroQuasistaticSolver", "EPS0"]
