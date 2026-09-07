"""Steady Stokes flow (creeping flow) on a 2-D mesh.

Governing equations
-------------------
At the Reynolds numbers of a microfluidic channel (``Re = rho U d / mu``, of
order 1e-2 for 5 uL/min in a 300 x 50 um channel) the inertial term is
negligible and the Navier-Stokes equations reduce to Stokes flow:

    -mu * laplacian(u) + grad(p) = f
    div(u) = 0

Discretised with the Taylor-Hood P2/P1 element pair, which is inf-sup stable
(Brezzi-Babuska); equal-order pairs would need pressure stabilisation.

Reference: Elman, Silvester & Wathen, *Finite Elements and Fast Iterative
Solvers*, 2nd ed., doi:10.1093/acprof:oso/9780199678792.001.0001, ch. 3.

Scope
-----
This solver exists for geometries where the analytic profile does not apply
(expansions, side channels, obstacles).  For a straight rectangular duct use
:func:`biosim_lab.instruments.saw_sorter.flow.poiseuille_rectangular`, which is
exact and far cheaper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    condense,
)
from skfem.helpers import ddot, div, sym_grad

from biosim_lab.core.geometry import MeshBundle
from biosim_lab.core.solver import BoundaryCondition, Solver


def _as_bc_list(bc: Any) -> list[BoundaryCondition]:
    if isinstance(bc, dict):
        return list(bc.values())
    if isinstance(bc, BoundaryCondition):
        return [bc]
    return list(bc)


class StokesSolver(Solver):
    """Taylor-Hood Stokes solver.

    Parameters
    ----------
    viscosity:
        Dynamic viscosity [Pa*s].

    Boundary conditions
    -------------------
    ``no_slip``
        ``u = 0`` on the given facets.
    ``velocity``
        Prescribed velocity vector; ``value`` is either a length-2 sequence or a
        callable ``f(x) -> (2, n)`` array [m/s].
    ``outflow``
        Natural (do-nothing) condition; the pressure level is fixed by pinning
        one pressure DOF.
    """

    name = "builtin_stokes"
    kind = "flow"
    required_modules = ("skfem", "scipy")

    def __init__(self, viscosity: float, **options: Any) -> None:
        super().__init__(**options)
        if viscosity <= 0:
            raise ValueError("viscosity must be positive")
        self.viscosity = float(viscosity)
        self._bundle: MeshBundle | None = None
        self._ubasis: Basis | None = None
        self._pbasis: Basis | None = None
        self._K = None
        self._f = None
        self._D: np.ndarray | None = None
        self._x0: np.ndarray | None = None
        self._solution: np.ndarray | None = None

    def setup(self, mesh: MeshBundle, bc: Any) -> None:
        """Assemble the saddle-point system."""
        if not isinstance(mesh, MeshBundle):
            raise TypeError("StokesSolver expects a MeshBundle")
        self._bundle = mesh
        ubasis = Basis(mesh.mesh, ElementVector(ElementTriP2()))
        # with_element() reuses the velocity quadrature, which the mixed form needs.
        pbasis = ubasis.with_element(ElementTriP1())
        self._ubasis, self._pbasis = ubasis, pbasis
        mu = self.viscosity

        @BilinearForm
        def viscous(u, v, _):
            return 2.0 * mu * ddot(sym_grad(u), sym_grad(v))

        @BilinearForm
        def divergence(u, q, _):
            return -div(u) * q

        A = viscous.assemble(ubasis)
        B = divergence.assemble(ubasis, pbasis)
        nu, npq = ubasis.N, pbasis.N
        K = sp.bmat([[A, B.T], [B, None]], format="csr")
        f = np.zeros(nu + npq)

        x0 = np.zeros(nu + npq)
        constrained: list[int] = []
        for cond in _as_bc_list(bc):
            facets = mesh.boundaries.get(cond.where)
            if facets is None or len(facets) == 0:
                raise KeyError(
                    f"boundary group {cond.where!r} not in mesh "
                    f"(have: {sorted(mesh.boundaries)})"
                )
            kind = cond.kind.lower()
            if kind == "outflow":
                continue
            if kind not in ("no_slip", "velocity", "dirichlet"):
                raise ValueError(
                    f"unknown flow BC kind {cond.kind!r}; expected no_slip, velocity "
                    "or outflow"
                )
            view = ubasis.get_dofs(facets)
            constrained.extend(int(d) for d in view.all())
            if kind in ("velocity", "dirichlet"):
                # skfem names vector components 'u^1' (x) and 'u^2' (y); ask for each
                # component separately rather than guessing the interleaving.
                for component, name in enumerate(("u^1", "u^2")):
                    comp_dofs = view.all(name)
                    if comp_dofs.size == 0:
                        continue
                    locs = ubasis.doflocs[:, comp_dofs]
                    if callable(cond.value):
                        vals = np.asarray(cond.value(locs), dtype=float)[component]
                    else:
                        vals = np.full(comp_dofs.size, float(np.asarray(cond.value)[component]))
                    x0[comp_dofs] = vals

        # Pin one pressure DOF to remove the constant-pressure null space.
        constrained.append(nu)
        self._K, self._f = K, f
        self._D = np.unique(np.asarray(constrained, dtype=int))
        self._x0 = x0
        self._solution = None

    def run(self) -> None:
        """Solve the saddle-point system with a direct sparse solver."""
        if self._K is None:
            raise RuntimeError("call setup() before run()")
        K, f, x, I = condense(self._K, self._f, x=self._x0, D=self._D)
        x = x.copy()
        x[I] = spla.spsolve(K.tocsc(), f)
        self._solution = x

    def fields(self) -> xr.Dataset:
        """Velocity [m/s] and pressure [Pa] at the mesh vertices."""
        if self._solution is None or self._ubasis is None or self._pbasis is None:
            raise RuntimeError("call run() before fields()")
        nu = self._ubasis.N
        u = self._solution[:nu]
        p = self._solution[nu:]
        nv = self._bundle.mesh.p.shape[1]  # type: ignore[union-attr]
        ux = u[0 : 2 * nv : 2]
        uy = u[1 : 2 * nv : 2]
        ds = xr.Dataset(
            {
                "u_x": ("node", ux),
                "u_y": ("node", uy),
                "pressure": ("node", p[:nv]),
            },
            coords={
                "x": ("node", self._bundle.mesh.p[0]),  # type: ignore[union-attr]
                "y": ("node", self._bundle.mesh.p[1]),  # type: ignore[union-attr]
            },
        )
        ds["u_x"].attrs["units"] = ds["u_y"].attrs["units"] = "m/s"
        ds["pressure"].attrs["units"] = "Pa"
        ds.attrs.update(solver=self.name, viscosity_Pa_s=self.viscosity)
        return ds


__all__ = ["StokesSolver"]
