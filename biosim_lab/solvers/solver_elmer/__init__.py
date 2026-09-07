"""Elmer FEM back-end for the full piezoelectric IDT problem.

Stage 4 status: **interface skeleton and a ``.sif`` template only.**

What this back-end would add
----------------------------
Stage 1 treats the substrate surface displacement ``u0`` as an *input*: the
FEM model imposes a prescribed normal velocity on the fluid's bottom boundary
and renormalises the solved field to a measured pressure amplitude
(``fem_model.PRESSURE_PER_VOLT_ASSUMPTION`` is explicitly flagged as a fitted
device constant, not physics).

Solving the coupled piezoelectric problem removes that assumption.  Elmer
solves the linear piezoelectric constitutive equations

    T = c^E : S - e^T . E
    D = e : S + eps^S . E

on the LiNbO3 substrate with the interdigital electrodes as electrical
boundary conditions, giving the surface displacement field directly from the
drive voltage — including the IDT's finite finger count, apodisation and the
electrical input admittance.

Reference for the constitutive model and the 128 deg YX LiNbO3 constants:
Warner, Onoe & Coquin (1967), *Determination of elastic and piezoelectric
constants for crystals in class (3m)*, JASA 42:1223, doi:10.1121/1.1910709.
Elmer itself: Malinen & Raback (2013), *Elmer finite element solver for
multiphysics and multiscale problems*, in *Multiscale Modelling Methods for
Applications in Materials Science*, ISBN 978-3-89336-899-0.

Bridge contract
---------------
``fields()`` returns ``u_x``, ``u_y``, ``u_z`` on the substrate surface as a
function of ``x``; :func:`surface_displacement_to_bc` converts that into the
prescribed-velocity boundary condition Stage 1's
:class:`~biosim_lab.instruments.saw_sorter.fem_model.SAWFieldModel` already
accepts, so switching from the assumed calibration to the solved one changes
one line in the sorter and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from biosim_lab.core.solver import BoundaryCondition, Solver

TEMPLATE_DIR = Path(__file__).parent / "templates"


class ElmerSolver(Solver):
    """Elmer bridge for the piezoelectric IDT problem.

    Availability requires ``ElmerSolver`` and ``ElmerGrid`` on ``PATH``; the
    Docker image ``biosim-lab-elmer`` provides them and the main image does not.
    """

    name = "elmer"
    kind = "piezo"
    required_executables = ("ElmerSolver", "ElmerGrid")

    def __init__(self, case_dir: str | Path = "elmer_case", **options: Any) -> None:
        super().__init__(**options)
        self.case_dir = Path(case_dir)

    @staticmethod
    def templates() -> dict[str, Path]:
        """Paths to the shipped ``.sif`` templates."""
        return {p.name: p for p in sorted(TEMPLATE_DIR.glob("*"))}

    def setup(
        self, mesh: Any, bc: dict[str, BoundaryCondition] | list[BoundaryCondition]
    ) -> None:
        raise NotImplementedError(
            "Elmer case generation is Stage 4 and not implemented. The .sif template "
            f"with the 128 deg YX LiNbO3 material block is in {TEMPLATE_DIR}. Stage 1 "
            "runs without it by treating the substrate displacement as an input; see "
            "biosim_lab.instruments.saw_sorter.fem_model.PRESSURE_PER_VOLT_ASSUMPTION "
            "for exactly which assumption this back-end would remove."
        )

    def run(self) -> None:
        raise NotImplementedError("Elmer execution is Stage 4; see setup().")

    def fields(self) -> xr.Dataset:
        raise NotImplementedError("Elmer result reading is Stage 4; see setup().")


def surface_displacement_to_bc(
    surface: xr.Dataset, *, frequency: float, component: str = "u_y"
) -> Callable[[np.ndarray], np.ndarray]:
    """Turn a solved substrate surface displacement into a Stage 1 velocity BC.

    Returns a callable ``f(x) -> complex normal velocity``, which is exactly what
    :meth:`SAWFieldModel.boundary_conditions
    <biosim_lab.instruments.saw_sorter.fem_model.SAWFieldModel.boundary_conditions>`
    expects.  With the ``exp(-i omega t)`` convention, ``v = -i * omega * u``.

    Parameters
    ----------
    surface:
        Dataset with coordinate ``x`` and complex displacement components
        (``<component>_real`` / ``<component>_imag``) [m].
    frequency:
        Drive frequency [Hz].
    """
    omega = 2.0 * np.pi * float(frequency)
    xs = surface["x"].values
    u = surface[f"{component}_real"].values + 1j * surface[f"{component}_imag"].values

    def velocity(x: np.ndarray) -> np.ndarray:
        return -1j * omega * (
            np.interp(x[0], xs, u.real) + 1j * np.interp(x[0], xs, u.imag)
        )

    return velocity


__all__ = ["ElmerSolver", "TEMPLATE_DIR", "surface_displacement_to_bc"]
