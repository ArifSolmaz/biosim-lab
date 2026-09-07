"""OpenFOAM back-end for 3-D acoustic streaming and inertial (Re > 1) flow.

Stage 4 status: **interface skeleton and templates only.**  The class below
implements the :class:`~biosim_lab.core.solver.Solver` contract, reports itself
unavailable without an OpenFOAM installation, and ships the case templates; the
numerical bridge is deliberately not written yet.

When this back-end is needed
----------------------------
Stages 1-3 model the *primary* acoustic radiation force in a Stokes flow. That
is the right model for cells larger than the acoustic boundary layer at moderate
drive levels. It is **not** sufficient when:

1. **Acoustic streaming dominates.** The steady second-order flow driven by
   viscous attenuation in the boundary layer scales with ``u_streaming ~
   (3/16) * u_1^2 / c``, independent of particle size, while the radiation force
   scales with ``r^3``. Below a critical radius the streaming drag wins and
   particles are stirred rather than focused. For water at 20 MHz that
   cross-over sits near 1 um, so it matters for platelets and bacteria but not
   for the 9 um tumour cells of the default scenario.

   Reference: Muller, Barnkob, Jensen & Bruus (2012), *A numerical study of
   microparticle acoustophoresis driven by acoustic radiation forces and
   streaming-induced drag forces*, Lab Chip 12:4617, doi:10.1039/c2lc40612h;
   and Muller et al. (2013), Phys. Rev. E 88:023006,
   doi:10.1103/PhysRevE.88.023006.

2. **The channel Reynolds number exceeds ~1**, where the Stokes solver is no
   longer the right model.

3. **Three-dimensional geometry matters** — expansions, bifurcations, trifurcated
   outlets — which the 2-D cross-section model cannot represent.

Interface contract
------------------
* ``setup(mesh, bc)`` writes an OpenFOAM case directory: the Gmsh mesh is
  converted with ``gmshToFoam``, boundary conditions are rendered from the
  templates in ``templates/``, and the acoustic pressure field solved in Stage 1
  is written as a ``volScalarField`` for the force function object to read.
* ``run()`` executes the solver (``DPMFoam`` for particle-laden runs) and
  blocks until the case finishes.
* ``fields()`` reads the results back with ``fluidfoam`` and returns an
  :class:`xarray.Dataset` with ``u_streaming``, ``pressure`` and, when particle
  tracking was enabled, the parcel positions.

The streaming field is handed back to
:mod:`biosim_lab.core.particles` as an additive ``u_streaming`` contribution to
the fluid velocity, so the Lagrangian tracker is unchanged whether or not this
back-end is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr

from biosim_lab.core.solver import BoundaryCondition, Solver

TEMPLATE_DIR = Path(__file__).parent / "templates"


class OpenFOAMSolver(Solver):
    """OpenFOAM bridge for acoustic streaming and inertial flow.

    Availability requires the ``simpleFoam`` and ``gmshToFoam`` executables on
    ``PATH`` plus the ``fluidfoam`` reader.  The Docker image
    ``biosim-lab-openfoam`` provides them; the main image intentionally does
    not.
    """

    name = "openfoam"
    kind = "streaming"
    required_executables = ("simpleFoam", "gmshToFoam")
    required_modules = ("fluidfoam",)

    def __init__(self, case_dir: str | Path = "openfoam_case", **options: Any) -> None:
        super().__init__(**options)
        self.case_dir = Path(case_dir)

    @staticmethod
    def templates() -> dict[str, Path]:
        """Paths to the shipped case templates, for inspection or manual use."""
        return {p.name: p for p in sorted(TEMPLATE_DIR.glob("*"))}

    def setup(
        self, mesh: Any, bc: dict[str, BoundaryCondition] | list[BoundaryCondition]
    ) -> None:
        raise NotImplementedError(
            "OpenFOAM case generation is Stage 4 and not implemented. The interface "
            "contract, the acousticRadiationForce function-object source and the "
            "controlDict/fvSolution templates are in "
            f"{TEMPLATE_DIR}. Stages 1-3 run without this back-end; when it is absent "
            "the sorter reports 'streaming disabled, using the analytic Rayleigh "
            "approximation'."
        )

    def run(self) -> None:
        raise NotImplementedError("OpenFOAM execution is Stage 4; see setup().")

    def fields(self) -> xr.Dataset:
        raise NotImplementedError("OpenFOAM result reading is Stage 4; see setup().")


def streaming_unavailable_message() -> str:
    """The exact warning the sorter emits when this back-end is missing."""
    return (
        "streaming devre dışı, analitik Rayleigh yaklaşımı kullanılıyor / "
        "streaming disabled, using the analytic Rayleigh approximation "
        "(install the optional OpenFOAM back-end for a full second-order solution)"
    )


def rayleigh_streaming_velocity(u1: float, sound_speed: float) -> float:
    """Analytic slip velocity of Rayleigh streaming outside the boundary layer [m/s].

    ``u_slip = -(3/8) * u1^2 / c`` for a standing wave, where ``u1`` is the
    first-order velocity amplitude; the classic result gives a bulk streaming
    magnitude of ``(3/16) * u1^2 / c``.

    Reference: Rayleigh (1884), *On the circulation of air observed in Kundt's
    tubes*, Phil. Trans. R. Soc. 175:1, doi:10.1098/rstl.1884.0002; modern
    treatment in Bruus, *Acoustofluidics 2*, doi:10.1039/c1lc20770a.

    This is the fallback used when the OpenFOAM back-end is absent: it gives the
    right order of magnitude for the streaming drag, but not its spatial
    structure.
    """
    return 3.0 / 16.0 * float(u1) ** 2 / float(sound_speed)


__all__ = [
    "OpenFOAMSolver",
    "TEMPLATE_DIR",
    "streaming_unavailable_message",
    "rayleigh_streaming_velocity",
]
