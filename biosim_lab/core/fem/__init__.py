"""Built-in finite-element solvers (scikit-fem).

These are the *always available* back-ends: pure Python + SciPy, no external
binaries.  Heavy alternatives (FEniCSx, OpenFOAM, Elmer) plug in through the
same :class:`~biosim_lab.core.solver.Solver` contract.
"""

from biosim_lab.core.fem.electroquasistatic import ElectroQuasistaticSolver
from biosim_lab.core.fem.helmholtz import HelmholtzSolver
from biosim_lab.core.fem.stokes import StokesSolver

__all__ = ["HelmholtzSolver", "ElectroQuasistaticSolver", "StokesSolver"]
