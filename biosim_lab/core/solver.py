"""Solver contract shared by built-in and external numerical back-ends.

A :class:`Solver` maps a mesh plus boundary conditions onto an
:class:`xarray.Dataset` of SI-unit fields.  The same contract is honoured by

* the built-in ``scikit-fem`` solvers (always available), and
* optional heavy back-ends (OpenFOAM, Elmer) that shell out to external
  binaries and are shipped as separate plugins.

Callers must treat an unavailable solver as a *degraded capability*: query
:meth:`Solver.is_available` first and fall back to an analytic approximation
with a visible warning, never a hard failure.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import xarray as xr

from biosim_lab.core.plugin import SOLVER_GROUP, MissingBackendWarning

SolverKind = Literal["acoustics", "electro", "flow", "piezo", "streaming"]


@dataclass
class BoundaryCondition:
    """One boundary condition attached to a named mesh facet group.

    Attributes
    ----------
    where:
        Physical-group name from the mesh (``"inlet"``, ``"saw_surface"`` ...).
    kind:
        ``"dirichlet"``, ``"neumann"``, ``"robin"`` or ``"impedance"``.
    value:
        Scalar, complex, array or callable ``f(x) -> value`` in SI units.
    meta:
        Free-form extras (e.g. wave number for a leaky-wave condition).
    """

    where: str
    kind: str
    value: Any
    meta: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    """Abstract numerical back-end."""

    name: ClassVar[str] = "abstract"
    kind: ClassVar[SolverKind] = "acoustics"
    #: external executables that must be on PATH for this solver to work
    required_executables: ClassVar[tuple[str, ...]] = ()
    #: python modules that must be importable
    required_modules: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._fields: xr.Dataset | None = None

    # -- availability -----------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(available, reason)``.

        The default implementation checks :attr:`required_executables` on
        ``PATH`` and :attr:`required_modules` importability.
        """
        missing_exe = [e for e in cls.required_executables if shutil.which(e) is None]
        missing_mod = []
        for mod in cls.required_modules:
            try:
                __import__(mod)
            except Exception:  # noqa: BLE001
                missing_mod.append(mod)
        if missing_exe or missing_mod:
            parts = []
            if missing_exe:
                parts.append("executables not on PATH: " + ", ".join(missing_exe))
            if missing_mod:
                parts.append("modules not importable: " + ", ".join(missing_mod))
            return False, "; ".join(parts)
        return True, "available"

    # -- contract ---------------------------------------------------------
    @abstractmethod
    def setup(
        self, mesh: Any, bc: dict[str, BoundaryCondition] | list[BoundaryCondition]
    ) -> None:
        """Assemble the discrete problem for *mesh* under boundary conditions *bc*."""

    @abstractmethod
    def run(self) -> None:
        """Solve the assembled problem and cache the fields."""

    @abstractmethod
    def fields(self) -> xr.Dataset:
        """Return the solution fields in SI units."""

    # -- helpers ----------------------------------------------------------
    def solve(self, mesh: Any, bc: Any) -> xr.Dataset:
        """Convenience: ``setup`` + ``run`` + ``fields`` in one call."""
        self.setup(mesh, bc)
        self.run()
        return self.fields()

    def __repr__(self) -> str:  # pragma: no cover
        ok, why = self.is_available()
        return f"<{type(self).__name__} kind={self.kind} available={ok} ({why})>"


class UnavailableSolver(Solver):
    """Placeholder returned when a requested back-end is not installed.

    Calling it raises, but merely *constructing* it does not — which lets
    ``biosim doctor`` list a back-end and explain why it is unusable.
    """

    def __init__(self, name: str, reason: str, kind: SolverKind = "acoustics") -> None:
        super().__init__()
        self.name = name  # type: ignore[misc]
        self.kind = kind  # type: ignore[misc]
        self.reason = reason

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return False, "placeholder"

    def _fail(self) -> None:
        raise RuntimeError(
            f"solver back-end {self.name!r} is not available: {self.reason}. "
            "Install the optional plugin or use the built-in analytic/FEM path."
        )

    def setup(self, mesh: Any, bc: Any) -> None:
        self._fail()

    def run(self) -> None:
        self._fail()

    def fields(self) -> xr.Dataset:
        self._fail()
        raise AssertionError("unreachable")


def discover_solvers(*, quiet: bool = True) -> dict[str, type[Solver]]:
    """Return ``{name: Solver subclass}`` for every installed solver plugin."""
    found: dict[str, type[Solver]] = {}
    try:
        eps = list(importlib_metadata.entry_points(group=SOLVER_GROUP))
    except Exception:  # noqa: BLE001
        eps = []
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                warnings.warn(
                    f"solver plugin '{ep.name}' failed to load: {exc}",
                    MissingBackendWarning,
                    stacklevel=2,
                )
            continue
        if isinstance(obj, type) and issubclass(obj, Solver):
            found[ep.name] = obj
    return found


def get_solver(name: str, **options: Any) -> Solver:
    """Instantiate solver *name*, or an :class:`UnavailableSolver` explaining why not."""
    solvers = discover_solvers()
    cls = solvers.get(name)
    if cls is None:
        return UnavailableSolver(name, "no entry point registered under "
                                f"'{SOLVER_GROUP}' with this name")
    ok, why = cls.is_available()
    if not ok:
        return UnavailableSolver(name, why, kind=cls.kind)
    return cls(**options)


def solver_report() -> list[dict[str, Any]]:
    """Rows for ``biosim doctor``: one dict per registered solver back-end."""
    rows: list[dict[str, Any]] = []
    for name, cls in sorted(discover_solvers().items()):
        ok, why = cls.is_available()
        rows.append({"name": name, "kind": cls.kind, "available": ok, "detail": why})
    return rows


__all__ = [
    "Solver",
    "SolverKind",
    "BoundaryCondition",
    "UnavailableSolver",
    "discover_solvers",
    "get_solver",
    "solver_report",
]
