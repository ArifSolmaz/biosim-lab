"""Plugin contract and discovery.

Every virtual instrument in biosim-lab is a subclass of :class:`Instrument`
advertised through the ``biosim_lab.instruments`` entry-point group.  The core
never imports an instrument module directly, so a third-party package can add a
new device without touching this repository (see CONTRIBUTING.md).

Heavy or platform-specific dependencies (Napari/Qt, OpenFOAM, Elmer, FEniCSx)
must be pulled in with :func:`optional_import` so that an absent back-end
degrades into a *missing capability* rather than an ``ImportError`` at package
import time.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover - typing only
    from biosim_lab.core.config import ExperimentConfig

INSTRUMENT_GROUP = "biosim_lab.instruments"
SOLVER_GROUP = "biosim_lab.solvers"


class MissingBackendWarning(UserWarning):
    """Raised when an optional back-end is absent and a fallback is used."""


class RegimeWarning(UserWarning):
    """Raised when a simulation runs outside the validity of its model."""


def optional_import(module: str, *, purpose: str = "", warn: bool = False) -> ModuleType | None:
    """Import *module*, returning ``None`` instead of raising when absent.

    Parameters
    ----------
    module:
        Dotted module path, e.g. ``"napari"``.
    purpose:
        Human-readable description used in the warning message.
    warn:
        Emit a :class:`MissingBackendWarning` when the import fails.
    """
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - a broken optional dep must not kill core
        if warn:
            warnings.warn(
                f"optional back-end '{module}' unavailable ({exc.__class__.__name__}: {exc})."
                + (f" {purpose}" if purpose else ""),
                MissingBackendWarning,
                stacklevel=2,
            )
        return None


@dataclass
class InstrumentResult:
    """Uniform result container returned by every instrument.

    Attributes
    ----------
    fields:
        Gridded / trajectory data as an :class:`xarray.Dataset` in SI units.
        Every variable carries a ``units`` attribute.
    metrics:
        Scalar summary values (efficiency, purity, IC50, cell count ...).
        Serialised into the NetCDF ``attrs`` as JSON on save.
    table:
        Optional per-object table (one row per particle / cell / well).
    meta:
        Provenance: instrument name, version, config hash, assumptions.
    """

    fields: xr.Dataset = field(default_factory=xr.Dataset)
    metrics: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dataset(self) -> xr.Dataset:
        """Fold metrics and metadata into a single serialisable Dataset."""
        ds = self.fields.copy()
        ds.attrs["biosim_metrics"] = json.dumps(self.metrics, default=str)
        ds.attrs["biosim_meta"] = json.dumps(self.meta, default=str)
        return ds

    @classmethod
    def from_dataset(
        cls, ds: xr.Dataset, table: pd.DataFrame | None = None
    ) -> InstrumentResult:
        """Inverse of :meth:`to_dataset`."""
        metrics = json.loads(ds.attrs.get("biosim_metrics", "{}"))
        meta = json.loads(ds.attrs.get("biosim_meta", "{}"))
        fields = ds.drop_attrs() if hasattr(ds, "drop_attrs") else ds
        return cls(fields=fields, metrics=metrics, table=table, meta=meta)


class Instrument(ABC):
    """Abstract base class for every virtual instrument.

    Lifecycle
    ---------
    ``setup()`` → ``run()`` → ``results()`` → ``dashboard()``

    ``setup()`` must be idempotent: calling it twice does no extra work.
    ``run()`` performs the expensive computation and stores the result.
    """

    #: entry-point name, must match the key in ``pyproject.toml``
    name: ClassVar[str] = "abstract"
    display_name: ClassVar[str] = "Abstract instrument"
    description: ClassVar[str] = ""
    #: pydantic model describing the instrument-specific config block
    ConfigModel: ClassVar[type | None] = None

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._result: InstrumentResult | None = None
        self._is_set_up = False

    # -- contract ---------------------------------------------------------
    @abstractmethod
    def setup(self) -> None:
        """Prepare meshes, material properties and fields. Idempotent."""

    @abstractmethod
    def run(self) -> InstrumentResult:
        """Execute the simulation / analysis and return the result."""

    def results(self) -> InstrumentResult:
        """Return the result of the last :meth:`run`."""
        if self._result is None:
            raise RuntimeError(f"{self.name}: run() has not been called yet")
        return self._result

    def dashboard(self) -> Any:
        """Return a Panel ``Viewable`` for interactive exploration.

        The default implementation renders the metrics table only; instruments
        are expected to override it.
        """
        pn = optional_import("panel", purpose="dashboards require Panel", warn=True)
        if pn is None:  # pragma: no cover - panel is a core dependency
            raise RuntimeError("panel is required for dashboards")
        pn.extension("plotly")
        res = self.results()
        return pn.Column(
            f"## {self.display_name}",
            pn.pane.DataFrame(pd.DataFrame([res.metrics]).T.rename(columns={0: "value"})),
        )

    @classmethod
    def example_config(cls) -> dict[str, Any]:
        """Return a minimal working configuration dictionary."""
        return {"instrument": cls.name, "params": {}}

    # -- helpers ----------------------------------------------------------
    def _ensure_setup(self) -> None:
        if not self._is_set_up:
            self.setup()
            self._is_set_up = True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} ran={self._result is not None}>"


# -- discovery ------------------------------------------------------------
def _entry_points(group: str) -> list[importlib_metadata.EntryPoint]:
    try:
        return list(importlib_metadata.entry_points(group=group))
    except Exception:  # noqa: BLE001 - broken metadata must not kill the CLI
        return []


def discover_instruments(*, quiet: bool = False) -> dict[str, type[Instrument]]:
    """Return ``{name: Instrument subclass}`` for every installed instrument.

    A plugin that fails to import is skipped with a warning; one broken device
    must never prevent the rest of the platform from starting.
    """
    found: dict[str, type[Instrument]] = {}
    for ep in _entry_points(INSTRUMENT_GROUP):
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                warnings.warn(
                    f"instrument plugin '{ep.name}' failed to load: {exc}",
                    MissingBackendWarning,
                    stacklevel=2,
                )
            continue
        if isinstance(obj, type) and issubclass(obj, Instrument):
            found[ep.name] = obj
        elif not quiet:
            warnings.warn(
                f"entry point '{ep.name}' is not an Instrument subclass", MissingBackendWarning,
                stacklevel=2,
            )
    return found


def get_instrument(name: str) -> type[Instrument]:
    """Look up a single instrument class by entry-point name."""
    instruments = discover_instruments()
    if name not in instruments:
        available = ", ".join(sorted(instruments)) or "none"
        raise KeyError(f"unknown instrument {name!r}; installed: {available}")
    return instruments[name]


__all__ = [
    "Instrument",
    "InstrumentResult",
    "MissingBackendWarning",
    "RegimeWarning",
    "discover_instruments",
    "get_instrument",
    "optional_import",
    "INSTRUMENT_GROUP",
    "SOLVER_GROUP",
]
