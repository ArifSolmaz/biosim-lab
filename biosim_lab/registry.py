"""Instrument lookup that also works from an uninstalled source checkout.

:func:`~biosim_lab.core.plugin.discover_instruments` deliberately knows nothing
about specific devices: it reads the ``biosim_lab.instruments`` entry-point
group, which only exists once the package has been ``pip install``-ed. That is
the right behaviour for the core and it is what makes third-party plugins
possible.

It is the wrong behaviour for someone who has just cloned the repository, and
for hosted environments (Streamlit Community Cloud, a notebook service) that run
straight from a checkout without installing anything. There, entry-point
metadata is absent and the platform would report zero instruments.

This module sits at the *package* level rather than in ``core/``, so the layering
rule in ARCHITECTURE.md §2 still holds: the core imports no instrument. The
package is allowed to know which devices ship inside it.
"""

from __future__ import annotations

import warnings

from biosim_lab.core.plugin import (
    Instrument,
    MissingBackendWarning,
    discover_instruments,
)

#: The devices that ship in this repository, as dotted paths. Kept as strings so
#: one broken instrument cannot stop the others from loading.
BUILTIN_INSTRUMENTS: tuple[str, ...] = (
    "biosim_lab.instruments.saw_sorter:SAWSorter",
    "biosim_lab.instruments.impedance_rtca:ImpedanceRTCA",
    "biosim_lab.instruments.cell_counter:CellCounter",
    "biosim_lab.instruments.cell_tracker:CellTracker",
)


def _load(path: str) -> type[Instrument] | None:
    import importlib

    module_name, _, attr = path.partition(":")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except Exception as exc:  # noqa: BLE001 - one broken device must not kill the rest
        warnings.warn(
            f"built-in instrument '{path}' failed to import: {exc}",
            MissingBackendWarning,
            stacklevel=2,
        )
        return None


def installed_instruments(*, quiet: bool = True) -> dict[str, type[Instrument]]:
    """Return ``{name: Instrument subclass}``, entry points first.

    Any instrument registered through the ``biosim_lab.instruments`` entry-point
    group wins --- that is how a third-party package overrides or extends the
    set. The built-ins are only imported directly to fill gaps, which in
    practice means "the package was never installed".
    """
    found = dict(discover_instruments(quiet=quiet))
    for path in BUILTIN_INSTRUMENTS:
        cls = _load(path)
        if cls is not None and cls.name not in found:
            found[cls.name] = cls
    return found


def discovery_source() -> str:
    """Describe how the instruments were found, for diagnostics."""
    via_entry_points = len(discover_instruments(quiet=True))
    total = len(installed_instruments())
    if via_entry_points == total and via_entry_points > 0:
        return f"entry points ({total} instrument(s))"
    if via_entry_points == 0:
        return (
            f"direct import ({total} built-in instrument(s)); the package is not "
            "pip-installed, so third-party plugins will not be discovered"
        )
    return (
        f"{via_entry_points} via entry points plus "
        f"{total - via_entry_points} built-in(s) imported directly"
    )


__all__ = ["BUILTIN_INSTRUMENTS", "installed_instruments", "discovery_source"]
