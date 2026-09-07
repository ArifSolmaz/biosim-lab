"""Single process-wide :mod:`pint` unit registry.

Unit policy (see ARCHITECTURE.md section 8)
-------------------------------------------
* Units live at the *boundary*: configuration files, public constructors and
  reported results.
* The numerical core (force kernels, ODE right-hand sides, FEM assembly) works
  on plain ``float`` / ``numpy.ndarray`` in **SI base units**, because wrapping
  every array in a ``pint.Quantity`` inside ``scipy.integrate.solve_ivp`` costs
  more than the physics.

Never instantiate a second ``UnitRegistry``: quantities created by different
registries cannot be combined and ``pint`` raises at the worst possible moment.
"""

from __future__ import annotations

from typing import Any

import pint

#: The one and only registry. Import this, do not create your own.
ureg = pint.UnitRegistry()
ureg.default_format = "~P"

#: Shorthand quantity constructor, e.g. ``Q_(20, "MHz")`` or ``Q_("20 MHz")``.
Q_ = ureg.Quantity

# Convenience aliases used across the code base.
ureg.define("vpp = volt = Vpp")  # peak-to-peak volts, dimensionally volts


def to_si(value: Any, expected: str) -> float:
    """Convert *value* to a bare SI float, checking dimensionality.

    Parameters
    ----------
    value:
        A ``pint.Quantity``, a parseable string such as ``"20 MHz"``, or a bare
        number **already assumed to be in SI base units**.
    expected:
        Unit string describing the expected dimensionality, e.g. ``"Hz"``,
        ``"m"``, ``"kg/m**3"``.

    Returns
    -------
    float
        Magnitude in SI base units.

    Raises
    ------
    pint.DimensionalityError
        If *value* carries units incompatible with *expected*.
    """
    target = ureg.Unit(expected)
    if isinstance(value, str):
        value = ureg.Quantity(value)
    if isinstance(value, ureg.Quantity):
        if value.dimensionality != target.dimensionality:
            raise pint.DimensionalityError(
                value.units, target, extra_msg=f" (expected something like {expected})"
            )
        return float(value.to_base_units().magnitude)
    return float(value)


def si_unit_of(expected: str) -> str:
    """Return the SI base-unit string for *expected* (for xarray attributes)."""
    return f"{ureg.Unit(expected).to_base_units()!s}" if hasattr(
        ureg.Unit(expected), "to_base_units"
    ) else str(ureg.Unit(expected))


__all__ = ["ureg", "Q_", "to_si", "si_unit_of"]
