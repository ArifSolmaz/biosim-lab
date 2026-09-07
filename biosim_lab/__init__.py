"""biosim-lab — open-source virtual laboratory instrument platform.

A plugin architecture for simulating what commercial bio-instruments do
(acoustic cell sorters, impedance analysers, cell counters, live-cell trackers)
using only free and open-source Python tooling.
"""

__version__ = "0.1.0"

from biosim_lab.core.units import Q_, ureg  # noqa: F401

__all__ = ["__version__", "ureg", "Q_"]
