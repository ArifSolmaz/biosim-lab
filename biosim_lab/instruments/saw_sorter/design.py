"""Design helpers: turn a stated design rule into a drive level, find operating windows.

Papers on acoustic sorters rarely report the acoustic energy density --- it is
hard to measure --- but they often state the *rule* the drive was tuned to
satisfy. Zhang et al. (2023, doi:10.3390/ijms24043338, Sec. 4.2) is explicit:

    "During 1 MHz mode, the displacement of MCF7 cells in the y direction
    should be greater than W/6, while that of PBMCs should be less than W/6"

Because the single-mode trajectory has a closed form (Barnkob et al. 2010,
doi:10.1039/b920376a, eq. 6), that sentence pins ``E_ac`` to an interval:
large enough to move the target ``W/6``, small enough not to move the
background that far. :func:`separation_rule_energy` returns the interval and
its logarithmic midpoint --- the level with equal *factor* margin on both sides,
which is the least arbitrary single number the rule supports. Using it is a
calibration, and every caller in this project labels it as one.

:func:`operating_window` does the complementary job on simulated sweeps: find
the settings where capture is high and contamination low at the same time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.materials import CellType, Fluid
from biosim_lab.instruments.saw_sorter.physics import baw
from biosim_lab.instruments.saw_sorter.physics.acoustics import bruus_phi


def _phi_b(cell: CellType, fluid: Fluid) -> float:
    return float(bruus_phi(cell.rho, fluid.rho, cell.kappa, fluid.kappa))


def _energy_to_move(
    cell: CellType, fluid: Fluid, *, n: int, width: float, duration: float,
    start: float, end: float,
) -> float:
    """``E_ac`` that carries a mean-sized *cell* from *start* to *end* in *duration*.

    Inverts ``tan(k y1) = tan(k y0) exp(t / tau)`` with
    ``tau = 3 mu / (4 Phi_B (k a)^2 E_ac)`` (doi:10.1039/b920376a, eq. 6).
    Positions are measured within one antinode-to-antinode cell.
    """
    k = baw.wavenumber(n, width)
    growth = np.log(np.tan(k * end) / np.tan(k * start))
    if not np.isfinite(growth) or growth <= 0:
        raise ValueError("end must lie between start and the node, on the same side")
    phi = _phi_b(cell, fluid)
    return float(3.0 * fluid.mu * growth / (4.0 * phi * (k * cell.r) ** 2 * duration))


@dataclass(frozen=True)
class RuleCalibration:
    """``E_ac`` interval implied by a displacement rule, and the value chosen from it."""

    target_minimum: float
    background_maximum: float
    chosen: float
    margin_factor: float
    rule: str

    @property
    def feasible(self) -> bool:
        """Whether one field can satisfy the rule for both mean-sized cells."""
        return self.target_minimum < self.background_maximum

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_minimum_J_m3": self.target_minimum,
            "background_maximum_J_m3": self.background_maximum,
            "chosen_J_m3": self.chosen,
            "margin_factor": self.margin_factor,
            "feasible": self.feasible,
            "rule": self.rule,
        }


def separation_rule_energy(
    target: CellType,
    background: CellType,
    fluid: Fluid,
    *,
    width: float,
    duration: float,
    n_low: int = 1,
    n_high: int = 3,
) -> RuleCalibration:
    """The ``E_ac`` window of the low mode that satisfies the two-node separation rule.

    Starting on the high mode's first node (``W/(2 n_high)``), the target must
    travel farther than the node-to-basin-edge distance ``W/(2 n_high)`` in
    *duration*, the background less. For ``n_low = 1, n_high = 3`` that is the
    ``W/6`` rule of doi:10.3390/ijms24043338. Returns the two bounds and their
    geometric mean.
    """
    start = width / (2.0 * n_high)
    end = start + width / (2.0 * n_high)
    e_t = _energy_to_move(target, fluid, n=n_low, width=width, duration=duration,
                          start=start, end=end)
    e_b = _energy_to_move(background, fluid, n=n_low, width=width, duration=duration,
                          start=start, end=end)
    chosen = float(np.sqrt(e_t * e_b))
    return RuleCalibration(
        target_minimum=e_t,
        background_maximum=e_b,
        chosen=chosen,
        margin_factor=float(np.sqrt(e_b / e_t)),
        rule=(
            f"mode {n_low} for {duration:g} s moves a mean {target.key} more than "
            f"W/{2 * n_high} and a mean {background.key} less, starting on the node at "
            f"W/{2 * n_high} (doi:10.3390/ijms24043338, Sec. 4.2)"
        ),
    )


def return_energy(
    cell: CellType,
    fluid: Fluid,
    *,
    width: float,
    duration: float,
    n: int = 3,
    start_fraction: float = 0.9,
    end_fraction: float = 0.1,
) -> float:
    """``E_ac`` of mode *n* that returns a mean-sized *cell* to its node in *duration*.

    The cell starts ``start_fraction`` of the way from the node to the basin
    edge (the antinode) and must end within ``end_fraction`` of that distance
    from the node. It is the quantitative reading of "during 3 MHz mode, the
    cells should be pulled back to the nodes as quickly as possible"
    (doi:10.3390/ijms24043338, Sec. 4.2); the two fractions are choices, and
    callers must report them.
    """
    if not 0.0 < end_fraction < start_fraction < 1.0:
        raise ValueError("need 0 < end_fraction < start_fraction < 1")
    half = width / (2.0 * n)  # antinode (y = 0) to node
    # Measured from the antinode, a cell at distance d from it moves toward the node at `half`.
    return _energy_to_move(
        cell, fluid, n=n, width=width, duration=duration,
        start=(1.0 - start_fraction) * half, end=(1.0 - end_fraction) * half,
    )


def operating_window(
    table: pd.DataFrame,
    parameter: str,
    *,
    capture: str = "capture_efficiency_percent",
    contamination: str = "contamination_rate_percent",
    capture_min: float = 90.0,
    contamination_max: float = 5.0,
) -> dict[str, Any]:
    """Settings of *parameter* at which capture is high AND contamination low.

    Returns the admissible values, their range, and the single best setting
    by Youden's index ``J = capture - contamination`` (Youden 1950,
    doi:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3) --- the
    point that is furthest, in both directions at once, from a device that
    does nothing. An empty window is a finding, not an error: it says no
    setting of this one knob separates the two populations to that standard.
    """
    df = table.sort_values(parameter).reset_index(drop=True)
    ok = (df[capture] >= capture_min) & (df[contamination] <= contamination_max)
    youden = df[capture] - df[contamination]
    best = int(youden.idxmax())
    values = df.loc[ok, parameter].tolist()
    return {
        "parameter": parameter,
        "criteria": {
            "capture_min_percent": capture_min,
            "contamination_max_percent": contamination_max,
        },
        "admissible": values,
        "window": [min(values), max(values)] if values else None,
        "best_setting": float(df.loc[best, parameter]),
        "best_capture_percent": float(df.loc[best, capture]),
        "best_contamination_percent": float(df.loc[best, contamination]),
        "best_youden_percent": float(youden.iloc[best]),
    }


__all__ = [
    "RuleCalibration",
    "separation_rule_energy",
    "return_energy",
    "operating_window",
]
