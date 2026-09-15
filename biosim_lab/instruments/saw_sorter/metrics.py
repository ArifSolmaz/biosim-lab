"""Sorting metrics, with the definitions the reference papers use --- verbatim.

Each function takes the per-cell outlet table the simulation produces (one row
per cell; columns ``label``, ``is_target``, ``outlet`` in ``{"collect",
"waste"}``, ``exited``, ``x_outlet_m``) and returns a number in the units the
paper reports, so a benchmark can put ours and theirs in the same column.

Why several near-identical ratios
---------------------------------
The literature does not agree on a single name, and the denominators differ in
a way that matters when some cells never leave the channel:

* **Capture efficiency** (Zhang et al. 2023, doi:10.3390/ijms24043338, eq. 1)
  ``= CCs_captured / CCs_in`` --- the denominator is every target cell that
  *entered*, so a cell stuck in the device counts as lost.
* **PBMC contamination rate** (ibid., eq. 2) ``= PBMCs_captured / PBMCs_in``
  --- the same construction for the background population.
* **Recovery rate** (Li et al. 2015, doi:10.1073/pnas.1504484112, p. 4973:
  "dividing the number of cells in the collection channel by the number of
  total cells from both outlets") ``= N_collection / (N_collection + N_waste)``
  --- the denominator is cells that *came out*, because that is all a Petri
  dish at each outlet can count.
* **WBC removal rate** (Li et al. 2015, Fig. 3) ``= N_waste / (N_collection +
  N_waste)`` for the background population: the complement of its recovery.
* **Separation distance** ``Delta Y`` (Li et al. 2015, Fig. 2) --- the lateral
  distance between the target and background populations at the outlet.

When every cell exits, capture efficiency equals recovery rate. They are kept
separate so a run in which they differ says so.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def _select(cells: pd.DataFrame, labels: Iterable[str] | None, *, target: bool) -> pd.DataFrame:
    if labels is not None:
        return cells[cells["label"].isin(list(labels))]
    return cells[cells["is_target"] == target]


def capture_efficiency(cells: pd.DataFrame, targets: Iterable[str] | None = None) -> float:
    """``CCs_captured / CCs_in`` [%] (doi:10.3390/ijms24043338, eq. 1)."""
    t = _select(cells, targets, target=True)
    return 100.0 * _ratio(int((t["outlet"] == "collect").sum()), len(t))


def contamination_rate(cells: pd.DataFrame, background: Iterable[str] | None = None) -> float:
    """``PBMCs_captured / PBMCs_in`` [%] (doi:10.3390/ijms24043338, eq. 2).

    Pools every non-target population unless *background* names them.
    """
    b = _select(cells, background, target=False)
    return 100.0 * _ratio(int((b["outlet"] == "collect").sum()), len(b))


def recovery_rate(cells: pd.DataFrame, targets: Iterable[str] | None = None) -> float:
    """``N_collection / (N_collection + N_waste)`` [%] (doi:10.1073/pnas.1504484112, p. 4973).

    Counts only cells that left the device --- the paper counted fluorescent
    cells in the two outlet dishes.
    """
    t = _select(cells, targets, target=True)
    out = t[t["exited"]]
    return 100.0 * _ratio(int((out["outlet"] == "collect").sum()), len(out))


def removal_rate(cells: pd.DataFrame, background: Iterable[str] | None = None) -> float:
    """``N_waste / (N_collection + N_waste)`` of the background [%].

    doi:10.1073/pnas.1504484112, Fig. 3 ("removal rate of WBCs").
    """
    b = _select(cells, background, target=False)
    out = b[b["exited"]]
    return 100.0 * _ratio(int((out["outlet"] == "waste").sum()), len(out))


def separation_distance(
    cells: pd.DataFrame,
    targets: Iterable[str] | None = None,
    background: Iterable[str] | None = None,
) -> float:
    """``Delta Y`` [um]: target mean outlet position minus background mean.

    Signed, so a run in which the background ends up *ahead* of the target
    reads as negative instead of as a successful separation.
    (Li et al. 2015, doi:10.1073/pnas.1504484112, Fig. 2.)
    """
    t = _select(cells, targets, target=True)
    b = _select(cells, background, target=False)
    if t.empty or b.empty:
        return float("nan")
    delta = float(t["x_outlet_m"].mean() - b["x_outlet_m"].mean())
    # Report it pointing from background to target whichever wall the target
    # was pushed toward: the sign that matters is "separated" vs "overtaken".
    direction = np.sign(float(t["x_outlet_m"].mean() - t["x_initial_m"].mean())) \
        if "x_initial_m" in t else 1.0
    return 1e6 * delta * (direction if direction != 0 else 1.0)


def outlet_histogram(
    cells: pd.DataFrame, channel_width: float, bins: int = 40
) -> pd.DataFrame:
    """Counts of outlet positions per population, one row per bin [um]."""
    edges = np.linspace(0.0, channel_width, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:]) * 1e6
    out = {"x_um": centres}
    for label, group in cells.groupby("label"):
        out[str(label)], _ = np.histogram(group["x_outlet_m"], bins=edges)
    return pd.DataFrame(out)


def paper_metrics(cells: pd.DataFrame) -> dict[str, float]:
    """All of the above under their paper names, for the metrics dictionary."""
    return {
        "capture_efficiency_percent": capture_efficiency(cells),
        "contamination_rate_percent": contamination_rate(cells),
        "recovery_rate_percent": recovery_rate(cells),
        "background_removal_percent": removal_rate(cells),
        "separation_distance_um": separation_distance(cells),
    }


__all__ = [
    "capture_efficiency",
    "contamination_rate",
    "recovery_rate",
    "removal_rate",
    "separation_distance",
    "outlet_histogram",
    "paper_metrics",
]
