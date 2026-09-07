"""Counting, concentration and viability from a segmented field of view.

Hemocytometer geometry
----------------------
A Neubauer improved chamber has a coverslip gap of **0.100 mm** and a ruled grid
whose four corner squares are 1 mm x 1 mm.  One corner square therefore holds
``1 mm * 1 mm * 0.1 mm = 0.1 uL = 1e-4 mL``, which is where the familiar
"multiply by 10^4" rule comes from:

    concentration [cells/mL] = mean_count_per_large_square * 1e4 * dilution

Reference for the chamber geometry and the counting convention (including the
"count two edges, skip the other two" rule that avoids double counting):
Absher, *Hemocytometer counting*, in Kruse & Patterson (eds.), *Tissue Culture:
Methods and Applications* (1973), doi:10.1016/B978-0-12-427150-0.50098-X.

For an arbitrary imaging chamber the same relation is used with the actual
field-of-view area and chamber depth, which is what
:func:`concentration_from_field` does.

Viability
---------
Trypan blue is excluded by intact membranes, so dead cells take up the dye and
appear **dark** in brightfield.  Viability is the fraction of unstained cells.
The dye is itself cytotoxic within minutes, so counts must be made promptly —
that is a protocol constraint, not something the software can correct for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NEUBAUER_DEPTH_M = 100e-6
"""Neubauer improved chamber depth [m] (0.100 mm, coverslip to floor)."""

NEUBAUER_LARGE_SQUARE_M = 1e-3
"""Side of one corner counting square [m]."""


@dataclass
class CountingResult:
    """Counts, concentration, viability and size statistics for one field."""

    n_total: int
    n_live: int
    n_dead: int
    concentration_per_ml: float
    live_concentration_per_ml: float
    viability_percent: float
    volume_ml: float
    diameters_um: np.ndarray
    table: pd.DataFrame

    def as_metrics(self) -> dict[str, Any]:
        """Flat dictionary for :class:`~biosim_lab.core.plugin.InstrumentResult`."""
        d = self.diameters_um
        return {
            "n_total": self.n_total,
            "n_live": self.n_live,
            "n_dead": self.n_dead,
            "viability_percent": self.viability_percent,
            "concentration_per_ml": self.concentration_per_ml,
            "live_concentration_per_ml": self.live_concentration_per_ml,
            "sampled_volume_ml": self.volume_ml,
            "mean_diameter_um": float(np.mean(d)) if d.size else float("nan"),
            "median_diameter_um": float(np.median(d)) if d.size else float("nan"),
            "cv_diameter_percent": float(100 * np.std(d) / np.mean(d))
            if d.size and np.mean(d) > 0
            else float("nan"),
        }


def field_volume_ml(
    shape_px: tuple[int, int], pixel_size: float, depth: float = NEUBAUER_DEPTH_M
) -> float:
    """Volume [mL] of liquid imaged by one field of view.

    ``V = (height_px * pixel_size) * (width_px * pixel_size) * depth``,
    converted from m^3 to mL (1 m^3 = 1e6 mL).
    """
    h, w = shape_px
    area = (h * pixel_size) * (w * pixel_size)
    return float(area * depth * 1e6)


def classify_viability(
    properties: pd.DataFrame,
    *,
    intensity_column: str = "mean_intensity",
    threshold: float | None = None,
) -> pd.Series:
    """Label each object ``live`` or ``dead`` from its trypan-blue uptake.

    Dead (stained) cells are darker than live ones.  With no explicit
    *threshold* the split is placed by Otsu on the intensity distribution, which
    only makes sense when both populations are present — with a unimodal
    distribution the caller should pass an explicit threshold from a control.
    """
    if intensity_column not in properties.columns or len(properties) == 0:
        return pd.Series(["live"] * len(properties), index=properties.index)
    values = properties[intensity_column].to_numpy(dtype=float)
    if threshold is None:
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(values)) if values.size > 1 else float(values.mean())
    return pd.Series(np.where(values < threshold, "dead", "live"), index=properties.index)


def concentration_from_field(
    n_cells: int,
    shape_px: tuple[int, int],
    pixel_size: float,
    *,
    depth: float = NEUBAUER_DEPTH_M,
    dilution_factor: float = 1.0,
) -> tuple[float, float]:
    """Concentration [cells/mL] and the sampled volume [mL] for one field.

    *dilution_factor* is the factor the sample was diluted by before counting
    (2.0 for the usual 1:1 trypan-blue mix).
    """
    volume = field_volume_ml(shape_px, pixel_size, depth)
    if volume <= 0:
        raise ValueError("field volume must be positive")
    return float(n_cells / volume * dilution_factor), volume


def count_field(
    segmentation: Any,
    *,
    shape_px: tuple[int, int],
    pixel_size: float,
    depth: float = NEUBAUER_DEPTH_M,
    dilution_factor: float = 1.0,
    viability_threshold: float | None = None,
    min_diameter_um: float = 0.0,
    max_diameter_um: float = np.inf,
) -> CountingResult:
    """Turn a :class:`SegmentationResult` into counts, concentration and viability.

    Objects outside ``[min_diameter_um, max_diameter_um]`` are treated as debris
    or clumps and excluded — a size gate is what a real counter does and it
    keeps a single merged doublet from being read as one giant cell.
    """
    props: pd.DataFrame = segmentation.properties.copy()
    if len(props) == 0:
        empty = np.array([], dtype=float)
        return CountingResult(0, 0, 0, 0.0, 0.0, float("nan"),
                              field_volume_ml(shape_px, pixel_size, depth), empty, props)

    props["diameter_um"] = props["equivalent_diameter"] * pixel_size * 1e6
    keep = (props["diameter_um"] >= min_diameter_um) & (
        props["diameter_um"] <= max_diameter_um
    )
    props = props.loc[keep].reset_index(drop=True)
    props["status"] = classify_viability(props, threshold=viability_threshold)

    n_total = int(len(props))
    n_dead = int((props["status"] == "dead").sum())
    n_live = n_total - n_dead

    concentration, volume = concentration_from_field(
        n_total, shape_px, pixel_size, depth=depth, dilution_factor=dilution_factor
    )
    live_conc, _ = concentration_from_field(
        n_live, shape_px, pixel_size, depth=depth, dilution_factor=dilution_factor
    )
    viability = 100.0 * n_live / n_total if n_total else float("nan")

    return CountingResult(
        n_total=n_total,
        n_live=n_live,
        n_dead=n_dead,
        concentration_per_ml=concentration,
        live_concentration_per_ml=live_conc,
        viability_percent=viability,
        volume_ml=volume,
        diameters_um=props["diameter_um"].to_numpy(),
        table=props,
    )


def counting_uncertainty(n: int) -> float:
    """Relative counting uncertainty from Poisson statistics, ``1/sqrt(n)``.

    Worth reporting: counting 100 cells carries a 10 % standard error, which is
    usually larger than any difference the user is trying to detect.
    """
    return float("inf") if n <= 0 else float(1.0 / np.sqrt(n))


__all__ = [
    "NEUBAUER_DEPTH_M",
    "NEUBAUER_LARGE_SQUARE_M",
    "CountingResult",
    "field_volume_ml",
    "classify_viability",
    "concentration_from_field",
    "count_field",
    "counting_uncertainty",
]
