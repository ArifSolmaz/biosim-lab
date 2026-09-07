"""One palette and one layout for every figure in the platform.

Colour policy
-------------
* **Categorical** hues are assigned in a fixed order and, once a population has a
  slot, it keeps it: colour follows the entity, not its rank in the current
  filter.  :func:`color_for` implements that by hashing known labels to fixed
  slots and only falling back to sequential assignment for unknown ones.
* **Sequential** (magnitude: heatmaps, pressure fields) is a single blue hue,
  light to dark — never a rainbow.
* **Diverging** (polarity: contrast factor, displacement direction) is
  blue <-> red with a neutral grey midpoint.
* Series identity is never carried by colour alone: every multi-series figure
  gets a legend, and the four-or-fewer case is also directly labelled.

The palette below is validated for colour-vision deficiency on both light and
dark surfaces (adjacent-pair separation in OKLab).  Scatter-type figures, where
every pair of series can appear side by side, cap out at the **first three
slots**; beyond that use facets rather than more hues.
"""

from __future__ import annotations

from typing import Any

#: Categorical hues in fixed assignment order: light and dark steps.
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

#: Single-hue sequential ramp (blue), light -> dark, for magnitude encoding.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: Diverging pair with a neutral (not hued) midpoint.
DIVERGING: tuple[str, str, str] = ("#2a78d6", "#f0efec", "#e34948")

PALETTE: dict[str, Any] = {
    "categorical_light": CATEGORICAL_LIGHT,
    "categorical_dark": CATEGORICAL_DARK,
    "sequential": SEQUENTIAL_BLUE,
    "diverging": DIVERGING,
    "surface_light": "#fcfcfb",
    "surface_dark": "#1a1a19",
    "text_primary_light": "#0b0b0b",
    "text_secondary_light": "#52514e",
    "text_primary_dark": "#ffffff",
    "text_secondary_dark": "#c3c2b7",
    "grid_light": "#e6e5e1",
    "grid_dark": "#2f2f2d",
}

#: Stable slot assignment for the cell types the platform ships with, so a
#: figure that drops a population does not repaint the survivors.
_FIXED_SLOTS: dict[str, int] = {
    "mcf7": 0,
    "hela": 6,
    "a549": 2,
    "rbc": 1,
    "wbc": 3,
    "platelet": 4,
    "ps_bead": 5,
    "lipid": 7,
    "target": 0,
    "background": 1,
}

_dynamic: dict[str, int] = {}


def color_for(label: str, *, dark: bool = False) -> str:
    """Return the fixed categorical colour for *label*.

    Known cell types get a permanent slot; anything else is assigned the next
    free slot the first time it is seen and keeps it for the process lifetime.
    """
    ramp = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    key = str(label).lower()
    if key in _FIXED_SLOTS:
        return ramp[_FIXED_SLOTS[key] % len(ramp)]
    if key not in _dynamic:
        used = set(_FIXED_SLOTS.values()) | set(_dynamic.values())
        free = next((i for i in range(len(ramp)) if i not in used), len(_dynamic))
        _dynamic[key] = free
    return ramp[_dynamic[key] % len(ramp)]


def sequential_colorscale() -> list[list[Any]]:
    """Plotly colorscale for magnitude data (single hue, light to dark)."""
    n = len(SEQUENTIAL_BLUE) - 1
    return [[i / n, c] for i, c in enumerate(SEQUENTIAL_BLUE)]


def diverging_colorscale() -> list[list[Any]]:
    """Plotly colorscale for signed data (blue <-> neutral <-> red)."""
    lo, mid, hi = DIVERGING
    return [[0.0, lo], [0.5, mid], [1.0, hi]]


def plotly_layout(
    title: str = "",
    *,
    xaxis_title: str = "",
    yaxis_title: str = "",
    dark: bool = False,
    height: int = 420,
    showlegend: bool = True,
) -> dict[str, Any]:
    """Shared Plotly layout: recessive grid, one axis, legend on by default.

    Never returns a secondary y-axis: two measures of different scale belong in
    two figures, not on two scales in one.
    """
    surface = PALETTE["surface_dark"] if dark else PALETTE["surface_light"]
    text = PALETTE["text_primary_dark"] if dark else PALETTE["text_primary_light"]
    muted = PALETTE["text_secondary_dark"] if dark else PALETTE["text_secondary_light"]
    grid = PALETTE["grid_dark"] if dark else PALETTE["grid_light"]
    axis = {
        "showgrid": True,
        "gridcolor": grid,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": grid,
        "tickfont": {"color": muted, "size": 11},
        "title": {"font": {"color": muted, "size": 12}},
    }
    return {
        "title": {"text": title, "font": {"color": text, "size": 15}},
        "paper_bgcolor": surface,
        "plot_bgcolor": surface,
        "font": {"color": text, "family": "system-ui, -apple-system, sans-serif", "size": 12},
        "xaxis": {**axis, "title": {**axis["title"], "text": xaxis_title}},
        "yaxis": {**axis, "title": {**axis["title"], "text": yaxis_title}},
        "height": height,
        "showlegend": showlegend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": muted, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "margin": {"l": 64, "r": 20, "t": 56 if title else 36, "b": 52},
        "hovermode": "closest",
    }


__all__ = [
    "PALETTE",
    "CATEGORICAL_LIGHT",
    "CATEGORICAL_DARK",
    "SEQUENTIAL_BLUE",
    "DIVERGING",
    "color_for",
    "sequential_colorscale",
    "diverging_colorscale",
    "plotly_layout",
]
