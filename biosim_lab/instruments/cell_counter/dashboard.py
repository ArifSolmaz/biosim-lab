"""Panel dashboard for the cell counter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biosim_lab.core.viz.dashboard import data_table, metric_tiles, require_panel, shell
from biosim_lab.core.viz.theme import color_for, plotly_layout

_TILE_FORMATS = {
    "n_total": "{:d}",
    "n_live": "{:d}",
    "n_dead": "{:d}",
    "viability_percent": "{:.1f} %",
    "concentration_per_ml": "{:.3g}",
    "mean_diameter_um": "{:.2f} µm",
    "counting_relative_uncertainty": "{:.1%}",
}


def _image_figure(image: np.ndarray, labels: np.ndarray | None, title: str) -> go.Figure:
    """Grayscale field of view with optional segmentation outlines."""
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(image),
            colorscale="Greys",
            reversescale=True,
            showscale=False,
            hovertemplate="row %{y}<br>col %{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    if labels is not None:
        from skimage.segmentation import find_boundaries

        edges = find_boundaries(labels, mode="outer")
        ys, xs = np.nonzero(edges)
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="markers",
                marker={"size": 1.6, "color": color_for("mcf7")},
                name="segmentation",
                hoverinfo="skip",
            )
        )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(**plotly_layout(title, xaxis_title="x (px)", yaxis_title="y (px)",
                                      height=560, showlegend=False))
    return fig


def _diameter_figure(table: pd.DataFrame) -> go.Figure:
    """Size distribution split by viability status."""
    fig = go.Figure()
    edges = np.linspace(
        float(table["diameter_um"].min()) * 0.9,
        float(table["diameter_um"].max()) * 1.05,
        31,
    )
    for status in ("live", "dead"):
        sel = table[table["status"] == status]
        counts, _ = np.histogram(sel["diameter_um"], bins=edges)
        fig.add_trace(
            go.Bar(
                x=0.5 * (edges[:-1] + edges[1:]),
                y=counts,
                name=status,
                marker={"color": color_for("mcf7" if status == "live" else "rbc"),
                        "line": {"width": 0}},
                opacity=0.85,
                hovertemplate=f"<b>{status}</b><br>d %{{x:.1f}} µm"
                              "<br>%{y} cells<extra></extra>",
            )
        )
    fig.update_layout(barmode="overlay", bargap=0.08)
    fig.update_layout(**plotly_layout("Cell size distribution",
                                      xaxis_title="equivalent diameter (µm)",
                                      yaxis_title="count"))
    return fig


def build_dashboard(instrument: Any) -> Any:
    """Assemble the counter dashboard from a run instrument."""
    pn = require_panel()
    result = instrument.results()
    table = result.table

    tabs = [
        ("Segmentation", pn.pane.Plotly(
            _image_figure(instrument.image, instrument.segmentation.labels,
                          "Field of view with segmentation"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=580)),
        ("Raw image", pn.pane.Plotly(
            _image_figure(instrument.image, None, "Raw field of view"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=580)),
    ]
    if len(table):
        tabs.append(("Size distribution", pn.pane.Plotly(
            _diameter_figure(table),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))
    tabs.append(("Object table", data_table(table, height=380)))

    return shell(
        "Automated cell counter",
        "Watershed segmentation · Neubauer chamber geometry · trypan-blue viability",
        tiles=metric_tiles(
            result.metrics,
            ["n_total", "n_live", "n_dead", "viability_percent", "concentration_per_ml",
             "mean_diameter_um", "counting_relative_uncertainty"],
            _TILE_FORMATS,
        ),
        tabs=tabs,
        footer="Counting uncertainty is Poisson, 1/sqrt(N) · chamber geometry "
        "doi:10.1016/B978-0-12-427150-0.50098-X",
    )


__all__ = ["build_dashboard"]
