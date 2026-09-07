"""Plotly figures shared by every instrument.

Design rules enforced here (see :mod:`biosim_lab.core.viz.theme`):

* one y-axis per figure — two measures of different scale get two figures;
* categorical colour follows the entity, never its rank;
* a legend whenever more than one series is drawn, plus direct labels when
  there are four or fewer;
* magnitude uses a single-hue sequential ramp, signed quantities a diverging
  one with a neutral midpoint;
* every figure carries a hover layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

from biosim_lab.core.viz.theme import (
    color_for,
    diverging_colorscale,
    plotly_layout,
    sequential_colorscale,
)

UM = 1e6  # metres -> micrometres


def _finish(fig: go.Figure, **layout: Any) -> go.Figure:
    fig.update_layout(**plotly_layout(**layout))
    return fig


def trajectory_figure(
    trajectories: xr.Dataset,
    *,
    node_positions: Sequence[float] = (),
    channel_width: float | None = None,
    channel_length: float | None = None,
    max_tracks_per_population: int = 40,
    dark: bool = False,
    title: str = "Cell trajectories",
) -> go.Figure:
    """Lateral position vs axial distance, one colour per population.

    Parameters
    ----------
    trajectories:
        Dataset from :class:`~biosim_lab.core.particles.LagrangianTracker` with
        ``position(particle, time, axis)``.
    node_positions:
        Pressure-node positions [m] drawn as reference lines.
    channel_length:
        Axial extent of the device [m]. Integration runs past the outlet so the
        slowest wall-hugging cell also exits, so tracks are trimmed here — the
        plot should show the device, not the integration window.
    max_tracks_per_population:
        Trajectories are thinned so the figure stays readable; the legend
        reports how many were drawn.
    """
    pos = trajectories["position"].values
    labels = np.asarray(trajectories["label"].values, dtype=str)
    fig = go.Figure()

    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        shown = idx[:: max(1, len(idx) // max_tracks_per_population)]
        colour = color_for(label, dark=dark)
        for j, i in enumerate(shown):
            z = pos[i, :, 2]
            x = pos[i, :, 0]
            if channel_length is not None:
                keep = z <= channel_length
                # Keep the first point beyond the outlet so the line reaches it.
                if (~keep).any() and keep.any():
                    keep[np.argmax(~keep)] = True
                z, x = z[keep], x[keep]
            fig.add_trace(
                go.Scatter(
                    x=z * 1e3,
                    y=x * UM,
                    mode="lines",
                    line={"color": colour, "width": 1.2},
                    opacity=0.55,
                    name=f"{label} (n={len(idx)})",
                    legendgroup=label,
                    showlegend=j == 0,
                    hovertemplate=(f"<b>{label}</b><br>z %{{x:.2f}} mm"
                                   "<br>x %{y:.1f} µm<extra></extra>"),
                )
            )

    for node in node_positions:
        fig.add_hline(
            y=node * UM,
            line={"color": "#52514e" if not dark else "#c3c2b7", "width": 1, "dash": "dot"},
            annotation_text="pressure node",
            annotation_position="top right",
            annotation_font_size=10,
        )
    if channel_width is not None:
        fig.update_yaxes(range=[0, channel_width * UM])
    if channel_length is not None:
        fig.update_xaxes(range=[0, channel_length * 1e3])

    return _finish(
        fig,
        title=title,
        xaxis_title="axial position z (mm)",
        yaxis_title="lateral position x (µm)",
        dark=dark,
    )


def outlet_histogram_figure(
    cells: pd.DataFrame,
    *,
    channel_width: float,
    collection_bounds: tuple[float, float] | None = None,
    bins: int = 40,
    dark: bool = False,
    title: str = "Outlet position distribution",
) -> go.Figure:
    """Histogram of exit positions, one trace per population."""
    edges = np.linspace(0.0, channel_width, bins + 1)
    fig = go.Figure()
    for label, group in cells.groupby("label"):
        counts, _ = np.histogram(group["x_outlet_m"], bins=edges)
        fig.add_trace(
            go.Bar(
                x=0.5 * (edges[:-1] + edges[1:]) * UM,
                y=counts,
                name=str(label),
                marker={"color": color_for(str(label), dark=dark),
                        "line": {"width": 0}},
                opacity=0.85,
                hovertemplate=(f"<b>{label}</b><br>x %{{x:.0f}} µm"
                               "<br>%{y} cells<extra></extra>"),
            )
        )
    if collection_bounds is not None:
        lo, hi = collection_bounds
        fig.add_vrect(
            x0=lo * UM,
            x1=hi * UM,
            fillcolor="#2a78d6",
            opacity=0.08,
            line_width=0,
            annotation_text="collection outlet",
            annotation_position="top left",
            annotation_font_size=10,
        )
    fig.update_layout(barmode="overlay", bargap=0.08)
    return _finish(
        fig,
        title=title,
        xaxis_title="outlet position x (µm)",
        yaxis_title="cell count",
        dark=dark,
    )


def force_profile_figure(
    x: np.ndarray,
    forces: dict[str, np.ndarray],
    *,
    node_positions: Sequence[float] = (),
    dark: bool = False,
    title: str = "Acoustic radiation force across the channel",
) -> go.Figure:
    """Signed force profiles ``F_x(x)`` in piconewtons, one line per population."""
    fig = go.Figure()
    for label, force in forces.items():
        fig.add_trace(
            go.Scatter(
                x=np.asarray(x) * UM,
                y=np.asarray(force) * 1e12,
                mode="lines",
                name=str(label),
                line={"color": color_for(str(label), dark=dark), "width": 2},
                hovertemplate=(f"<b>{label}</b><br>x %{{x:.0f}} µm"
                               "<br>F %{y:.2f} pN<extra></extra>"),
            )
        )
    for node in node_positions:
        fig.add_vline(
            x=node * UM,
            line={"color": "#52514e" if not dark else "#c3c2b7", "width": 1, "dash": "dot"},
        )
    fig.add_hline(y=0, line={"color": "#52514e" if not dark else "#c3c2b7", "width": 1})
    return _finish(
        fig,
        title=title,
        xaxis_title="lateral position x (µm)",
        yaxis_title="F_x (pN)",
        dark=dark,
    )


def field_heatmap_figure(
    grid: xr.Dataset,
    variable: str = "p_abs",
    *,
    diverging: bool = False,
    dark: bool = False,
    title: str | None = None,
    colorbar_title: str = "",
) -> go.Figure:
    """Heatmap of a gridded 2-D field (pressure magnitude, Gor'kov potential, ...).

    Magnitude fields use the single-hue sequential ramp; pass
    ``diverging=True`` for signed fields so zero maps to the neutral midpoint.
    """
    if variable not in grid:
        if {"p_real", "p_imag"} <= set(grid.data_vars) and variable == "p_abs":
            values = np.hypot(grid["p_real"].values, grid["p_imag"].values)
        else:
            raise KeyError(f"{variable!r} not in dataset (have {sorted(grid.data_vars)})")
    else:
        values = grid[variable].values

    kwargs: dict[str, Any] = {}
    if diverging:
        limit = float(np.nanmax(np.abs(values)))
        kwargs = {"colorscale": diverging_colorscale(), "zmid": 0.0,
                  "zmin": -limit, "zmax": limit}
    else:
        kwargs = {"colorscale": sequential_colorscale()}

    fig = go.Figure(
        go.Heatmap(
            x=grid["x"].values * UM,
            y=grid["y"].values * UM,
            z=values.T,
            colorbar={"title": {"text": colorbar_title, "side": "right"}, "thickness": 12},
            hovertemplate="x %{x:.0f} µm<br>y %{y:.1f} µm<br>%{z:.3g}<extra></extra>",
            **kwargs,
        )
    )
    fig.update_yaxes(scaleanchor=None)
    return _finish(
        fig,
        title=title or variable,
        xaxis_title="x (µm)",
        yaxis_title="y (µm)",
        dark=dark,
        showlegend=False,
    )


def sweep_heatmap_figure(
    sweep: xr.Dataset,
    metric: str = "efficiency_percent",
    *,
    x: str | None = None,
    y: str | None = None,
    scale: dict[str, float] | None = None,
    unit_labels: dict[str, str] | None = None,
    dark: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Heatmap of one sweep metric over two swept parameters."""
    dims = list(sweep[metric].dims)
    if len(dims) < 2:
        raise ValueError(f"{metric!r} has dims {dims}; a heatmap needs two swept parameters")
    x = x or dims[0]
    y = y or dims[1]
    scale = scale or {}
    unit_labels = unit_labels or {}
    data = sweep[metric].transpose(y, x).values

    fig = go.Figure(
        go.Heatmap(
            x=sweep[x].values * scale.get(x, 1.0),
            y=sweep[y].values * scale.get(y, 1.0),
            z=data,
            colorscale=sequential_colorscale(),
            colorbar={"title": {"text": metric.replace("_", " "), "side": "right"},
                      "thickness": 12},
            hovertemplate=(f"{x} %{{x:.4g}}<br>{y} %{{y:.4g}}<br>"
                           f"{metric} %{{z:.1f}}<extra></extra>"),
        )
    )
    return _finish(
        fig,
        title=title or metric.replace("_", " "),
        xaxis_title=unit_labels.get(x, x),
        yaxis_title=unit_labels.get(y, y),
        dark=dark,
        showlegend=False,
    )


def size_distribution_figure(
    cells: pd.DataFrame, *, bins: int = 40, dark: bool = False,
    title: str = "Cell size distribution",
) -> go.Figure:
    """Radius histogram per population — the log-normal sampling, made visible."""
    fig = go.Figure()
    radii = cells["radius_m"].to_numpy() * UM
    edges = np.linspace(radii.min() * 0.9, radii.max() * 1.05, bins + 1)
    for label, group in cells.groupby("label"):
        counts, _ = np.histogram(group["radius_m"] * UM, bins=edges)
        fig.add_trace(
            go.Bar(
                x=0.5 * (edges[:-1] + edges[1:]),
                y=counts,
                name=str(label),
                marker={"color": color_for(str(label), dark=dark), "line": {"width": 0}},
                opacity=0.85,
                hovertemplate=(f"<b>{label}</b><br>r %{{x:.2f}} µm"
                               "<br>%{y} cells<extra></extra>"),
            )
        )
    fig.update_layout(barmode="overlay", bargap=0.08)
    return _finish(fig, title=title, xaxis_title="radius (µm)", yaxis_title="cell count",
                   dark=dark)


def timeseries_figure(
    df: pd.DataFrame,
    *,
    x: str = "time",
    series: Sequence[str] | None = None,
    time_scale: float = 1.0 / 3600.0,
    dark: bool = False,
    title: str = "",
    xaxis_title: str = "time (h)",
    yaxis_title: str = "",
) -> go.Figure:
    """Generic multi-series time course (Cell Index, growth curves, ...)."""
    series = list(series) if series is not None else [c for c in df.columns if c != x]
    fig = go.Figure()
    for name in series:
        fig.add_trace(
            go.Scatter(
                x=df[x] * time_scale,
                y=df[name],
                mode="lines",
                name=str(name),
                line={"color": color_for(str(name), dark=dark), "width": 2},
                hovertemplate=(f"<b>{name}</b><br>%{{x:.2f}}<br>%{{y:.3f}}<extra></extra>"),
            )
        )
    fig.update_layout(hovermode="x unified")
    return _finish(fig, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                   dark=dark, showlegend=len(series) > 1)


def nyquist_figure(
    frequency: np.ndarray, impedance: np.ndarray, *, dark: bool = False,
    title: str = "Nyquist plot",
) -> go.Figure:
    """Complex-plane impedance locus, ``-Im(Z)`` against ``Re(Z)``."""
    z = np.asarray(impedance)
    fig = go.Figure(
        go.Scatter(
            x=z.real,
            y=-z.imag,
            mode="lines+markers",
            marker={"size": 8, "color": color_for("target", dark=dark),
                    "line": {"width": 2, "color": "#fcfcfb" if not dark else "#1a1a19"}},
            line={"color": color_for("target", dark=dark), "width": 2},
            customdata=np.asarray(frequency),
            hovertemplate=("f %{customdata:.3g} Hz<br>Re Z %{x:.4g} Ω"
                           "<br>-Im Z %{y:.4g} Ω<extra></extra>"),
            name="Z(f)",
        )
    )
    return _finish(fig, title=title, xaxis_title="Re Z (Ω)", yaxis_title="−Im Z (Ω)",
                   dark=dark, showlegend=False)


def bode_magnitude_figure(
    frequency: np.ndarray, impedance: np.ndarray, *, dark: bool = False,
    title: str = "Impedance magnitude",
) -> go.Figure:
    """``|Z|`` against frequency on log-log axes.

    Magnitude and phase are deliberately **two figures**, not one plot with two
    y-scales.
    """
    fig = go.Figure(
        go.Scatter(
            x=np.asarray(frequency),
            y=np.abs(np.asarray(impedance)),
            mode="lines",
            line={"color": color_for("target", dark=dark), "width": 2},
            hovertemplate="f %{x:.4g} Hz<br>|Z| %{y:.4g} Ω<extra></extra>",
            name="|Z|",
        )
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _finish(fig, title=title, xaxis_title="frequency (Hz)", yaxis_title="|Z| (Ω)",
                   dark=dark, showlegend=False)


def bode_phase_figure(
    frequency: np.ndarray, impedance: np.ndarray, *, dark: bool = False,
    title: str = "Impedance phase",
) -> go.Figure:
    """Phase of ``Z`` in degrees against frequency (log x)."""
    fig = go.Figure(
        go.Scatter(
            x=np.asarray(frequency),
            y=np.degrees(np.angle(np.asarray(impedance))),
            mode="lines",
            line={"color": color_for("background", dark=dark), "width": 2},
            hovertemplate="f %{x:.4g} Hz<br>phase %{y:.2f}°<extra></extra>",
            name="phase",
        )
    )
    fig.update_xaxes(type="log")
    return _finish(fig, title=title, xaxis_title="frequency (Hz)", yaxis_title="phase (°)",
                   dark=dark, showlegend=False)


def well_plate_heatmap(
    values: dict[str, float],
    *,
    rows: int = 8,
    cols: int = 12,
    dark: bool = False,
    title: str = "Plate map",
    colorbar_title: str = "",
) -> go.Figure:
    """Microplate heatmap from ``{"A1": value, ...}``."""
    row_labels = [chr(ord("A") + r) for r in range(rows)]
    grid = np.full((rows, cols), np.nan)
    for well, value in values.items():
        well = str(well).upper().strip()
        r = ord(well[0]) - ord("A")
        c = int(well[1:]) - 1
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = value
    fig = go.Figure(
        go.Heatmap(
            z=grid,
            x=[str(c + 1) for c in range(cols)],
            y=row_labels,
            colorscale=sequential_colorscale(),
            colorbar={"title": {"text": colorbar_title, "side": "right"}, "thickness": 12},
            hovertemplate="well %{y}%{x}<br>%{z:.4g}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title, xaxis_title="column", yaxis_title="row", dark=dark,
                   showlegend=False)


def dose_response_figure(
    concentration: np.ndarray,
    response: np.ndarray,
    *,
    fit: tuple[np.ndarray, np.ndarray] | None = None,
    ic50: float | None = None,
    dark: bool = False,
    title: str = "Dose-response",
) -> go.Figure:
    """Measured points plus the fitted four-parameter logistic, with IC50 marked."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.asarray(concentration),
            y=np.asarray(response),
            mode="markers",
            name="measured",
            marker={"size": 9, "color": color_for("target", dark=dark),
                    "line": {"width": 2, "color": "#fcfcfb" if not dark else "#1a1a19"}},
            hovertemplate="c %{x:.4g}<br>response %{y:.3f}<extra></extra>",
        )
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=fit[0], y=fit[1], mode="lines", name="4PL fit",
                line={"color": color_for("background", dark=dark), "width": 2},
                hovertemplate="c %{x:.4g}<br>fit %{y:.3f}<extra></extra>",
            )
        )
    if ic50 is not None and np.isfinite(ic50):
        fig.add_vline(
            x=ic50,
            line={"color": "#52514e" if not dark else "#c3c2b7", "width": 1, "dash": "dash"},
            annotation_text=f"IC50 = {ic50:.3g}",
            annotation_position="top right",
            annotation_font_size=11,
        )
    fig.update_xaxes(type="log")
    return _finish(
        fig, title=title, xaxis_title="concentration",
        yaxis_title="normalised response", dark=dark,
    )


__all__ = [
    "trajectory_figure",
    "outlet_histogram_figure",
    "force_profile_figure",
    "field_heatmap_figure",
    "sweep_heatmap_figure",
    "size_distribution_figure",
    "timeseries_figure",
    "nyquist_figure",
    "bode_magnitude_figure",
    "bode_phase_figure",
    "well_plate_heatmap",
    "dose_response_figure",
]
