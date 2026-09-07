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


def live_view_figure(
    trajectories: xr.Dataset,
    *,
    channel_width: float,
    channel_length: float,
    node_positions: Sequence[float] = (),
    alive: np.ndarray | None = None,
    collection_bounds: tuple[float, float] | None = None,
    max_cells: int = 400,
    n_frames: int = 40,
    frame_ms: int = 90,
    dark: bool = False,
    title: str = "Live view — cells moving through the channel",
) -> go.Figure:
    """An animated top view of the individual cells, as a microscope would see them.

    A trajectory plot answers "where did the cells end up". This answers "what
    does the device look like while it is running": every cell is a dot, sized by
    its actual radius, moving left to right down the channel and drifting
    towards the pressure node. Press play.

    Parameters
    ----------
    alive:
        Optional boolean array, one entry per cell. Dead cells are drawn hollow
        and grey, so it is visible at a glance whether the collection outlet is
        being contaminated by debris.
    collection_bounds:
        ``(lo, hi)`` in metres; shaded to show which cells are being collected.
    max_cells, n_frames:
        Animation frames are sent to the browser in full, so both are capped to
        keep the figure a sensible size.
    """
    pos = trajectories["position"].values
    labels = np.asarray(trajectories["label"].values, dtype=str)
    radii = np.asarray(trajectories["radius"].values, dtype=float)
    times = np.asarray(trajectories["time"].values, dtype=float)

    if pos.shape[0] > max_cells:
        keep = np.linspace(0, pos.shape[0] - 1, max_cells).astype(int)
        pos, labels, radii = pos[keep], labels[keep], radii[keep]
        alive = None if alive is None else np.asarray(alive)[keep]
    if alive is None:
        alive = np.ones(pos.shape[0], dtype=bool)
    alive = np.asarray(alive, dtype=bool)

    # Only animate while the cells are still inside the device.
    inside = np.flatnonzero(times <= _exit_time(pos, channel_length, times))
    frame_idx = np.unique(
        np.linspace(0, (inside[-1] if inside.size else len(times) - 1),
                    min(n_frames, len(times))).astype(int)
    )

    populations = sorted(set(labels))
    # Marker area should track cell volume the way the eye reads area, so the
    # size is proportional to radius, not to r^3 — a 9 um cell must not appear
    # 34 times bigger than a 2.8 um one.
    sizes = 4.0 + 22.0 * (radii / radii.max())

    def traces_at(k: int) -> list[go.Scatter]:
        out = []
        for label in populations:
            for state, opacity, name in ((True, 0.9, label), (False, 0.55, f"{label} (dead)")):
                sel = (labels == label) & (alive == state)
                if not sel.any():
                    continue
                out.append(go.Scatter(
                    x=pos[sel, k, 2] * 1e3,
                    y=pos[sel, k, 0] * UM,
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    marker={
                        "size": sizes[sel],
                        "color": color_for(label, dark=dark) if state else "#B9B7B0",
                        "opacity": opacity,
                        "line": {"width": 1.2,
                                 "color": "#fcfcfb" if not dark else "#1a1a19"},
                        "symbol": "circle" if state else "circle-open",
                    },
                    customdata=np.column_stack([radii[sel] * UM,
                                                pos[sel, k, 1] * UM]),
                    hovertemplate=(f"<b>{name}</b><br>z %{{x:.2f}} mm"
                                   "<br>x %{y:.1f} µm<br>depth %{customdata[1]:.1f} µm"
                                   "<br>radius %{customdata[0]:.2f} µm<extra></extra>"),
                ))
        return out

    fig = go.Figure(
        data=traces_at(frame_idx[0]),
        frames=[
            go.Frame(data=traces_at(k), name=f"{times[k]:.3f}")
            for k in frame_idx
        ],
    )

    if collection_bounds is not None:
        lo, hi = collection_bounds
        fig.add_hrect(y0=lo * UM, y1=hi * UM, fillcolor="#2a78d6", opacity=0.07,
                      line_width=0, layer="below",
                      annotation_text="collection outlet",
                      annotation_position="top left", annotation_font_size=10)
    for node in node_positions:
        fig.add_hline(y=node * UM, line={"color": "#52514e" if not dark else "#c3c2b7",
                                         "width": 1, "dash": "dot"})

    fig.update_xaxes(range=[0, channel_length * 1e3])
    fig.update_yaxes(range=[0, channel_width * UM])
    fig.update_layout(
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0, "y": -0.30,
            "xanchor": "left", "yanchor": "top", "pad": {"r": 8, "t": 0},
            "showactive": False,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "❚❚ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.17, "len": 0.83, "y": -0.24,
            "xanchor": "left", "yanchor": "top",
            "currentvalue": {"prefix": "t = ", "suffix": " s", "font": {"size": 12},
                             "xanchor": "right"},
            "pad": {"t": 0, "b": 0},
            "steps": [
                {"label": f"{times[k]:.2f}", "method": "animate",
                 "args": [[f"{times[k]:.3f}"],
                          {"frame": {"duration": 0, "redraw": True},
                           "mode": "immediate"}]}
                for k in frame_idx
            ],
        }],
    )
    out = _finish(
        fig, title=title, xaxis_title="axial position z (mm)",
        yaxis_title="lateral position x (µm)", dark=dark, height=560,
    )
    # Leave room for the play controls and the time slider underneath.
    out.update_layout(margin={"l": 64, "r": 20, "t": 56, "b": 130})
    return out


def _exit_time(pos: np.ndarray, channel_length: float, times: np.ndarray) -> float:
    """Time by which the median cell has left the device."""
    z = pos[:, :, 2]
    reached = np.array([
        np.interp(channel_length, z[i], times) if z[i, -1] >= channel_length else times[-1]
        for i in range(z.shape[0])
    ])
    return float(np.percentile(reached, 90))


def cross_section_figure(
    trajectories: xr.Dataset,
    *,
    channel_width: float,
    channel_height: float,
    channel_length: float | None = None,
    frame: int | None = None,
    alive: np.ndarray | None = None,
    node_positions: Sequence[float] = (),
    max_cells: int = 500,
    dark: bool = False,
    title: str = "Looking down the channel",
) -> go.Figure:
    """The channel cross-section at one instant, cells drawn to scale.

    This is the view you would get by cutting the chip and looking along it:
    lateral position across, depth up. The dots are drawn **to scale** --- their
    diameter is the cell's real diameter in the same units as the axes --- so the
    size difference that drives the whole separation is visible directly.

    Parameters
    ----------
    channel_length:
        When given, the snapshot is taken at the moment the median cell reaches
        the outlet rather than at the end of the integration window. Those are
        very different pictures: integration runs on until the slowest
        wall-hugging cell escapes, by which time even the background population
        has drifted onto the node, which is not what leaves the device.
    frame:
        Explicit time index, overriding *channel_length*.
    """
    pos = trajectories["position"].values
    labels = np.asarray(trajectories["label"].values, dtype=str)
    radii = np.asarray(trajectories["radius"].values, dtype=float)
    times = np.asarray(trajectories["time"].values, dtype=float)
    if frame is None:
        if channel_length is None:
            frame = -1
        else:
            t_exit = _exit_time(pos, channel_length, times)
            frame = int(np.argmin(np.abs(times - np.median([t_exit]))))
    if pos.shape[0] > max_cells:
        keep = np.linspace(0, pos.shape[0] - 1, max_cells).astype(int)
        pos, labels, radii = pos[keep], labels[keep], radii[keep]
        alive = None if alive is None else np.asarray(alive)[keep]
    if alive is None:
        alive = np.ones(pos.shape[0], dtype=bool)
    alive = np.asarray(alive, dtype=bool)

    fig = go.Figure()
    for label in sorted(set(labels)):
        for state, name in ((True, label), (False, f"{label} (dead)")):
            sel = (labels == label) & (alive == state)
            if not sel.any():
                continue
            fig.add_trace(go.Scatter(
                x=pos[sel, frame, 0] * UM,
                y=pos[sel, frame, 1] * UM,
                mode="markers",
                name=name,
                marker={
                    # Diameter in data units -> the dots really are to scale.
                    "size": 2 * radii[sel] * UM,
                    "sizemode": "diameter",
                    "color": color_for(label, dark=dark) if state else "#B9B7B0",
                    "opacity": 0.85 if state else 0.5,
                    "line": {"width": 1, "color": "#fcfcfb" if not dark else "#1a1a19"},
                    "symbol": "circle" if state else "circle-open",
                },
                customdata=radii[sel] * UM,
                hovertemplate=(f"<b>{name}</b><br>x %{{x:.1f}} µm<br>depth %{{y:.1f}} µm"
                               "<br>radius %{customdata:.2f} µm<extra></extra>"),
            ))
    for node in node_positions:
        fig.add_vline(x=node * UM,
                      line={"color": "#52514e" if not dark else "#c3c2b7",
                            "width": 1, "dash": "dot"})
    # constrain="domain" shrinks the plot box to honour equal scaling, instead of
    # widening the y range and leaving the channel floating in empty space.
    fig.update_xaxes(range=[0, channel_width * UM], constrain="domain")
    fig.update_yaxes(range=[0, channel_height * UM], scaleanchor="x", scaleratio=1,
                     constrain="domain")
    return _finish(fig, title=title, xaxis_title="lateral position x (µm)",
                   yaxis_title="depth y (µm)", dark=dark, height=300)


def cumulative_count_figure(
    trajectories: xr.Dataset,
    cells: pd.DataFrame,
    *,
    channel_length: float,
    collection_bounds: tuple[float, float],
    dark: bool = False,
    title: str = "Live outlet count",
) -> go.Figure:
    """Running tally of cells leaving each outlet, as an instrument would show it.

    Each cell is counted at the moment it crosses the outlet plane, so the slope
    of these lines is the throughput in cells per second — the number that
    decides whether a device is fast enough to process a clinical sample.
    """
    lo, hi = collection_bounds
    crossed = cells["residence_time_s"].to_numpy(dtype=float)
    collected = ((cells["x_outlet_m"] >= lo) & (cells["x_outlet_m"] <= hi)).to_numpy()
    labels = cells["label"].to_numpy(dtype=str)
    alive = (cells["alive"].to_numpy(dtype=bool) if "alive" in cells
             else np.ones(len(cells), dtype=bool))

    fig = go.Figure()
    for label in sorted(set(labels)):
        for outlet, dash in (("collect", "solid"), ("waste", "dot")):
            sel = (labels == label) & (collected if outlet == "collect" else ~collected)
            sel = sel & alive
            if not sel.any():
                continue
            t_cross = np.sort(crossed[sel][np.isfinite(crossed[sel])])
            counts = np.arange(1, t_cross.size + 1)
            fig.add_trace(go.Scatter(
                x=np.concatenate([[0.0], t_cross]),
                y=np.concatenate([[0], counts]),
                mode="lines",
                line={"color": color_for(label, dark=dark), "width": 2, "dash": dash},
                name=f"{label} → {outlet}",
                hovertemplate=(f"<b>{label} → {outlet}</b><br>t %{{x:.3f}} s"
                               "<br>%{y} cells<extra></extra>"),
            ))
    fig.update_layout(hovermode="x unified")
    return _finish(fig, title=title, xaxis_title="time (s)",
                   yaxis_title="live cells counted", dark=dark)


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
    "live_view_figure",
    "cross_section_figure",
    "cumulative_count_figure",
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
