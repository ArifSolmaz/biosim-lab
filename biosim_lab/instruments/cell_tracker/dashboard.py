"""Panel dashboard for the live-cell tracker."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biosim_lab.core.viz.dashboard import data_table, metric_tiles, require_panel, shell
from biosim_lab.core.viz.theme import color_for, plotly_layout

_TILE_FORMATS = {
    "n_tracks": "{:d}",
    "n_detections": "{:d}",
    "mean_track_length_frames": "{:.1f}",
    "mean_speed_um_per_min": "{:.3f}",
    "median_persistence": "{:.3f}",
    "msd_alpha": "{:.2f}",
}


def _track_overlay(movie: np.ndarray, tracks: pd.DataFrame, max_tracks: int = 60) -> go.Figure:
    """First frame with the linked trajectories drawn on top."""
    fig = go.Figure(
        go.Heatmap(z=movie[0], colorscale="Greys", reversescale=True, showscale=False,
                   hoverinfo="skip")
    )
    ids = list(pd.unique(tracks["particle"]))[:max_tracks]
    for i, pid in enumerate(ids):
        g = tracks[tracks["particle"] == pid].sort_values("frame")
        fig.add_trace(
            go.Scattergl(
                x=g["x"], y=g["y"], mode="lines",
                line={"color": color_for("mcf7" if i % 2 == 0 else "a549"), "width": 1.4},
                opacity=0.8,
                name=f"track {pid}",
                showlegend=False,
                hovertemplate=f"track {pid}<br>frame %{{customdata}}"
                              "<br>x %{x:.0f}<br>y %{y:.0f}<extra></extra>",
                customdata=g["frame"],
            )
        )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(**plotly_layout(f"{len(ids)} of {tracks['particle'].nunique()} tracks",
                                      xaxis_title="x (px)", yaxis_title="y (px)",
                                      height=560, showlegend=False))
    return fig


def _histogram(values: np.ndarray, title: str, xaxis_title: str, label: str) -> go.Figure:
    counts, edges = np.histogram(values[np.isfinite(values)], bins=24)
    fig = go.Figure(
        go.Bar(
            x=0.5 * (edges[:-1] + edges[1:]), y=counts,
            marker={"color": color_for(label), "line": {"width": 0}},
            hovertemplate="%{x:.3g}<br>%{y} tracks<extra></extra>",
            name=title,
        )
    )
    fig.update_layout(bargap=0.08)
    fig.update_layout(**plotly_layout(title, xaxis_title=xaxis_title,
                                      yaxis_title="track count", showlegend=False))
    return fig


def _msd_figure(lag: np.ndarray, msd: np.ndarray, alpha: float) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=lag / 60.0, y=msd * 1e12, mode="lines+markers",
            line={"color": color_for("mcf7"), "width": 2},
            marker={"size": 8, "line": {"width": 2, "color": "#fcfcfb"}},
            name="ensemble MSD",
            hovertemplate="lag %{x:.1f} min<br>MSD %{y:.3g} µm²<extra></extra>",
        )
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(**plotly_layout(f"Mean-squared displacement (α = {alpha:.2f})",
                                      xaxis_title="lag time (min)", yaxis_title="MSD (µm²)",
                                      showlegend=False))
    return fig


def build_dashboard(instrument: Any) -> Any:
    """Assemble the tracker dashboard from a run instrument."""
    pn = require_panel()
    result = instrument.results()
    per_track = result.table
    tracks = instrument.tracks

    tabs: list[tuple[str, Any]] = []
    if tracks is not None and len(tracks):
        tabs.append(("Tracks", pn.pane.Plotly(
            _track_overlay(instrument.movie, tracks),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=580)))
    if len(per_track):
        tabs.append(("Speed", pn.pane.Plotly(
            _histogram(per_track["mean_speed_m_s"].to_numpy() * 6e7,
                       "Mean speed per track", "speed (µm/min)", "mcf7"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=420)))
        tabs.append(("Persistence", pn.pane.Plotly(
            _histogram(per_track["persistence"].to_numpy(),
                       "Directional persistence (net / path)", "persistence", "a549"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=420)))
    if "msd" in result.fields:
        tabs.append(("MSD", pn.pane.Plotly(
            _msd_figure(result.fields["lag"].values, result.fields["msd"].values,
                        result.metrics["msd_alpha"]),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=420)))
    tabs.append(("Track table", data_table(per_track, height=380)))

    return shell(
        "Live-cell tracker",
        "Watershed detection · trackpy linking · speed, persistence and MSD",
        tiles=metric_tiles(
            result.metrics,
            ["n_tracks", "n_detections", "mean_track_length_frames",
             "mean_speed_um_per_min", "median_persistence", "msd_alpha"],
            _TILE_FORMATS,
        ),
        tabs=tabs,
        footer="Linking: Crocker & Grier, doi:10.1006/jcis.1996.0217 · "
        "MSD analysis: doi:10.1529/biophysj.105.061150",
    )


__all__ = ["build_dashboard"]
