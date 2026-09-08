"""The tracker page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from biosim_lab.app.runners import (
    run_tracker,
)
from biosim_lab.app.shared import (
    PLOTLY_CONFIG,
    download_frame,
    explained_settings,
    note,
)
from biosim_lab.core.viz.theme import color_for


def page_tracker() -> None:
    st.title("Live-cell tracker")
    note(
        "Segment every frame, link the detections between frames, and reduce the "
        "paths to numbers: speed, how straight the motion is, and whether the cells "
        "are wandering randomly or going somewhere."
    )

    with st.sidebar:
        st.subheader("Movie")
        n_frames = st.slider("Frames", 8, 30, 20, 1)
        n_cells = st.slider("Cells", 10, 120, 60, 5)
        image_size = st.select_slider("Image size (px)", [192, 256, 320], value=256)
        frame_interval_min = st.slider("Minutes between frames", 1.0, 30.0, 10.0, 1.0)
        st.subheader("Cell behaviour")
        speed_px = st.slider("Speed (px per frame)", 0.5, 6.0, 2.2, 0.1)
        persistence = st.slider(
            "Directional persistence", 0.0, 0.98, 0.85, 0.01,
            help="0 = a random walk, 1 = a straight line",
        )
        st.subheader("Linking")
        search_range = st.slider(
            "Search range (px)", 4.0, 30.0, 12.0, 1.0,
            help="Maximum distance a cell may move between frames. Too small "
                 "loses fast cells; too large swaps identities.",
        )
        seed = st.number_input("Random seed", 0, 999_999, 2, 1, key="track_seed")

    if search_range < speed_px * 1.5:
        st.warning(
            f"The search range ({search_range:.0f} px) is close to the distance "
            f"cells actually move ({speed_px:.1f} px per frame). Tracks will "
            "fragment. A rule of thumb is 3–5× the typical step.",
            icon="⚠️",
        )

    with explained_settings():
        out = run_tracker(n_frames, n_cells, image_size, speed_px, persistence,
                          search_range, frame_interval_min, int(seed))
    m = out["metrics"]

    cols = st.columns(5)
    cols[0].metric("Tracks found", f"{m['n_tracks']}",
                   delta=f"{m['n_tracks'] - m['ground_truth_n_tracks']} vs truth")
    speed_delta = m["mean_speed_um_per_min"] - m["ground_truth_speed_um_per_min"]
    cols[1].metric("Mean speed", f"{m['mean_speed_um_per_min']:.4f} µm/min",
                   delta=f"{speed_delta:+.4f}")
    cols[2].metric("Persistence", f"{m['median_persistence']:.3f}")
    cols[3].metric("MSD exponent α", f"{m['msd_alpha']:.2f}",
                   help="1 = random wandering, 2 = walking in a straight line")
    cols[4].metric("Mean track length", f"{m['mean_track_length_frames']:.1f} frames")

    alpha = m["msd_alpha"]
    verdict = ("mostly directed motion" if alpha > 1.5
               else "mixed" if alpha > 1.2 else "essentially random wandering")
    st.info(
        f"α = {alpha:.2f} means **{verdict}**. A drunk stumbling randomly gives "
        "α = 1; someone walking somewhere gives α = 2. Cancer cells becoming more "
        "directional is what invasion looks like under a microscope.",
        icon="🧭",
    )

    tabs = st.tabs(["Tracks", "Speed", "MSD", "Data"])
    with tabs[0]:
        st.plotly_chart(_tracks_figure(out["first_frame"], out["tracks"]),
                        width="stretch", config=PLOTLY_CONFIG)
    with tabs[1]:
        import plotly.graph_objects as go

        from biosim_lab.core.viz.theme import plotly_layout

        speeds = out["per_track"]["mean_speed_m_s"].to_numpy() * 6e7
        counts, edges = np.histogram(speeds[np.isfinite(speeds)], bins=20)
        fig = go.Figure(go.Bar(
            x=0.5 * (edges[:-1] + edges[1:]), y=counts,
            marker={"color": color_for("mcf7"), "line": {"width": 0}},
            hovertemplate="%{x:.3f} µm/min<br>%{y} tracks<extra></extra>"))
        fig.update_layout(bargap=0.08)
        fig.update_layout(**plotly_layout("Speed per track",
                                          xaxis_title="mean speed (µm/min)",
                                          yaxis_title="tracks", showlegend=False))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    with tabs[2]:
        if out["msd"] is not None:
            import plotly.graph_objects as go

            from biosim_lab.core.viz.theme import plotly_layout

            fig = go.Figure(go.Scatter(
                x=out["msd_lag"] / 60.0, y=out["msd"] * 1e12, mode="lines+markers",
                line={"color": color_for("mcf7"), "width": 2},
                marker={"size": 8, "line": {"width": 2, "color": "#fcfcfb"}},
                hovertemplate="lag %{x:.1f} min<br>MSD %{y:.3g} µm²<extra></extra>"))
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            fig.update_layout(**plotly_layout(
                f"Mean-squared displacement (α = {alpha:.2f})",
                xaxis_title="lag time (min)", yaxis_title="MSD (µm²)",
                showlegend=False))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            note(
                "On log axes the slope of this line <i>is</i> α. A slope of 1 is "
                "diffusion; a slope of 2 is straight-line travel."
            )
    with tabs[3]:
        st.dataframe(out["per_track"], width="stretch", hide_index=True)
        download_frame(out["per_track"], "tracks.csv", "Download tracks (CSV)")


def _tracks_figure(frame: np.ndarray, tracks: pd.DataFrame, max_tracks: int = 60):
    import plotly.graph_objects as go

    from biosim_lab.core.viz.theme import plotly_layout

    fig = go.Figure(go.Heatmap(z=frame, colorscale="Greys", reversescale=True,
                               showscale=False, hoverinfo="skip"))
    ids = list(pd.unique(tracks["particle"]))[:max_tracks]
    for i, pid in enumerate(ids):
        g = tracks[tracks["particle"] == pid].sort_values("frame")
        fig.add_trace(go.Scattergl(
            x=g["x"], y=g["y"], mode="lines", showlegend=False,
            line={"color": color_for("mcf7" if i % 2 == 0 else "a549"), "width": 1.5},
            opacity=0.85, hovertemplate=f"track {pid}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(**plotly_layout(
        f"{len(ids)} of {tracks['particle'].nunique()} tracks, over the first frame",
        xaxis_title="x (px)", yaxis_title="y (px)", height=560, showlegend=False))
    return fig


