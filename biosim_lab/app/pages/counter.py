"""The counter page."""

from __future__ import annotations

import numpy as np
import streamlit as st

from biosim_lab.app.runners import (
    run_counter,
)
from biosim_lab.app.shared import (
    PLOTLY_CONFIG,
    download_frame,
    note,
)
from biosim_lab.core.viz.theme import color_for


def page_counter() -> None:
    st.title("Automated cell counter")
    note(
        "Segment a field of view, gate it by size, classify trypan-blue viability, "
        "and convert the count into a concentration using the chamber geometry."
    )

    with st.sidebar:
        st.subheader("Sample")
        n_cells = st.slider("Cells in the field", 20, 400, 150, 10)
        dead_fraction = st.slider("Fraction dead", 0.0, 0.6, 0.15, 0.01)
        st.subheader("Optics and chamber")
        image_size = st.select_slider("Image size (px)", [256, 384, 512], value=512)
        pixel_size_um = st.slider("Pixel size (µm)", 0.2, 2.0, 0.65, 0.05)
        dilution = st.slider("Dilution factor", 1.0, 10.0, 2.0, 0.5,
                             help="2.0 is the usual 1:1 mix with trypan blue")
        st.subheader("Segmentation")
        min_radius_px = st.slider("Minimum object radius (px)", 2.0, 12.0, 4.0, 0.5)
        seed = st.number_input("Random seed", 0, 999_999, 0, 1, key="count_seed")

    out = run_counter(n_cells, dead_fraction, image_size, pixel_size_um,
                      min_radius_px, dilution, int(seed))
    m, table = out["metrics"], out["table"]

    cols = st.columns(5)
    cols[0].metric("Detected", f"{m['n_total']}",
                   delta=f"{m['n_total'] - m['ground_truth_n']} vs truth")
    viability_delta = m["viability_percent"] - m["ground_truth_viability_percent"]
    cols[1].metric("Viability", f"{m['viability_percent']:.1f} %",
                   delta=f"{viability_delta:+.1f} pt")
    cols[2].metric("Concentration", f"{m['concentration_per_ml']:.3g} /mL")
    cols[3].metric("Mean diameter", f"{m['mean_diameter_um']:.2f} µm")
    cols[4].metric("Counting error", f"±{m['counting_relative_uncertainty'] * 100:.1f} %",
                   help="Poisson: 1/√N. Counting is random, so this is unavoidable.")

    st.info(
        f"Counting {m['n_total']} cells carries an unavoidable statistical "
        f"uncertainty of ±{m['counting_relative_uncertainty'] * 100:.1f} % — before "
        "any question of whether the segmentation is right. Two people counting the "
        "same tube perfectly will still differ by about that much.",
        icon="📊",
    )

    tabs = st.tabs(["Segmentation", "Size distribution", "Objects"])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.image(out["image"], caption="Raw field of view", width="stretch",
                     clamp=True)
        with right:
            st.image(_overlay(out["image"], out["labels"]),
                     caption="Detected outlines", width="stretch")
        note(
            f"{m['n_total']} of {m['ground_truth_n']} cells found. The misses are "
            "cells touching the frame edge (discarded on purpose — a cell cut in "
            "half has no measurable size) and a few genuinely merged pairs."
        )
    with tabs[1]:
        if len(table):
            counts, edges = np.histogram(table["diameter_um"], bins=28)
            import plotly.graph_objects as go

            from biosim_lab.core.viz.theme import plotly_layout

            fig = go.Figure(
                go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=counts,
                       marker={"color": color_for("mcf7"), "line": {"width": 0}},
                       hovertemplate="d %{x:.1f} µm<br>%{y} cells<extra></extra>")
            )
            fig.update_layout(bargap=0.08)
            fig.update_layout(**plotly_layout(
                "Measured size distribution", xaxis_title="equivalent diameter (µm)",
                yaxis_title="count", showlegend=False))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    with tabs[2]:
        st.dataframe(table, width="stretch", hide_index=True)
        download_frame(table, "cell_counts.csv", "Download objects (CSV)")


def _overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Grayscale image with segmentation outlines drawn in the palette blue."""
    from skimage.segmentation import find_boundaries

    rgb = np.repeat(np.asarray(image, dtype=float)[:, :, None], 3, axis=2)
    rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-12)
    edges = find_boundaries(labels, mode="outer")
    rgb[edges] = [0.165, 0.471, 0.839]  # #2A78D6
    return (rgb * 255).astype(np.uint8)


