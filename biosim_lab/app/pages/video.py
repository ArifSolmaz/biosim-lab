"""The video-readout page.

Neither reference paper computed its sorting metrics from a model: they counted
cells in a microscope video. This page repeats that step on a simulated sort,
so the cost of the counting itself is visible rather than assumed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from biosim_lab.app.runners import run_video
from biosim_lab.app.shared import (
    download_frame,
    explained_settings,
    note,
)
from biosim_lab.core.materials import CELL_TYPES

#: Rendering dominates the cost, so the page never films on page load.
_RUN_KEY = "video_readout_ran"


def page_video() -> None:
    st.title("Video readout of a sort")
    note(
        "A published capture efficiency was <b>counted</b>, not computed: Zhang et al. "
        "2023 counted cells in a video of the chip and told the populations apart by "
        "size; Li et al. 2015 counted a calcein stain. A simulation that reports its "
        "own efficiency skips that step, and every error the step introduces. This "
        "page films the last stretch of channel before the outlets, segments and links "
        "the frames, classifies each track, and computes the metrics <b>from those "
        "counts alone</b>. The simulation's labels are used only to score the result."
    )

    with st.sidebar:
        st.subheader("Device")
        cell_keys = sorted(CELL_TYPES)
        target = st.selectbox("Target (to collect)", cell_keys,
                              index=cell_keys.index("mcf7"), key="video_target")
        background = st.selectbox("Background (to reject)", cell_keys,
                                  index=cell_keys.index("rbc"), key="video_background")
        width_um = st.slider("Channel width (µm)", 100.0, 800.0, 300.0, 10.0,
                             key="video_width")
        height_um = st.slider("Channel height (µm)", 20.0, 200.0, 50.0, 5.0,
                              key="video_height")
        length_mm = st.slider("Active length (mm)", 0.2, 10.0, 2.0, 0.1, key="video_length")

        st.subheader("Drive")
        frequency_mhz = st.slider("Frequency (MHz)", 1.0, 40.0, 6.632, 0.001,
                                  key="video_frequency")
        voltage_pp = st.slider("Drive voltage (Vpp)", 1.0, 40.0, 15.0, 0.5, key="video_voltage")
        flow_ul_min = st.slider("Flow rate (µL/min)", 0.5, 60.0, 5.0, 0.5, key="video_flow")
        inlet = st.selectbox("Inlet focusing", ["sheath_sides", "uniform", "centre", "side"],
                             index=0, key="video_inlet")
        collection_fraction = st.slider(
            "Collection outlet width (fraction of channel)", 0.05, 0.9, 1 / 3, 0.01,
            key="video_collection",
        )

        st.subheader("Camera")
        n_cells = st.slider(
            "Cells per population", 20, 200, 60, 20, key="video_n_cells",
            help="Every cell has to be rendered frame by frame, so this sets the cost "
                 "more than anything else does.",
        )
        cells_in_view = st.slider(
            "Mean cells in view", 2.0, 14.0, 6.0, 1.0, key="video_crowd",
            help="The arrival rate. Crowd the field and cells overlap, merge in "
                 "segmentation and swap identities — a real limit of video counting.",
        )
        pixel_size_um = st.slider("Pixel size (µm)", 0.5, 2.0, 1.0, 0.1, key="video_pixel")
        seed = st.number_input("Random seed", 0, 999_999, 0, 1, key="video_seed")

    st.info(
        "Rendering is the slow part — roughly 15 ms a frame, and a run is thousands of "
        "frames. The result is cached per parameter set, so a repeat view is instant.",
        icon="⏱️",
    )
    if st.button("Film this sort and count it", type="primary"):
        st.session_state[_RUN_KEY] = True
    if not st.session_state.get(_RUN_KEY):
        st.stop()

    with explained_settings():
        out = run_video(
            target, background, frequency_mhz, voltage_pp, flow_ul_min, width_um,
            height_um, length_mm, int(n_cells), collection_fraction, inlet,
            cells_in_view, pixel_size_um, int(seed),
        )

    truth = out["truth_metrics"]
    classifier = st.radio(
        "Classify the tracks by", ["size", "fluorescence"], horizontal=True,
        format_func=lambda k: {"size": "size (as Zhang et al. did)",
                               "fluorescence": "stain (as Li et al. did)"}[k],
    )
    shown = out["by"][classifier]
    video, budget = shown["video_metrics"], shown["error_budget"]

    cols = st.columns(4)
    cap_delta = video["capture_efficiency_percent"] - truth["capture_efficiency_percent"]
    cols[0].metric(
        "Capture efficiency, counted", f"{video['capture_efficiency_percent']:.1f} %",
        delta=f"{cap_delta:+.1f} pt vs the simulation",
        help=f"95 % counting interval [{video['capture_ci_low']:.1f}, "
             f"{video['capture_ci_high']:.1f}]. The simulation says "
             f"{truth['capture_efficiency_percent']:.1f} %.",
    )
    con_delta = video["contamination_rate_percent"] - truth["contamination_rate_percent"]
    cols[1].metric(
        "Contamination, counted", f"{video['contamination_rate_percent']:.1f} %",
        delta=f"{con_delta:+.1f} pt vs the simulation", delta_color="inverse",
        help=f"The simulation says {truth['contamination_rate_percent']:.1f} %.",
    )
    cols[2].metric("Cells counted", f"{budget['counted_fraction_percent']:.1f} %",
                   help=f"{budget['cells_through_window']} cells passed the window; "
                        f"{budget['missed_cells']} were never counted, "
                        f"{budget['missed_occluded']} of those because another cell "
                        "was in front of them.")
    cols[3].metric("Classified correctly", f"{budget['classification_accuracy_percent']:.1f} %",
                   help=f"outlet agrees for {budget['outlet_agreement_percent']:.1f} % "
                        "of counted cells")

    _verdict(video, truth, budget)

    tabs = st.tabs(["What the camera sees", "Error budget", "Counted vs simulated", "Tracks"])
    with tabs[0]:
        left, right = st.columns(2)
        if out["brightfield"] is not None:
            left.image(out["brightfield"], caption="Brightfield frame", width="stretch",
                       clamp=True)
        if out["fluorescence"] is not None:
            right.image(out["fluorescence"], caption="Stained channel", width="stretch",
                        clamp=True)
        g = out["geometry"]
        note(
            f"{g['n_frames']} frames at {g['frame_rate_hz']:.0f} fps over the last "
            f"{g['cols'] * g['pixel_size_m'] * 1e6:.0f} µm of "
            f"channel, {g['n_cells_filmed']} cells filmed. The frame rate is chosen so "
            "that no cell moves further between frames than the smallest cell's radius, "
            "which is what keeps the linking unambiguous."
        )
    with tabs[1]:
        st.dataframe(_budget_table(budget), width="stretch", hide_index=True)
        note(
            "Every departure of the counted number from the simulated one, cell by "
            "cell. <b>Occluded</b> means another cell sat in front of it in "
            "projection — the acoustic nodes line cells up, and cells at different "
            "heights pass each other."
        )
    with tabs[2]:
        st.dataframe(_comparison_table(out, truth), width="stretch", hide_index=True)
        st.code(shown["summary"], language=None)
    with tabs[3]:
        st.dataframe(shown["tracks"], width="stretch", hide_index=True)
        download_frame(shown["tracks"], "video_readout_tracks.csv", "Download tracks (CSV)")


def _verdict(video: dict[str, Any], truth: dict[str, Any], budget: dict[str, Any]) -> None:
    """Say which way the counting biased the answer, and why."""
    bias = video["contamination_rate_percent"] - truth["contamination_rate_percent"]
    collected = budget["missed_background_collected_percent"]
    waste = budget["missed_background_waste_percent"]
    if bias < -0.5 and collected > waste:
        st.warning(
            f"**Counting under-reported contamination by {abs(bias):.1f} points** "
            f"({truth['contamination_rate_percent']:.1f} % simulated → "
            f"{video['contamination_rate_percent']:.1f} % counted). The reason is in "
            f"the error budget: contaminants heading for the collection outlet were "
            f"missed {collected:.1f} % of the time against {waste:.1f} % for those "
            "heading to waste. They share a line with the larger target cells and "
            "hide behind them, so the cells that go uncounted are exactly the ones "
            "the number is about. A video count is biased here, not merely noisy.",
            icon="⚠️",
        )
    elif abs(bias) <= 0.5:
        st.success(
            "Counting and simulation agree to within half a point on both metrics at "
            "this operating point. Crowd the field of view, or narrow the size "
            "difference between the two populations, to find where it breaks.",
            icon="✅",
        )
    else:
        st.info(
            f"Counting moved contamination by {bias:+.1f} points "
            f"({truth['contamination_rate_percent']:.1f} % → "
            f"{video['contamination_rate_percent']:.1f} %). The error budget says "
            "which cells account for it.",
            icon="🔍",
        )


def _budget_table(budget: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Cells through the window", f"{budget['cells_through_window']}"),
        ("Counted", f"{budget['counted_fraction_percent']:.1f} %"),
        ("  targets counted", f"{budget['counted_target_percent']:.1f} %"),
        ("  background counted", f"{budget['counted_background_percent']:.1f} %"),
        ("Never counted", f"{budget['missed_cells']}"),
        ("  because occluded", f"{budget['missed_occluded']}"),
        ("  for another reason", f"{budget['missed_other']}"),
        ("Background missed, collection outlet",
         f"{budget['missed_background_collected_percent']:.1f} %"),
        ("Background missed, waste outlet",
         f"{budget['missed_background_waste_percent']:.1f} %"),
        ("Spurious tracks", f"{budget['extra_tracks']}"),
        ("Identity swaps", f"{budget['identity_swaps']}"),
        ("Misclassified: target called background", f"{budget['targets_called_background']}"),
        ("Misclassified: background called target", f"{budget['background_called_targets']}"),
        ("Outlet disagreements", f"{budget['outlet_disagreements']}"),
    ]
    return pd.DataFrame(rows, columns=["what", "value"])


def _comparison_table(out: dict[str, Any], truth: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for metric, key in [("capture efficiency", "capture_efficiency_percent"),
                        ("contamination", "contamination_rate_percent")]:
        row = {"metric": metric, "simulation": f"{truth[key]:.1f} %"}
        for by in ("size", "fluorescence"):
            row[f"video, by {by}"] = f"{out['by'][by]['video_metrics'][key]:.1f} %"
        rows.append(row)
    return pd.DataFrame(rows)
