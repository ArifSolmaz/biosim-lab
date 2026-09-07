"""biosim-lab --- Streamlit front end.

A browser interface to the same simulation core the CLI and the Panel dashboards
use. Nothing physical lives here: every number on screen comes from
``biosim_lab``, and every figure is built by
:mod:`biosim_lab.core.viz.curves`, which returns plain Plotly figures and is
therefore framework-agnostic.

Deployment notes (see docs/USER_MANUAL.md for the full walkthrough)
------------------------------------------------------------------
* Entry point for Streamlit Community Cloud: this file, at the repository root.
* Dependencies come from ``requirements.txt``, which is deliberately *smaller*
  than the ones in ``pyproject.toml``: PyVista/VTK, Gmsh, Napari, Panel and the
  NetCDF back-ends are all left out. The core degrades to a structured mesh, and
  results download as CSV instead of NetCDF. That keeps the image inside the
  memory budget of a free hosting tier.
* Because the package is not ``pip install``-ed there, entry-point discovery
  finds nothing; :mod:`biosim_lab.registry` falls back to importing the built-in
  instruments directly. The Environment page reports which route was taken.
* Every simulation is wrapped in ``st.cache_data`` keyed on its parameters, so
  dragging a slider back to a previous value is instant and the host is not
  asked to recompute the same thing twice.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from biosim_lab import __version__
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.materials import CELL_TYPES, all_values, audit, get_cell
from biosim_lab.core.plugin import MissingBackendWarning, RegimeWarning
from biosim_lab.core.viz.curves import (
    bode_magnitude_figure,
    bode_phase_figure,
    dose_response_figure,
    force_profile_figure,
    nyquist_figure,
    outlet_histogram_figure,
    size_distribution_figure,
    timeseries_figure,
    trajectory_figure,
    well_plate_heatmap,
)
from biosim_lab.core.viz.theme import PALETTE, color_for
from biosim_lab.registry import discovery_source

UL_MIN = 1e-9 / 60.0  # m^3/s per uL/min

st.set_page_config(
    page_title="biosim-lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
      [data-testid="stMetricValue"] {{ font-size: 1.9rem; }}
      .biosim-note {{
        font-size: 0.82rem; color: {PALETTE['text_secondary_light']};
        border-left: 3px solid {PALETTE['grid_light']}; padding: 0.1rem 0 0.1rem 0.7rem;
        margin: 0.4rem 0 0.9rem 0;
      }}
      .biosim-doi {{ font-size: 0.76rem; color: {PALETTE['text_secondary_light']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def note(text: str) -> None:
    """A small muted caption used to explain what a number means."""
    st.markdown(f"<div class='biosim-note'>{text}</div>", unsafe_allow_html=True)


def show_warnings(caught: list[warnings.WarningMessage]) -> None:
    """Surface the model's own regime warnings instead of swallowing them."""
    for w in caught:
        if issubclass(w.category, RegimeWarning):
            st.warning(f"**Model outside its validated range** — {w.message}", icon="⚠️")
        elif issubclass(w.category, MissingBackendWarning):
            st.info(f"**Optional back-end missing** — {w.message}", icon="ℹ️")


def download_frame(df: pd.DataFrame, filename: str, label: str) -> None:
    """CSV download button. CSV rather than Parquet/NetCDF so the hosted app
    needs neither pyarrow nor an HDF5 stack."""
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
        mime="text/csv", use_container_width=False,
    )


# ---------------------------------------------------------------------------
# cached simulation wrappers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Solving the acoustic field and tracking cells…", max_entries=48)
def run_sorter(
    frequency_mhz: float,
    voltage_pp: float,
    flow_ul_min: float,
    width_um: float,
    height_um: float,
    length_mm: float,
    n_cells: int,
    collection_fraction: float,
    inlet: str,
    mode: str,
    target: str,
    background: str,
    fem_resolution: int,
    seed: int,
) -> dict[str, Any]:
    """Run one sorter experiment. Arguments are primitives so caching works."""
    from biosim_lab.instruments.saw_sorter.simulate import (
        SAWSorterParams,
        SAWSorterSimulation,
    )

    params = SAWSorterParams(
        frequency=frequency_mhz * 1e6,
        voltage_pp=voltage_pp,
        channel_width=width_um * 1e-6,
        channel_height=height_um * 1e-6,
        channel_length=length_mm * 1e-3,
        flow_rate=flow_ul_min * UL_MIN,
        inlet=inlet,
        collection_fraction=collection_fraction,
        mode=mode,
        fem_resolution=fem_resolution,
        fem_grid=(201, 33),
        populations=[
            {"cell_type": target, "count": n_cells, "target": True},
            {"cell_type": background, "count": n_cells, "target": False},
        ],
        seed=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = SAWSorterSimulation(params)
        outcome = sim.run()

    return {
        "metrics": outcome.metrics,
        "cells": outcome.cells,
        "trajectories": outcome.tracks.trajectories,
        "diagnostics": outcome.diagnostics,
        "wavelength": sim.wavelength,
        "node_offset": sim.node_offset,
        "kappa_f": sim.fluid.kappa,
        "phi": {
            label: sim.phi_for(get_cell(pop.cell_type))
            for label, pop in zip(
                [p.resolved_label() for p in params.populations], params.populations
            )
        },
        "params": params.model_dump(mode="json"),
        "warnings": [
            {"category": w.category.__name__, "message": str(w.message)} for w in caught
        ],
    }


@st.cache_data(show_spinner="Simulating the impedance plate…", max_entries=16)
def run_rtca(
    frequency_khz: float,
    duration_h: float,
    treatment_h: float,
    true_ic50: float,
    hill: float,
    replicates: int,
    noise_cv: float,
    doubling_h: float,
    seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.impedance_rtca import ImpedanceRTCA

    cfg = ExperimentConfig(
        name="rtca", instrument="impedance_rtca",
        params={
            "frequency": frequency_khz * 1e3,
            "duration": duration_h * 3600.0,
            "treatment_time": treatment_h * 3600.0,
            "true_ic50": true_ic50,
            "hill_slope": hill,
            "replicates": replicates,
            "noise_cv": noise_cv,
            "doubling_time": doubling_h * 3600.0,
            "seed": seed,
        },
    )
    result = ImpedanceRTCA(cfg).run()
    return {
        "metrics": result.metrics,
        "table": result.table,
        "time": result.fields["time"].values,
        "cell_index": result.fields["cell_index"].values,
        "wells": [str(w) for w in result.fields["well"].values],
        "frequency": result.fields["frequency"].values,
        "z_real": result.fields["impedance_real"].values,
        "z_imag": result.fields["impedance_imag"].values,
    }


@st.cache_data(show_spinner="Segmenting the field of view…", max_entries=16)
def run_counter(
    n_cells: int, dead_fraction: float, image_size: int, pixel_size_um: float,
    min_radius_px: float, dilution: float, seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.cell_counter import CellCounter

    cfg = ExperimentConfig(
        name="counter", instrument="cell_counter",
        params={
            "n_cells": n_cells, "dead_fraction": dead_fraction,
            "image_size": image_size, "pixel_size": pixel_size_um * 1e-6,
            "min_radius_px": min_radius_px, "dilution_factor": dilution, "seed": seed,
        },
    )
    inst = CellCounter(cfg)
    result = inst.run()
    return {
        "metrics": result.metrics,
        "table": result.table,
        "image": np.asarray(inst.image),
        "labels": np.asarray(inst.segmentation.labels),
    }


@st.cache_data(show_spinner="Tracking cells frame by frame…", max_entries=12)
def run_tracker(
    n_frames: int, n_cells: int, image_size: int, speed_px: float,
    persistence: float, search_range: float, frame_interval_min: float, seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.cell_tracker import CellTracker

    cfg = ExperimentConfig(
        name="tracker", instrument="cell_tracker",
        params={
            "n_frames": n_frames, "n_cells": n_cells, "image_size": image_size,
            "speed_px_per_frame": speed_px, "persistence": persistence,
            "search_range_px": search_range,
            "frame_interval": frame_interval_min * 60.0, "seed": seed,
        },
    )
    inst = CellTracker(cfg)
    result = inst.run()
    return {
        "metrics": result.metrics,
        "per_track": result.table,
        "tracks": inst.tracks,
        "first_frame": np.asarray(inst.movie[0]),
        "msd_lag": result.fields["lag"].values if "msd" in result.fields else None,
        "msd": result.fields["msd"].values if "msd" in result.fields else None,
    }


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("biosim-lab")
    st.markdown(
        "#### Open-source virtual laboratory instruments — "
        "simulations of what commercial bio-instruments measure"
    )
    st.markdown(
        """
Four instruments share one core. Pick one from the sidebar; every control
re-runs the real simulation, not a lookup table.

| Instrument | Commercial equivalent | What it answers |
|---|---|---|
| **SAW cell sorter** | acoustic separators | can I pull tumour cells from blood, how pure? |
| **Impedance (RTCA)** | xCELLigence | how fast are they growing, what dose kills half? |
| **Cell counter** | Countess, Cellometer | how many cells per mL, how many alive? |
| **Cell tracker** | Incucyte | how fast do they crawl, and in a direction or not? |
"""
    )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Read this before trusting a number")
        st.markdown(
            """
- **The model tells you when it is out of its depth.** Orange warning banners
  are the simulation flagging that an assumption behind it has been stretched.
  They are not errors and they are not decoration — read them.
- **Every physical constant has a source.** The *Material provenance* page lists
  all 65, split into the 27 with a published DOI and the 38 that are explicit
  assumptions, each with the reason.
- **The fast model is the optimistic one.** The analytic sorter mode applies the
  sound strength at the chip surface to cells at every height. The real field
  weakens upwards. Switch the sorter to *FEM* mode to see the honest version.
"""
        )
    with right:
        st.subheader("What this is not")
        st.markdown(
            """
- It is **not** a replacement for an experiment. Its value is telling you in
  seconds that a design will not work, before you fabricate the chip.
- It does **not** predict how loud the sound is for a given drive voltage.
  That needs a piezoelectric solve this project has not implemented; there is a
  documented linear calibration standing in for it.
- Some effects are deliberately left out — acoustic streaming, cell–cell
  interaction, cell deformability. The full list is in the manual.
"""
        )

    st.divider()
    st.markdown(
        "<span class='biosim-doi'>Gor'kov radiation force: doi:10.1039/c2lc21068a"
        " &nbsp;·&nbsp; SSAW node spacing: doi:10.1039/b910595f"
        " &nbsp;·&nbsp; impedance model: doi:10.1073/pnas.88.17.7896"
        " &nbsp;·&nbsp; particle linking: doi:10.1006/jcis.1996.0217</span>",
        unsafe_allow_html=True,
    )


def page_sorter() -> None:
    st.title("SAW acoustophoretic cell sorter")
    note(
        "Two surface acoustic waves make a standing pressure pattern across a "
        "microchannel. Cells drift towards the quiet lines at a speed that scales "
        "with the <b>square of their radius</b>, so big tumour cells reach the "
        "middle within the channel and small blood cells do not."
    )

    with st.sidebar:
        st.subheader("Device")
        cell_keys = sorted(CELL_TYPES)
        target = st.selectbox("Target (to collect)", cell_keys, index=cell_keys.index("mcf7"))
        background = st.selectbox(
            "Background (to reject)", cell_keys, index=cell_keys.index("rbc")
        )
        width_um = st.slider("Channel width (µm)", 100.0, 800.0, 300.0, 10.0)
        height_um = st.slider("Channel height (µm)", 20.0, 200.0, 50.0, 5.0)
        length_mm = st.slider("Active length (mm)", 0.2, 10.0, 2.0, 0.1)

        st.subheader("Drive")
        single_node = 3979.0 / (2 * width_um * 1e-6) / 1e6
        frequency_mhz = st.slider(
            "Frequency (MHz)", 1.0, 40.0, float(round(single_node, 3)), 0.001,
            help=f"{single_node:.3f} MHz puts exactly one pressure node in a "
                 f"{width_um:.0f} µm channel.",
        )
        voltage_pp = st.slider("Drive voltage (Vpp)", 1.0, 40.0, 15.0, 0.5)
        flow_ul_min = st.slider("Flow rate (µL/min)", 0.5, 60.0, 5.0, 0.5)

        st.subheader("Sample and outlets")
        inlet = st.selectbox(
            "Inlet focusing", ["sheath_sides", "uniform", "centre"], index=0
        )
        collection_fraction = st.slider(
            "Collection outlet width (fraction of channel)", 0.05, 0.9, 1 / 3, 0.01
        )
        n_cells = st.slider("Cells per population", 50, 600, 300, 50)

        st.subheader("Model")
        mode = st.radio(
            "Force field", ["analytic", "fem"], horizontal=True,
            help="analytic = closed form, fast, optimistic. "
                 "fem = solves the wave equation, slower, realistic.",
        )
        fem_resolution = 40
        if mode == "fem":
            fem_resolution = st.slider("Mesh resolution", 16, 64, 40, 8)
        seed = st.number_input("Random seed", 0, 999_999, 12345, 1)

    out = run_sorter(
        frequency_mhz, voltage_pp, flow_ul_min, width_um, height_um, length_mm,
        n_cells, collection_fraction, inlet, mode, target, background,
        fem_resolution, int(seed),
    )
    m, d = out["metrics"], out["diagnostics"]

    cols = st.columns(5)
    cols[0].metric("Recovery", f"{m['efficiency_percent']:.1f} %",
                   help="fraction of target cells that reached the collection outlet")
    cols[1].metric("Purity", f"{m['purity_percent']:.1f} %",
                   help="fraction of the collected cells that are targets")
    cols[2].metric("Enrichment", f"{m['enrichment_fold']:.2f}×",
                   help="how much richer the output is than the input")
    cols[3].metric("Collected", f"{m['n_collected']}")
    cols[4].metric("Cells simulated", f"{m['n_cells']}")

    for w in out["warnings"]:
        if w["category"] == "RegimeWarning":
            st.warning(f"**Model outside its validated range** — {w['message']}", icon="⚠️")

    if not m["all_cells_exited"]:
        st.error(
            "Some cells never reached the outlet in the time simulated. The "
            "metrics above are computed at their last position, so treat them as "
            "unreliable — lower the flow rate or shorten the channel.",
            icon="🚫",
        )

    nodes = d["node_positions_m"]
    st.markdown(
        f"SAW wavelength **{d['saw_wavelength_m'] * 1e6:.1f} µm** &nbsp;·&nbsp; "
        f"node spacing **{d['node_spacing_m'] * 1e6:.1f} µm** &nbsp;·&nbsp; "
        f"nodes at **{', '.join(f'{v * 1e6:.0f}' for v in nodes)} µm** &nbsp;·&nbsp; "
        f"p₀ = **{d['pressure_amplitude_Pa'] / 1e6:.3f} MPa** &nbsp;·&nbsp; "
        f"mean flow **{d['mean_velocity_m_s'] * 1e3:.2f} mm/s** &nbsp;·&nbsp; "
        f"transit **{d['transit_time_s']:.2f} s** &nbsp;·&nbsp; "
        f"Re = **{m['channel_reynolds']:.3g}**"
    )

    tabs = st.tabs(
        ["Trajectories", "Outlet histogram", "Force profile", "Size distribution",
         "Per-population", "Data"]
    )
    half = 0.5 * collection_fraction * width_um * 1e-6

    with tabs[0]:
        st.plotly_chart(
            trajectory_figure(
                out["trajectories"], node_positions=nodes,
                channel_width=width_um * 1e-6, channel_length=length_mm * 1e-3,
                title="Cell paths through the acoustic field",
            ),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "Each line is one cell, seen from above. The dotted line is the "
            "pressure node. A working device separates the two colours before the "
            "right-hand edge of the plot, which is the end of the channel."
        )

    with tabs[1]:
        st.plotly_chart(
            outlet_histogram_figure(
                out["cells"], channel_width=width_um * 1e-6,
                collection_bounds=(out["node_offset"] - half, out["node_offset"] + half),
            ),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "Where each cell is when it leaves. The shaded band is the collection "
            "outlet; overlap inside it is what limits purity."
        )

    with tabs[2]:
        from biosim_lab.instruments.saw_sorter.physics.acoustics import (
            primary_radiation_force_1d,
        )

        x = np.linspace(0.0, width_um * 1e-6, 400)
        profiles = {}
        for label, phi in out["phi"].items():
            cell = get_cell(label if label in CELL_TYPES else target)
            profiles[label] = primary_radiation_force_1d(
                x, p0=d["pressure_amplitude_Pa"],
                volume=4 / 3 * np.pi * cell.r**3, kappa_f=out["kappa_f"],
                wavelength=out["wavelength"], phi=phi, node_offset=out["node_offset"],
            )
        st.plotly_chart(
            force_profile_figure(x, profiles, node_positions=nodes),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "The sideways push, in piconewtons, across the channel. It is zero at "
            "the node — a cell that arrives there stops. The gap between the two "
            "curves is the whole separation, and it comes almost entirely from the "
            "cube of the radius."
        )

    with tabs[3]:
        st.plotly_chart(
            size_distribution_figure(out["cells"]),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "Radii are drawn from a log-normal distribution, as real populations "
            "are. Because force scales with r³, this spread is the main reason "
            "purity is never 100 %."
        )

    with tabs[4]:
        rows = [{"population": k, **v} for k, v in m["per_population"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("**Acoustic contrast factors**")
        contrast = pd.DataFrame(
            [
                {"population": k, "Φ (classical)": v["phi_classical"],
                 "Φ effective (this device)": v["phi_effective_ssaw"]}
                for k, v in d["contrast_factors"].items()
            ]
        )
        st.dataframe(contrast, use_container_width=True, hide_index=True)
        note(
            "Φ > 0 means the cell moves to the quiet node; Φ < 0 would send it to "
            "the loud antinode. The effective value is lower because in a surface-"
            "wave device the sound does not travel straight across the channel."
        )

    with tabs[5]:
        st.dataframe(out["cells"].head(200), use_container_width=True, hide_index=True)
        download_frame(out["cells"], "saw_sorter_cells.csv", "Download all cells (CSV)")
        st.download_button(
            "Download the configuration (YAML)",
            _params_to_yaml(out["params"]).encode("utf-8"),
            file_name="saw_sorter_config.yaml", mime="text/yaml",
        )
        note(
            "The YAML runs unchanged on the command line: "
            "<code>biosim run saw_sorter_config.yaml</code>"
        )


def _params_to_yaml(params: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(
        {"name": "from_streamlit", "instrument": "saw_sorter", "params": params},
        sort_keys=False, allow_unicode=True,
    )


def page_rtca() -> None:
    st.title("Real-time cell impedance analyser")
    note(
        "Cells growing on a gold electrode obstruct the current, because their "
        "membranes do not conduct. Impedance therefore reports how covered the "
        "electrode is — continuously, without touching the cells."
    )

    with st.sidebar:
        st.subheader("Measurement")
        frequency_khz = st.slider("Readout frequency (kHz)", 1.0, 100.0, 10.0, 1.0)
        duration_h = st.slider("Experiment duration (h)", 12.0, 120.0, 48.0, 6.0)
        st.subheader("Biology")
        doubling_h = st.slider("Doubling time (h)", 8.0, 48.0, 20.0, 1.0)
        st.subheader("Treatment")
        treatment_h = st.slider("Drug added at (h)", 0.0, 72.0, 24.0, 1.0)
        true_ic50 = st.slider("True IC50 (arbitrary units)", 0.05, 20.0, 1.0, 0.05)
        hill = st.slider("Hill slope", 0.5, 4.0, 1.3, 0.1)
        replicates = st.slider("Replicates per dose", 1, 6, 3, 1)
        noise_cv = st.slider("Measurement noise (CV)", 0.0, 0.10, 0.02, 0.005)
        seed = st.number_input("Random seed", 0, 999_999, 7, 1, key="rtca_seed")

    uploaded = st.file_uploader(
        "Optional: upload a real RTCA export (CSV or XLSX) to analyse instead",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is not None:
        _rtca_from_upload(uploaded)
        return

    out = run_rtca(
        frequency_khz, duration_h, treatment_h, true_ic50, hill, replicates,
        noise_cv, doubling_h, int(seed),
    )
    m, table = out["metrics"], out["table"]

    cols = st.columns(5)
    cols[0].metric("Fitted IC50", f"{m.get('ic50', float('nan')):.3g}",
                   delta=f"{m.get('ic50', float('nan')) / true_ic50:.2f}× the planted value")
    cols[1].metric("Standard error", f"{m.get('ic50_stderr', float('nan')):.2g}")
    cols[2].metric("Hill slope", f"{m.get('hill_slope', float('nan')):.2f}")
    cols[3].metric("Fit R²", f"{m.get('fit_r_squared', float('nan')):.4f}")
    cols[4].metric("Peak Cell Index", f"{m['max_cell_index']:.2f}")

    if "ic50" in m and abs(m["ic50"] / true_ic50 - 1) > 0.15:
        st.info(
            f"The fitted IC50 ({m['ic50']:.3g}) differs from the planted value "
            f"({true_ic50:.3g}) because the drug has only acted for "
            f"{duration_h - treatment_h:.0f} h against a {doubling_h:.0f} h doubling "
            "time — the population has not finished responding. Real endpoint "
            "assays behave the same way, which is why a published IC50 is "
            "meaningless without its exposure time. Lengthen the experiment and "
            "watch the fit converge.",
            icon="🧪",
        )

    tabs = st.tabs(["Cell Index", "Dose–response", "Nyquist", "Bode", "Plate map", "Data"])

    curves = pd.DataFrame({"time": out["time"]})
    for dose, group in table.groupby("concentration"):
        idx = [out["wells"].index(w) for w in group["well"] if w in out["wells"]]
        if idx:
            curves["control" if dose == 0 else f"{dose:g}"] = (
                out["cell_index"][:, idx].mean(axis=1)
            )

    with tabs[0]:
        st.plotly_chart(
            timeseries_figure(curves, title="Cell Index over time, by dose",
                              yaxis_title="Cell Index (dimensionless)"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "Zero is a bare electrode. The curves rise as cells attach and divide, "
            "and separate by dose after the drug goes in."
        )

    z = out["z_real"] + 1j * out["z_imag"]
    with tabs[1]:
        if "ic50" in m:
            from biosim_lab.instruments.impedance_rtca.physics import (
                four_parameter_logistic,
            )

            treated = table[(table["kind"] == "treated") & (table["concentration"] > 0)]
            grouped = treated.groupby("concentration")["normalised_response"].mean()
            grid = np.logspace(
                np.log10(grouped.index.min()) - 0.5,
                np.log10(grouped.index.max()) + 0.5, 200,
            )
            fitted = four_parameter_logistic(
                grid, float(grouped.min()), float(grouped.max()),
                m["ic50"], m["hill_slope"],
            )
            st.plotly_chart(
                dose_response_figure(
                    grouped.index.to_numpy(), grouped.to_numpy(),
                    fit=(grid, fitted), ic50=m["ic50"],
                ),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
            note(
                "Each point is the mean of the replicates at that dose, normalised "
                "to the untreated controls. The curve is a four-parameter logistic; "
                "IC50 is where it crosses halfway."
            )
        else:
            st.info("Not enough doses to fit a curve.")

    with tabs[2]:
        st.plotly_chart(
            nyquist_figure(out["frequency"], z[:, -1],
                           title="Nyquist — confluent monolayer"),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note(
            "Impedance in the complex plane, swept across frequency. The arc shape "
            "is the fingerprint of a capacitor (the membranes) in parallel with a "
            "resistor (the gaps between cells)."
        )

    with tabs[3]:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                bode_magnitude_figure(out["frequency"], z[:, 0],
                                      title="|Z| — bare electrode"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
        with right:
            st.plotly_chart(
                bode_phase_figure(out["frequency"], z[:, 0],
                                  title="Phase — bare electrode"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
        note(
            "Magnitude and phase are two separate plots on purpose. Putting two "
            "quantities with different units on one pair of axes is the most common "
            "way to make a chart lie."
        )

    with tabs[4]:
        st.plotly_chart(
            well_plate_heatmap(
                dict(zip(table["well"], table["endpoint_cell_index"])),
                title="Endpoint Cell Index by well", colorbar_title="CI",
            ),
            use_container_width=True, config=PLOTLY_CONFIG,
        )
        note("Column 1 holds the untreated controls; dose increases left to right.")

    with tabs[5]:
        st.dataframe(table, use_container_width=True, hide_index=True)
        download_frame(table, "rtca_wells.csv", "Download well table (CSV)")


def _rtca_from_upload(uploaded: io.BytesIO) -> None:
    """Analyse a real RTCA export uploaded through the browser."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from biosim_lab.core.io import read_rtca_csv

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / uploaded.name
        path.write_bytes(uploaded.getvalue())
        try:
            ds = read_rtca_csv(path)
        except Exception as exc:  # noqa: BLE001 - user file, report don't crash
            st.error(
                f"Could not read that file: {exc}\n\nThe reader expects a table "
                "with a time column and one column per well labelled A1 … H12. "
                "Metadata rows above the header are skipped automatically.",
                icon="🚫",
            )
            return

    ci = ds["cell_index"].values
    wells = [str(w) for w in ds["well"].values]
    st.success(f"Read {len(wells)} wells over {ds['time'].values[-1] / 3600:.1f} hours.")

    cols = st.columns(3)
    cols[0].metric("Wells", len(wells))
    cols[1].metric("Duration", f"{ds['time'].values[-1] / 3600:.1f} h")
    cols[2].metric("Peak Cell Index", f"{np.nanmax(ci):.2f}")

    df = pd.DataFrame({"time": ds["time"].values})
    for j, well in enumerate(wells[:16]):
        df[well] = ci[:, j]
    st.plotly_chart(
        timeseries_figure(df, title="Cell Index — uploaded data (first 16 wells)",
                          yaxis_title="Cell Index"),
        use_container_width=True, config=PLOTLY_CONFIG,
    )
    long = pd.DataFrame(ci, columns=wells)
    long.insert(0, "time_s", ds["time"].values)
    download_frame(long, "rtca_uploaded.csv", "Download parsed data (CSV)")


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
            st.image(out["image"], caption="Raw field of view", use_container_width=True,
                     clamp=True)
        with right:
            st.image(_overlay(out["image"], out["labels"]),
                     caption="Detected outlines", use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with tabs[2]:
        st.dataframe(table, use_container_width=True, hide_index=True)
        download_frame(table, "cell_counts.csv", "Download objects (CSV)")


def _overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Grayscale image with segmentation outlines drawn in the palette blue."""
    from skimage.segmentation import find_boundaries

    rgb = np.repeat(np.asarray(image, dtype=float)[:, :, None], 3, axis=2)
    rgb = (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-12)
    edges = find_boundaries(labels, mode="outer")
    rgb[edges] = [0.165, 0.471, 0.839]  # #2A78D6
    return (rgb * 255).astype(np.uint8)


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
                        use_container_width=True, config=PLOTLY_CONFIG)
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
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
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
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            note(
                "On log axes the slope of this line <i>is</i> α. A slope of 1 is "
                "diffusion; a slope of 2 is straight-line travel."
            )
    with tabs[3]:
        st.dataframe(out["per_track"], use_container_width=True, hide_index=True)
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


def page_materials() -> None:
    st.title("Material provenance")
    note(
        "Every physical constant in this software carries either a published "
        "reference or an explicit label saying it was assumed, with the reason. "
        "There is no third category, and an automated test refuses to let anyone "
        "add one."
    )

    assumptions = audit()
    total = len(all_values())

    cols = st.columns(3)
    cols[0].metric("Values in the library", total)
    cols[1].metric("With a published DOI", total - len(assumptions))
    cols[2].metric("Flagged as assumptions", len(assumptions))

    st.subheader("Everything, with its source")
    everything = pd.DataFrame(
        [
            {
                "group": group, "material": key, "property": prop,
                "value": f"{v.magnitude:g} {v.unit}",
                "kind": "ASSUMPTION" if v.prov.is_assumption else "published",
                "source": (v.prov.assumption if v.prov.is_assumption
                           else f"doi:{v.prov.doi} — {v.prov.citation}"),
            }
            for group, key, prop, v in all_values()
        ]
    )
    only_assumptions = st.checkbox(
        "Show only the assumptions (the list a methods section should disclose)",
        value=False,
    )
    view = everything[everything["kind"] == "ASSUMPTION"] if only_assumptions else everything
    st.dataframe(view, use_container_width=True, hide_index=True, height=460)
    download_frame(everything, "biosim_lab_materials.csv", "Download the whole library (CSV)")

    st.subheader("The single biggest assumption")
    st.warning(
        "Nothing in this software predicts how loud the sound is for a given drive "
        "voltage — that needs a piezoelectric solve this project has not "
        "implemented. A straight-line calibration stands in for it: 15 Vpp gives "
        "0.45 MPa. It scales **every** acoustic force here. If you build the "
        "device, measure the pressure by tracking calibration beads and enter that "
        "number instead of a voltage.",
        icon="⚠️",
    )


def page_environment() -> None:
    st.title("Environment")
    note("What is installed here, what is not, and what that costs you.")

    from biosim_lab.core.geometry import gmsh_available
    from biosim_lab.core.plugin import optional_import
    from biosim_lab.core.solver import solver_report
    from biosim_lab.instruments.cell_counter.segmentation import available_backends

    st.metric("biosim-lab version", __version__)
    st.caption(f"Instruments found via {discovery_source()}")

    rows = []
    for module, unlocks in (
        ("numpy", "core"), ("scipy", "core"), ("xarray", "core"),
        ("pandas", "core"), ("pydantic", "configuration validation"),
        ("pint", "unit checking"), ("skfem", "built-in FEM solvers"),
        ("plotly", "figures"), ("skimage", "segmentation"),
        ("trackpy", "track linking"), ("streamlit", "this interface"),
        ("meshio", "Gmsh mesh import (optional)"),
        ("pyvista", "3-D field rendering (optional, omitted from this deployment)"),
        ("panel", "the desktop dashboards (optional)"),
        ("napari", "interactive image review (optional)"),
        ("pyarrow", "Parquet output (optional)"),
        ("h5netcdf", "NetCDF output (optional)"),
    ):
        mod = optional_import(module)
        rows.append({
            "component": module,
            "status": "installed" if mod is not None else "missing",
            "what it unlocks": unlocks,
        })

    ok, why = gmsh_available()
    rows.append({"component": "gmsh", "status": "installed" if ok else "missing",
                 "what it unlocks": why})
    for name, (avail, detail) in available_backends().items():
        rows.append({"component": f"segmentation: {name}",
                     "status": "installed" if avail else "missing",
                     "what it unlocks": detail})
    for row in solver_report():
        rows.append({"component": f"solver: {row['name']}",
                     "status": "available" if row["available"] else "missing",
                     "what it unlocks": f"{row['kind']} — {row['detail']}"})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=520)

    st.subheader("What is deliberately absent from this deployment")
    st.markdown(
        """
- **PyVista / VTK** — 3-D field rendering and trajectory movies. Left out because
  it is large and needs OpenGL; the Plotly figures here show the same fields in 2-D.
- **Gmsh** — only needed to mesh a PDMS wall layer. Without it the straight-channel
  template falls back to an exact structured triangulation, which is what every
  simulation on this site uses anyway.
- **Panel / Bokeh** — the desktop dashboards. This *is* the dashboard.
- **NetCDF and Parquet writers** — results download as CSV instead.
- **OpenFOAM and Elmer** — the heavy external solvers. Without them acoustic
  streaming is not modelled and the analytic Rayleigh approximation is used, and
  the piezoelectric problem is replaced by the documented voltage calibration.

None of these change any number you see. They change what you could *additionally*
compute if you ran the project locally.
"""
    )


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "SAW cell sorter": page_sorter,
    "Impedance (RTCA)": page_rtca,
    "Cell counter": page_counter,
    "Cell tracker": page_tracker,
    "Material provenance": page_materials,
    "Environment": page_environment,
}


def main() -> None:
    with st.sidebar:
        st.markdown("### 🔬 biosim-lab")
        choice = st.radio("Instrument", list(PAGES), label_visibility="collapsed")
        st.divider()

    PAGES[choice]()

    with st.sidebar:
        st.divider()
        st.caption(
            f"biosim-lab {__version__} · MIT licence\n\n"
            "Every number on screen is computed live from the same code the "
            "command line runs. Nothing is pre-baked."
        )


if __name__ == "__main__":
    main()
