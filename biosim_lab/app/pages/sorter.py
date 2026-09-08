"""The sorter page."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from biosim_lab.app.runners import (
    run_replicates,
    run_sorter,
)
from biosim_lab.app.shared import (
    PLOTLY_CONFIG,
    download_frame,
    explained_settings,
    note,
    show_warnings,
)
from biosim_lab.core.materials import CELL_TYPES, get_cell
from biosim_lab.core.viz.curves import (
    cross_section_figure,
    cumulative_count_figure,
    force_profile_figure,
    live_view_figure,
    outlet_histogram_figure,
    size_distribution_figure,
    trajectory_figure,
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

        st.subheader("Environment")
        temperature_c = st.slider(
            "Temperature (°C)", 4.0, 45.0, 25.0, 0.5,
            help="Changes viscosity by tens of percent, and migration speed is "
                 "inversely proportional to it. 25 °C is a bench, 37 °C an incubator.",
        )
        rf_power = st.slider(
            "Applied RF power (W)", 0.0, 2.0, 0.0, 0.05,
            help="Used only for the flagged transducer-heating estimate. 0 omits it.",
        )

        st.subheader("Sample and outlets")
        inlet_viability = st.slider(
            "Viability of the incoming sample", 0.50, 1.0, 0.95, 0.01,
            help="A freshly prepared suspension is typically 90-97 % viable, "
                 "so the honest baseline is not 100 %.",
        )
        inlet = st.selectbox(
            "Inlet focusing", ["sheath_sides", "uniform", "centre"], index=0
        )
        collection_fraction = st.slider(
            "Collection outlet width (fraction of channel)", 0.05, 0.9, 1 / 3, 0.01
        )
        n_cells = st.slider("Cells per population", 50, 600, 300, 50)

        st.subheader("Uncertainty")
        n_replicates = st.slider(
            "Replicates", 1, 12, 1, 1,
            help="Re-run on freshly drawn cells. 1 reports only the counting "
                 "error; more also measures sample-to-sample spread.",
        )

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

    with explained_settings():
        out = run_sorter(
            frequency_mhz, voltage_pp, flow_ul_min, width_um, height_um, length_mm,
            n_cells, collection_fraction, inlet, mode, target, background,
            fem_resolution, temperature_c, inlet_viability, rf_power, int(seed),
        )
    m, d = out["metrics"], out["diagnostics"]

    cols = st.columns(6)
    cols[0].metric(
        "Recovery", f"{m['efficiency_percent']:.1f} %",
        help="Fraction of target cells that reached the collection outlet. "
             f"95 % counting interval [{m['efficiency_ci_low']:.1f}, "
             f"{m['efficiency_ci_high']:.1f}] — binomial, from the finite number "
             "of cells simulated.",
    )
    cols[1].metric(
        "Purity", f"{m['purity_percent']:.1f} %",
        help="Fraction of the collected cells that are targets. "
             f"95 % counting interval [{m['purity_ci_low']:.1f}, "
             f"{m['purity_ci_high']:.1f}].",
    )
    cols[2].metric("Live purity", f"{m['live_purity_percent']:.1f} %",
                   help="of the LIVE cells collected, the fraction that are targets — "
                        "a collected dead cell is of no use downstream")
    cols[3].metric("Enrichment", f"{m['enrichment_fold']:.2f}×",
                   help="how much richer the output is than the input")
    viability_delta = m["viability_out_percent"] - m["viability_in_percent"]
    cols[4].metric("Viability out", f"{m['viability_out_percent']:.1f} %",
                   delta=f"{viability_delta:+.1f} pt through the device")
    cols[5].metric("Collected", f"{m['n_collected']}",
                   help=f"{m['n_alive_collected']} of them alive")

    show_warnings(out["warnings"])

    if not m["all_cells_exited"]:
        st.error(
            "Some cells never reached the outlet in the time simulated. The "
            "metrics above are computed at their last position, so treat them as "
            "unreliable — lower the flow rate or shorten the channel.",
            icon="🚫",
        )

    nodes = d["node_positions_m"]
    tb = d["thermal_budget"]
    st.markdown(
        f"**{d['temperature_C']:.1f} °C** &nbsp;·&nbsp; "
        f"viscosity **{d['viscosity_Pa_s'] * 1e3:.3f} mPa·s** &nbsp;·&nbsp; "
        f"sound speed **{d['sound_speed_m_s']:.0f} m/s** &nbsp;·&nbsp; "
        f"heating **+{tb['total_rise_K']:.3f} K** "
        f"(water absorbs {tb['bulk_absorption_K']:.4f} K of it)"
    )
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
        ["Live view", "Cross-section", "Live count", "Cell safety", "Trajectories",
         "Outlet histogram", "Force profile", "Size distribution", "Per-population",
         "Uncertainty", "Data"]
    )
    half = 0.5 * collection_fraction * width_um * 1e-6
    bounds = (out["node_offset"] - half, out["node_offset"] + half)

    with tabs[0]:
        st.plotly_chart(
            live_view_figure(
                out["trajectories"], channel_width=width_um * 1e-6,
                channel_length=length_mm * 1e-3, node_positions=nodes,
                alive=out["alive"], collection_bounds=bounds,
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        note(
            "Every dot is one cell, seen from above, moving left to right down the "
            "channel. Marker size follows the real radius; hollow grey markers are "
            "dead cells. Press <b>Play</b>, or drag the time slider to step through "
            "the run. This is the view a microscope over the chip would give you."
        )

    with tabs[1]:
        st.plotly_chart(
            cross_section_figure(
                out["trajectories"], channel_width=width_um * 1e-6,
                channel_height=height_um * 1e-6, channel_length=length_mm * 1e-3,
                alive=out["alive"], node_positions=nodes,
            ),
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "The channel sliced across, at the moment the cells reach the outlet. "
            "The dots are drawn <b>to scale</b> — that visible size difference is "
            "the entire basis of the separation, because the acoustic force grows "
            "with the cube of the radius while the drag grows only with the radius."
        )

    with tabs[2]:
        st.plotly_chart(
            cumulative_count_figure(
                out["trajectories"], out["cells"],
                channel_length=length_mm * 1e-3, collection_bounds=bounds,
            ),
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "A running tally at the outlet, as an instrument's counter would show "
            "it. The slope is throughput in cells per second. Focused target cells "
            "sit in the fast centre of the flow and arrive in a tight burst; the "
            "background is spread across the channel, including the slow region "
            "near the walls, so it dribbles out over a much longer window."
        )

    with tabs[3]:
        _cell_safety_panel(m, d)

    with tabs[4]:
        st.plotly_chart(
            trajectory_figure(
                out["trajectories"], node_positions=nodes,
                channel_width=width_um * 1e-6, channel_length=length_mm * 1e-3,
                title="Cell paths through the acoustic field",
            ),
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "Each line is one cell, seen from above. The dotted line is the "
            "pressure node. A working device separates the two colours before the "
            "right-hand edge of the plot, which is the end of the channel."
        )

    with tabs[5]:
        st.plotly_chart(
            outlet_histogram_figure(
                out["cells"], channel_width=width_um * 1e-6,
                collection_bounds=bounds,
            ),
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "Where each cell is when it leaves. The shaded band is the collection "
            "outlet; overlap inside it is what limits purity."
        )

    with tabs[6]:
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
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "The sideways push, in piconewtons, across the channel. It is zero at "
            "the node — a cell that arrives there stops. The gap between the two "
            "curves is the whole separation, and it comes almost entirely from the "
            "cube of the radius."
        )

    with tabs[7]:
        st.plotly_chart(
            size_distribution_figure(out["cells"]),
            width="stretch", config=PLOTLY_CONFIG,
        )
        note(
            "Radii are drawn from a log-normal distribution, as real populations "
            "are. Because force scales with r³, this spread is the main reason "
            "purity is never 100 %."
        )

    with tabs[8]:
        rows = [{"population": k, **v} for k, v in m["per_population"].items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("**Acoustic contrast factors**")
        contrast = pd.DataFrame(
            [
                {"population": k, "Φ (classical)": v["phi_classical"],
                 "Φ effective (this device)": v["phi_effective_ssaw"]}
                for k, v in d["contrast_factors"].items()
            ]
        )
        st.dataframe(contrast, width="stretch", hide_index=True)
        note(
            "Φ > 0 means the cell moves to the quiet node; Φ < 0 would send it to "
            "the loud antinode. The effective value is lower because in a surface-"
            "wave device the sound does not travel straight across the channel."
        )

    with tabs[9]:
        _uncertainty_panel(
            m, n_replicates, frequency_mhz, voltage_pp, flow_ul_min, width_um,
            height_um, length_mm, n_cells, collection_fraction, inlet, mode,
            target, background, fem_resolution, temperature_c, inlet_viability,
            rf_power, int(seed),
        )

    with tabs[10]:
        st.dataframe(out["cells"].head(200), width="stretch", hide_index=True)
        _download_row(out)
        note(
            "The YAML runs unchanged on the command line: "
            "<code>biosim run saw_sorter_config.yaml</code>. The NetCDF holds the "
            "full trajectory array, which the CSV cannot: it is "
            "<code>position(particle, time, axis)</code>, readable with "
            "<code>xarray.open_dataset</code>."
        )


def _format_margin(value: float) -> str:
    """Safety margins span twelve orders of magnitude; don't print all of them."""
    if not np.isfinite(value):
        return "∞"
    if value >= 1e4:
        return "> 10 000×"
    if value >= 10:
        return f"{value:,.0f}×"
    return f"{value:.1f}×"


def _cell_safety_panel(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    """Are the cells harmed by the device? Three mechanisms, each with a margin."""
    st.markdown("#### Is the device gentle?")
    note(
        "<b>Gentle</b> is a claim, so it is computed. Each row is a published damage "
        "threshold and how far the current operating point sits from it. A margin "
        "below 1 means the threshold has been crossed."
    )

    rows = [
        {
            "mechanism": "Heat",
            "indicator": f"{metrics['thermal_dose_cem43_max']:.3g} CEM43 min",
            "threshold": f"{metrics['thermal_dose_threshold_cem43']:.0f} CEM43 min",
            "margin": metrics["thermal_margin"],
            "reference": "Sapareto & Dewey, doi:10.1016/0360-3016(84)90379-1",
        },
        {
            "mechanism": "Shear",
            "indicator": f"{metrics['peak_shear_on_a_cell_Pa']:.2f} Pa",
            "threshold": f"{metrics['shear_lysis_threshold_Pa']:.0f} Pa",
            "margin": metrics["shear_margin"],
            "reference": "Leverett et al., doi:10.1016/S0006-3495(72)86085-5",
        },
        {
            "mechanism": "Cavitation",
            "indicator": f"MI = {metrics['mechanical_index']:.3f}",
            "threshold": f"MI = {metrics['mechanical_index_limit']:.1f}",
            "margin": metrics["cavitation_margin"],
            "reference": "Apfel & Holland, doi:10.1016/0301-5629(91)90125-Q",
        },
    ]
    table = pd.DataFrame(rows)
    table["margin"] = table["margin"].map(_format_margin)
    st.dataframe(table, width="stretch", hide_index=True)

    worst = min(
        metrics["thermal_margin"], metrics["shear_margin"], metrics["cavitation_margin"]
    )
    if worst < 1:
        st.error(
            "At least one damage threshold has been exceeded. The viability figures "
            "above are the model's estimate of the consequence, but a device run "
            "here would be doing real harm.", icon="🚫",
        )
    elif worst < 3:
        st.warning(
            f"The narrowest safety margin is {worst:.1f}×. That is not comfortable — "
            "small errors in the assumed pressure calibration could cross it.",
            icon="⚠️",
        )
    else:
        st.success(
            f"The narrowest safety margin is {_format_margin(worst)}. The dominant reason is "
            "exposure time: cells are in the field for "
            f"{diagnostics['transit_time_s']:.2f} s. A trap that held them for "
            "minutes at the same intensity would be a different proposition.",
            icon="✅",
        )

    cols = st.columns(3)
    cols[0].metric("Acoustic intensity", f"{metrics['acoustic_intensity_W_cm2']:.2f} W/cm²")
    tb = diagnostics["thermal_budget"]
    cols[1].metric("Heating from absorption", f"+{tb['bulk_absorption_K']:.4f} K",
                   help="computed from first principles — the water barely absorbs")
    cols[2].metric("Heating from the transducer", f"+{tb['transducer_K']:.2f} K",
                   help="ESTIMATED from an assumed coefficient; this project cannot "
                        "compute it, and in a real chip it is the dominant term")

    st.info(
        "**What is not modelled.** Membrane poration below the lysis threshold, "
        "which can let trypan blue in without killing the cell and therefore "
        "corrupts a viability readout; and any change in a cell's acoustic "
        "properties once it dies — dead cells are propagated with live-cell "
        "density and compressibility, so their predicted destination is less "
        "trustworthy than that of live ones.",
        icon="🔬",
    )



def _uncertainty_panel(
    metrics: dict[str, Any], n_replicates: int, frequency_mhz: float,
    voltage_pp: float, flow_ul_min: float, width_um: float, height_um: float,
    length_mm: float, n_cells: int, collection_fraction: float, inlet: str,
    mode: str, target: str, background: str, fem_resolution: int,
    temperature_c: float, inlet_viability: float, rf_power: float, seed: int,
) -> None:
    """Two different uncertainties, side by side, because they are not the same."""
    st.markdown("#### How much should you trust these numbers?")
    note(
        "A simulated result has two independent uncertainties, and quoting "
        "either one alone is misleading in opposite directions."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**1 · Counting error, within this run**")
        st.caption(
            "Even a perfectly deterministic device sorts a finite number of "
            "cells, so the measured proportion is not the population "
            "proportion. Present even with the seed fixed."
        )
        st.dataframe(
            pd.DataFrame([
                {"metric": "recovery",
                 "value": f"{metrics['efficiency_percent']:.1f} %",
                 "95 % interval": f"[{metrics['efficiency_ci_low']:.1f}, "
                                  f"{metrics['efficiency_ci_high']:.1f}]"},
                {"metric": "purity",
                 "value": f"{metrics['purity_percent']:.1f} %",
                 "95 % interval": f"[{metrics['purity_ci_low']:.1f}, "
                                  f"{metrics['purity_ci_high']:.1f}]"},
            ]),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Wilson score interval, not the textbook `p ± z√(p(1−p)/n)` — that "
            "one gives a width of **exactly zero** at 100 %, which a working "
            "sorter reaches constantly."
        )

    with right:
        st.markdown("**2 · Sample-to-sample error, across runs**")
        st.caption(
            "Each replicate draws fresh radii from the log-normal size "
            "distribution, fresh inlet positions and fresh viability outcomes. "
            "This is how much of the answer is the device and how much is the "
            "particular batch of cells."
        )
        if n_replicates < 2:
            st.info(
                "Set **Replicates** above 1 in the sidebar to measure this.",
                icon="🎲",
            )
        else:
            table = run_replicates(
                n_replicates, frequency_mhz=frequency_mhz, voltage_pp=voltage_pp,
                flow_ul_min=flow_ul_min, width_um=width_um, height_um=height_um,
                length_mm=length_mm, n_cells=n_cells,
                collection_fraction=collection_fraction, inlet=inlet, mode=mode,
                target=target, background=background,
                fem_resolution=fem_resolution, temperature_c=temperature_c,
                inlet_viability=inlet_viability, rf_power=rf_power, seed=seed,
            )
            st.dataframe(table.round(3), width="stretch", hide_index=True)
            download_frame(table, "replicates.csv", "Download (CSV)")

    st.info(
        "**Which do you quote?** If you are comparing two designs on the same "
        "simulated sample, the counting error is the relevant one. If you are "
        "predicting what a real experiment will measure, you need both, and the "
        "real experiment has a third source this model does not cover at all — "
        "device-to-device variation in fabrication.",
        icon="📐",
    )


def _download_row(out: dict[str, Any]) -> None:
    """Offer every export format the deployment actually has a writer for.

    Which buttons appear depends on what is installed: NetCDF needs an HDF5
    stack and Parquet needs pyarrow, and both are optional. Rather than failing
    at click time, the buttons are only built when the writer is importable, and
    the Environment page explains any that are absent.
    """
    from biosim_lab.core.plugin import optional_import

    cols = st.columns(4)
    with cols[0]:
        st.download_button(
            "Cells (CSV)", out["cells"].to_csv(index=False).encode("utf-8"),
            file_name="saw_sorter_cells.csv", mime="text/csv",
        )
    with cols[1]:
        st.download_button(
            "Configuration (YAML)", _params_to_yaml(out["params"]).encode("utf-8"),
            file_name="saw_sorter_config.yaml", mime="text/yaml",
        )
    with cols[2]:
        if optional_import("pyarrow") is not None:
            buffer = io.BytesIO()
            out["cells"].to_parquet(buffer, index=False)
            st.download_button(
                "Cells (Parquet)", buffer.getvalue(),
                file_name="saw_sorter_cells.parquet", mime="application/octet-stream",
            )
        else:
            st.caption("Parquet needs pyarrow")
    with cols[3]:
        payload = _netcdf_bytes(out)
        if payload is not None:
            st.download_button(
                "Trajectories (NetCDF)", payload,
                file_name="saw_sorter_trajectories.nc",
                mime="application/x-netcdf",
            )
        else:
            st.caption("NetCDF needs h5netcdf or netCDF4")


def _netcdf_bytes(out: dict[str, Any]) -> bytes | None:
    """Serialise the trajectory Dataset to NetCDF in memory, or None if unavailable."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    try:
        from biosim_lab.core.io import save_dataset
    except Exception:  # noqa: BLE001
        return None
    dataset = out["trajectories"].copy()
    dataset.attrs["biosim_metrics"] = json.dumps(out["metrics"], default=str)
    dataset.attrs["biosim_params"] = json.dumps(out["params"], default=str)
    try:
        with TemporaryDirectory() as tmp:
            path = save_dataset(dataset, Path(tmp) / "trajectories.nc")
            return path.read_bytes()
    except Exception:  # noqa: BLE001 - no writer installed, or a driver problem
        return None


def _params_to_yaml(params: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(
        {"name": "from_streamlit", "instrument": "saw_sorter", "params": params},
        sort_keys=False, allow_unicode=True,
    )


