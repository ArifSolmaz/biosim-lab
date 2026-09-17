"""The rtca page."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from biosim_lab.app.runners import (
    run_rtca,
)
from biosim_lab.app.shared import (
    PLOTLY_CONFIG,
    download_frame,
    explained_settings,
    note,
)
from biosim_lab.core.viz.curves import (
    bode_magnitude_figure,
    bode_phase_figure,
    dose_response_figure,
    nyquist_figure,
    timeseries_figure,
    well_plate_heatmap,
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
        electrode = st.selectbox(
            "Electrode", ["interdigitated", "disc"],
            help="interdigitated: two equal gold combs, as on an RTCA plate. "
            "disc: the classic ECIS working disc against a large counter electrode.",
        )
        # Streamlit keeps a disabled widget's value, so a box ticked under the
        # interdigitated electrode would stay ticked (and look active) after a
        # switch to the disc. Only offer it where it means something.
        # Same control as the sorter's, so the choice between a closed form and
        # a field solve is one idiom across the app rather than two.
        if electrode == "interdigitated":
            field_model = st.radio(
                "Electrode field", ["analytic", "fem"], horizontal=True,
                key="rtca_field_model",
                help="analytic = the lumped series model, which assumes the current "
                "crosses every finger uniformly; instant. fem = electro-quasistatic "
                "Poisson on the finger pattern, which resolves the crowding at the "
                "finger edges: 221 solves, cached per parameter set, so the first view "
                "costs seconds and a return visit is instant. It changes the spectra "
                "above ~50 kHz, not the Cell Index.",
            )
            fem = field_model == "fem"
        else:
            fem = False
            st.caption(
                "FEM applies to the interdigitated electrode. A disc against a large "
                "counter electrode has a closed-form spreading resistance already."
            )
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

    note(
        "The electrode geometry is an <b>assumption</b>: a generic 50/50 µm gold "
        "interdigitated comb on a 3 × 3 mm patch. The dimensions of the commercial "
        "E-Plate electrodes are not published, so the shape of these curves is right "
        "and the absolute impedance is only as right as that guess. Measure your own "
        "chip before comparing ohms."
    )

    uploaded = st.file_uploader(
        "Optional: upload a real RTCA export (CSV or XLSX) to analyse instead",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is not None:
        # The sidebar still shows the simulation's controls, and they now drive
        # nothing at all. Saying so is cheaper than a user concluding the drug
        # model is broken because moving IC50 changes no curve.
        with st.sidebar:
            st.info(
                "Reading the uploaded file. The simulation controls above are "
                "inactive until you remove it (press × on the file).",
                icon="📄",
            )
        _rtca_from_upload(uploaded)
        return

    with explained_settings():
        out = run_rtca(
            frequency_khz, duration_h, treatment_h, true_ic50, hill, replicates,
            noise_cv, doubling_h, int(seed), electrode, bool(fem),
        )
    m, table = out["metrics"], out["table"]

    cols = st.columns(5)
    cols[0].metric("Fitted IC50", f"{m.get('ic50', float('nan')):.3g}",
                   delta=f"{m.get('ic50', float('nan')) / true_ic50:.2f}× the planted value")
    cols[1].metric("Standard error", f"{m.get('ic50_stderr', float('nan')):.2g}")
    cols[2].metric("Hill slope", f"{m.get('hill_slope', float('nan')):.2f}")
    cols[3].metric("Fit R²", f"{m.get('fit_r_squared', float('nan')):.4f}")
    cols[4].metric("Peak Cell Index", f"{m['max_cell_index']:.2f}")

    if m.get("field_model") == "fem":
        st.caption(
            f"Electrode solved by FEM ({m['fem_mesh_nodes']} nodes). At the readout "
            f"frequency the lumped model is within "
            f"{m['readout_lumped_vs_fem_max_percent']:.2f} % of it; across the spectrum "
            f"it is off by up to {m['spectrum_lumped_vs_fem_max_percent']:.1f} % "
            f"(worst near {m['spectrum_worst_frequency_Hz'] / 1e3:.0f} kHz), where "
            "current crowds onto the finger edges."
        )

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

    names = ["Cell Index", "Dose–response", "Nyquist", "Bode", "Plate map", "Data"]
    if out["potential"] is not None:
        names.append("Electrode field")
    tabs = st.tabs(names)

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
            width="stretch", config=PLOTLY_CONFIG,
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
                grid, m.get("fit_bottom", float(grouped.min())),
                m.get("fit_top", float(grouped.max())), m["ic50"], m["hill_slope"],
            )
            st.plotly_chart(
                dose_response_figure(
                    grouped.index.to_numpy(), grouped.to_numpy(),
                    fit=(grid, fitted), ic50=m["ic50"],
                ),
                width="stretch", config=PLOTLY_CONFIG,
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
            width="stretch", config=PLOTLY_CONFIG,
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
                width="stretch", config=PLOTLY_CONFIG,
            )
        with right:
            st.plotly_chart(
                bode_phase_figure(out["frequency"], z[:, 0],
                                  title="Phase — bare electrode"),
                width="stretch", config=PLOTLY_CONFIG,
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
            width="stretch", config=PLOTLY_CONFIG,
        )
        note("Column 1 holds the untreated controls; dose increases left to right.")

    with tabs[5]:
        st.dataframe(table, width="stretch", hide_index=True)
        download_frame(table, "rtca_wells.csv", "Download well table (CSV)")

    if out["potential"] is not None:
        with tabs[6]:
            _electrode_field(out["potential"], m)


def _electrode_field(field: dict, metrics: dict) -> None:
    """The FEM's potential map — the picture behind the lumped model's error.

    The solve already produced this on the way to the impedance; showing it is
    the difference between asserting that current crowds at the finger edges
    and letting the reader see it.
    """
    import plotly.graph_objects as go

    from biosim_lab.core.viz.theme import plotly_layout, sequential_colorscale

    x_um = np.asarray(field["x"]) * 1e6
    y_um = np.asarray(field["y"]) * 1e6
    fig = go.Figure(go.Heatmap(
        z=np.asarray(field["phi"]).T, x=x_um, y=y_um,
        colorscale=sequential_colorscale(), colorbar={"title": "|φ| (V)"},
        hovertemplate="x %{x:.1f} µm<br>y %{y:.1f} µm<br>|φ| %{z:.3f} V<extra></extra>",
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(**plotly_layout(
        "Potential in the electrolyte over one IDE unit cell",
        xaxis_title="across the fingers (µm)", yaxis_title="height above the chip (µm)",
        showlegend=False,
    ))
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    note(
        f"One symmetry cell of the comb, driven as: {field['drive']}. The whole "
        "electrode is many of these in parallel. Where the contours crowd against "
        "the metal edge, so does the current — which is why the lumped series "
        "model, which assumes the current crosses every finger uniformly, is off "
        f"by up to {metrics['spectrum_lumped_vs_fem_max_percent']:.1f} % across the "
        "spectrum while staying within "
        f"{metrics['readout_lumped_vs_fem_max_percent']:.2f} % at the readout "
        "frequency. Wagner number at the readout frequency: "
        f"{metrics['readout_wagner_number']:.2f} — of order one is exactly where "
        "crowding bites."
    )


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
        width="stretch", config=PLOTLY_CONFIG,
    )
    long = pd.DataFrame(ci, columns=wells)
    long.insert(0, "time_s", ds["time"].values)
    download_frame(long, "rtca_uploaded.csv", "Download parsed data (CSV)")


