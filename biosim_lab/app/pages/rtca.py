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
                grid, float(grouped.min()), float(grouped.max()),
                m["ic50"], m["hill_slope"],
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


