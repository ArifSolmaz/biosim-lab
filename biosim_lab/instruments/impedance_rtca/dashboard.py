"""Panel dashboard for the RTCA instrument."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from biosim_lab.core.viz.curves import (
    bode_magnitude_figure,
    bode_phase_figure,
    dose_response_figure,
    field_heatmap_figure,
    nyquist_figure,
    timeseries_figure,
    well_plate_heatmap,
)
from biosim_lab.core.viz.dashboard import data_table, metric_tiles, require_panel, shell
from biosim_lab.core.viz.theme import color_for, plotly_layout
from biosim_lab.instruments.impedance_rtca.physics import four_parameter_logistic

_TILE_FORMATS = {
    "ic50": "{:.3g}",
    "ic50_stderr": "{:.2g}",
    "hill_slope": "{:.2f}",
    "fit_r_squared": "{:.4f}",
    "max_cell_index": "{:.2f}",
    "n_wells": "{:d}",
    "blank_impedance_ohm": "{:.1f} Ω",
    "solution_resistance_ohm": "{:.1f} Ω",
    "cell_constant_1_per_m": "{:.3g} /m",
    "readout_lumped_vs_fem_max_percent": "{:.2f} %",
    "spectrum_lumped_vs_fem_max_percent": "{:.1f} %",
}


def lumped_error_figure(frequency: np.ndarray, fem: np.ndarray, lumped: np.ndarray,
                        coverage: list[float]) -> go.Figure:
    """How far the lumped series model is from the FEM, per coverage, against frequency.

    One axis, one quantity (percent deviation of ``|Z|``): the useful fact is
    *where* the two disagree --- at the Wagner-number crossover, where current
    starts crowding onto the finger edges --- and a log |Z| overlay hides that.
    """
    fig = go.Figure()
    for j, c in enumerate(coverage):
        dev = 100.0 * (np.abs(lumped[:, j]) - np.abs(fem[:, j])) / np.abs(fem[:, j])
        fig.add_trace(go.Scatter(
            x=frequency, y=dev, mode="lines", name=f"coverage {c:g}",
            line={"color": color_for(("background", "target", "wbc")[j % 3]), "width": 2},
            hovertemplate="f %{x:.3g} Hz<br>%{y:.2f} %<extra></extra>",
        ))
    fig.add_hline(y=0.0, line={"color": "#52514e", "width": 1, "dash": "dot"})
    fig.update_xaxes(type="log")
    fig.update_layout(**plotly_layout(
        title="Lumped model against the FEM (|Z|, %)", xaxis_title="frequency (Hz)",
        yaxis_title="lumped − FEM (%)"))
    return fig


def build_dashboard(instrument: Any) -> Any:
    """Assemble the RTCA dashboard from a run instrument."""
    pn = require_panel()
    result = instrument.results()
    fields = result.fields
    table: pd.DataFrame = result.table

    # -- Cell Index time courses, one line per dose (replicates averaged)
    time_h = fields["time"].values
    ci = fields["cell_index"].values
    wells = [str(w) for w in fields["well"].values]

    curves = pd.DataFrame({"time": time_h})
    if "concentration" in table.columns and table["concentration"].notna().any():
        for dose, group in table.groupby("concentration"):
            idx = [wells.index(w) for w in group["well"] if w in wells]
            if idx:
                label = "control" if dose == 0 else f"{dose:g}"
                curves[label] = ci[:, idx].mean(axis=1)
    else:
        for j, well in enumerate(wells[:12]):
            curves[well] = ci[:, j]

    ci_fig = timeseries_figure(
        curves,
        title="Cell Index over time",
        yaxis_title="Cell Index (dimensionless)",
    )

    tabs: list[tuple[str, Any]] = [("Cell Index", pn.pane.Plotly(
        ci_fig, config={"displayModeBar": False}, sizing_mode="stretch_width", height=440
    ))]

    if "normalized_cell_index" in fields:
        nci = fields["normalized_cell_index"].values
        norm = pd.DataFrame({"time": time_h})
        for col in curves.columns:
            if col == "time":
                continue
            if col in wells:
                norm[col] = nci[:, wells.index(col)]
            else:
                dose = 0.0 if col == "control" else float(col)
                group = table[table["concentration"] == dose]
                idx = [wells.index(w) for w in group["well"] if w in wells]
                norm[col] = np.nanmean(nci[:, idx], axis=1)
        tabs.append(("Normalised CI", pn.pane.Plotly(
            timeseries_figure(norm, title="Cell Index normalised at the last point "
                              "before treatment", yaxis_title="normalised Cell Index"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))

    # -- spectra
    if "impedance_real" in fields:
        freq = fields["frequency"].values
        z = (fields["impedance_real"].values + 1j * fields["impedance_imag"].values)
        confluent = z[:, -1]
        tabs.append(("Nyquist", pn.pane.Plotly(
            nyquist_figure(freq, confluent, title="Nyquist — confluent monolayer"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))
        tabs.append(("Bode |Z|", pn.pane.Plotly(
            bode_magnitude_figure(freq, confluent),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))
        tabs.append(("Bode phase", pn.pane.Plotly(
            bode_phase_figure(freq, confluent),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))
        if "impedance_lumped_real" in fields:
            lumped = fields["impedance_lumped_real"].values + 1j * fields[
                "impedance_lumped_imag"].values
            tabs.append(("FEM vs lumped", pn.pane.Plotly(
                lumped_error_figure(freq, z, lumped,
                                    [float(c) for c in fields["coverage"].values]),
                config={"displayModeBar": False}, sizing_mode="stretch_width",
                height=440)))
    potential = getattr(instrument, "_potential_map", None)
    if potential is not None:
        tabs.append(("IDE potential", pn.pane.Plotly(
            field_heatmap_figure(potential, "phi_abs",
                                 title="|φ| in the IDE unit cell (1 V on the left comb)",
                                 colorbar_title="V"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))

    # -- plate heat map of the endpoint Cell Index
    if "endpoint_cell_index" in table.columns:
        values = dict(zip(table["well"], table["endpoint_cell_index"]))
        rows, cols = (8, 12) if len(values) <= 96 else (16, 24)
        tabs.append(("Plate map", pn.pane.Plotly(
            well_plate_heatmap(values, rows=rows, cols=cols,
                               title="Endpoint Cell Index by well",
                               colorbar_title="CI"),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))

    # -- dose-response
    metrics = result.metrics
    if "ic50" in metrics and "concentration" in table.columns:
        treated = table[(table["kind"] == "treated") & (table["concentration"] > 0)]
        grouped = treated.groupby("concentration")["normalised_response"].mean()
        c = grouped.index.to_numpy()
        grid = np.logspace(np.log10(c.min()) - 0.5, np.log10(c.max()) + 0.5, 200)
        fitted = four_parameter_logistic(
            grid,
            metrics.get("fit_bottom", float(np.min(grouped))),
            metrics.get("fit_top", float(np.max(grouped))),
            metrics["ic50"],
            metrics["hill_slope"],
        )
        tabs.append(("Dose-response", pn.pane.Plotly(
            dose_response_figure(c, grouped.to_numpy(), fit=(grid, fitted),
                                 ic50=metrics["ic50"]),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))

    tabs.append(("Well table", data_table(table, height=360)))

    tile_keys = ["ic50", "ic50_stderr", "hill_slope", "fit_r_squared", "max_cell_index",
                 "n_wells"]
    tile_keys += [k for k in ("blank_impedance_ohm", "solution_resistance_ohm",
                              "readout_lumped_vs_fem_max_percent",
                              "spectrum_lumped_vs_fem_max_percent") if k in metrics]
    electrode = metrics.get("electrode", "measured")
    return shell(
        "Real-time cell impedance analyser",
        f"Giaever-Keese cell layer · {electrode} electrode"
        + (" · electro-quasistatic FEM" if metrics.get("field_model") == "fem" else "")
        + " · Cell Index · |Z|(f) spectra · 4PL IC50 fit",
        tiles=metric_tiles(metrics, tile_keys, _TILE_FORMATS),
        tabs=tabs,
        footer="Cell layer: Giaever & Keese, doi:10.1073/pnas.88.17.7896 · IDE cell constant: "
        "Olthuis et al., doi:10.1016/0925-4005(95)85053-8 · IC50 fit: 4PL, doi:10.1002/pst.426",
    )


__all__ = ["build_dashboard"]
