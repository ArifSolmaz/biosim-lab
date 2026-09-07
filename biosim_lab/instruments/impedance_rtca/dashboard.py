"""Panel dashboard for the RTCA instrument."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.viz.curves import (
    bode_magnitude_figure,
    bode_phase_figure,
    dose_response_figure,
    nyquist_figure,
    timeseries_figure,
    well_plate_heatmap,
)
from biosim_lab.core.viz.dashboard import data_table, metric_tiles, require_panel, shell
from biosim_lab.instruments.impedance_rtca.physics import four_parameter_logistic

_TILE_FORMATS = {
    "ic50": "{:.3g}",
    "ic50_stderr": "{:.2g}",
    "hill_slope": "{:.2f}",
    "fit_r_squared": "{:.4f}",
    "max_cell_index": "{:.2f}",
    "n_wells": "{:d}",
}


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
            float(np.min(grouped)),
            float(np.max(grouped)),
            metrics["ic50"],
            metrics["hill_slope"],
        )
        tabs.append(("Dose-response", pn.pane.Plotly(
            dose_response_figure(c, grouped.to_numpy(), fit=(grid, fitted),
                                 ic50=metrics["ic50"]),
            config={"displayModeBar": False}, sizing_mode="stretch_width", height=440)))

    tabs.append(("Well table", data_table(table, height=360)))

    return shell(
        "Real-time cell impedance analyser",
        "Giaever-Keese electrode model · Cell Index · |Z|(f) spectra · 4PL IC50 fit",
        tiles=metric_tiles(
            metrics,
            ["ic50", "ic50_stderr", "hill_slope", "fit_r_squared", "max_cell_index",
             "n_wells"],
            _TILE_FORMATS,
        ),
        tabs=tabs,
        footer="Model: Giaever & Keese, doi:10.1073/pnas.88.17.7896 · "
        "IC50 fit: 4PL, doi:10.1002/pst.426",
    )


__all__ = ["build_dashboard"]
