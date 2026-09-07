"""Example 5 — impedance-based real-time cell analysis (Stage 2).

Simulates a 96-well impedance plate: cells attach and proliferate, a drug is
added at 24 h, and the endpoint Cell Index gives a dose-response curve from
which an IC50 is fitted.  Also shows the |Z|(f) spectrum of a blank versus a
confluent electrode, which is what the Giaever-Keese model actually predicts.

Run::

    python examples/05_impedance_rtca.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.io import save_result
from biosim_lab.core.viz.curves import (
    bode_magnitude_figure,
    bode_phase_figure,
    dose_response_figure,
    nyquist_figure,
    timeseries_figure,
    well_plate_heatmap,
)
from biosim_lab.instruments.impedance_rtca import ImpedanceRTCA
from biosim_lab.instruments.impedance_rtca.physics import four_parameter_logistic

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    cfg = ExperimentConfig.model_validate(ImpedanceRTCA.example_config())
    instrument = ImpedanceRTCA(cfg)
    result = instrument.run()
    m = result.metrics

    print("=" * 72)
    print("Real-time cell impedance analysis — Giaever-Keese model")
    print("=" * 72)
    print(f"  wells                 {m['n_wells']:8d}")
    print(f"  duration              {m['duration_h']:8.1f} h")
    print(f"  readout frequency     {m['readout_frequency_Hz'] / 1e3:8.1f} kHz")
    print(f"  peak Cell Index       {m['max_cell_index']:8.2f}")
    print()
    print(f"  fitted IC50           {m['ic50']:8.4f} +/- {m['ic50_stderr']:.4f}")
    print(f"  Hill slope            {m['hill_slope']:8.3f}")
    print(f"  fit R^2               {m['fit_r_squared']:8.4f}")
    print(f"  planted IC50          {1.0:8.4f}  (recovery "
          f"{m['ic50_recovery_ratio']:.2f} x)")
    print()
    print("  Note: the apparent IC50 from a fixed endpoint depends on exposure time.")
    print("  Here the drug acts for 24 h against a ~20 h doubling time, so the")
    print("  population has not fully relaxed to the reduced carrying capacity and")
    print("  the fit reads below the planted value. That is the behaviour of a real")
    print("  endpoint assay, not a numerical artefact.")

    # ---- figures ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = result.fields
    table = result.table

    import pandas as pd

    wells = [str(w) for w in fields["well"].values]
    ci = fields["cell_index"].values
    curves = pd.DataFrame({"time": fields["time"].values})
    for dose, group in table.groupby("concentration"):
        idx = [wells.index(w) for w in group["well"] if w in wells]
        if idx:
            curves["control" if dose == 0 else f"{dose:g}"] = ci[:, idx].mean(axis=1)

    freq = fields["frequency"].values
    z_real = fields["impedance_real"].values
    z_imag = fields["impedance_imag"].values
    z_blank = z_real[:, 0] + 1j * z_imag[:, 0]
    z_full = z_real[:, -1] + 1j * z_imag[:, -1]

    grid = np.logspace(-2.5, 2, 200)
    fitted = four_parameter_logistic(
        grid,
        float(table["normalised_response"].min()),
        float(table["normalised_response"].max()),
        m["ic50"], m["hill_slope"],
    )
    treated = table[(table["kind"] == "treated") & (table["concentration"] > 0)]
    grouped = treated.groupby("concentration")["normalised_response"].mean()

    figures = {
        "05_cell_index": timeseries_figure(
            curves, title="Cell Index over time (mean of replicates, by dose)",
            yaxis_title="Cell Index (dimensionless)",
        ),
        "05_nyquist": nyquist_figure(freq, z_full,
                                     title="Nyquist — confluent monolayer"),
        "05_bode_magnitude": bode_magnitude_figure(
            freq, z_blank, title="Impedance magnitude — blank electrode"),
        "05_bode_phase": bode_phase_figure(freq, z_blank,
                                           title="Impedance phase — blank electrode"),
        "05_dose_response": dose_response_figure(
            grouped.index.to_numpy(), grouped.to_numpy(), fit=(grid, fitted),
            ic50=m["ic50"], title="Endpoint dose-response with 4PL fit",
        ),
        "05_plate_map": well_plate_heatmap(
            dict(zip(table["well"], table["endpoint_cell_index"])),
            title="Endpoint Cell Index by well", colorbar_title="CI",
        ),
    }
    png_ok = True
    for name, fig in figures.items():
        fig.write_html(OUT_DIR / f"{name}.html", include_plotlyjs="cdn")
        print(f"  wrote {OUT_DIR / f'{name}.html'}")
        if png_ok:
            try:
                fig.write_image(OUT_DIR / f"{name}.png", width=1000, height=500, scale=2)
            except Exception as exc:  # noqa: BLE001
                print(f"  (PNG export unavailable: {type(exc).__name__})")
                png_ok = False

    written = save_result(result, OUT_DIR, "05_impedance_rtca")
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")


if __name__ == "__main__":
    main()
