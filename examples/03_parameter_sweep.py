"""Example 3 — frequency x voltage x flow-rate sweep.

Maps the operating window of the sorter: where is the separation both efficient
(the tumour cells arrive) and pure (the erythrocytes do not)?  Writes the sweep
as CSV / Parquet / NetCDF plus a heatmap per metric.

Run::

    python examples/03_parameter_sweep.py            # 3 x 4 x 3 = 36 points
    python examples/03_parameter_sweep.py --quick    # 2 x 3 = 6 points
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from biosim_lab.core.io import save_dataset, save_table
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.viz.curves import sweep_heatmap_figure
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, parameter_sweep

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"

UL_MIN = 1e-9 / 60  # m^3/s per uL/min

SCALE = {"frequency": 1e-6, "voltage_pp": 1.0, "flow_rate": 1 / UL_MIN}
LABELS = {
    "frequency": "frequency (MHz)",
    "voltage_pp": "drive voltage (Vpp)",
    "flow_rate": "flow rate (µL/min)",
}


def base_params(count: int) -> SAWSorterParams:
    return SAWSorterParams(
        frequency="6.632 MHz",
        voltage_pp="15 V",
        channel_width="300 um",
        channel_height="50 um",
        channel_length="2 mm",
        flow_rate="5 uL/min",
        populations=[
            {"cell_type": "mcf7", "count": count, "target": True},
            {"cell_type": "rbc", "count": count, "target": False},
        ],
        n_time_samples=41,
        seed=20260907,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a 6-point grid")
    args = parser.parse_args()

    if args.quick:
        params = base_params(40)
        grid = {"voltage_pp": [5.0, 10.0, 20.0], "flow_rate": [5 * UL_MIN, 20 * UL_MIN]}
    else:
        params = base_params(120)
        # Frequencies chosen so the channel stays commensurate with lambda_SAW/2:
        # W = n * lambda_SAW / 2  ->  f = n * c_SAW / (2 W).
        c_saw, width = 3979.0, 300e-6
        grid = {
            "frequency": [n * c_saw / (2 * width) for n in (1, 2, 3)],
            "voltage_pp": [5.0, 10.0, 15.0, 25.0],
            "flow_rate": [5 * UL_MIN, 15 * UL_MIN, 40 * UL_MIN],
        }

    n_points = int(np.prod([len(v) for v in grid.values()]))
    print(f"sweeping {n_points} operating points: "
          + ", ".join(f"{k} ({len(v)})" for k, v in grid.items()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        df, ds = parameter_sweep(params, grid, progress=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "03_sweep.csv", index=False)
    save_table(df, OUT_DIR / "03_sweep.parquet")
    save_dataset(ds, OUT_DIR / "03_sweep.nc")
    print(f"\nwrote {OUT_DIR / '03_sweep.csv'} (+ .parquet, .nc)")

    best = df.sort_values(["purity_percent", "efficiency_percent"], ascending=False).iloc[0]
    print("\nbest operating point by purity, then efficiency:")
    for key in grid:
        print(f"  {LABELS[key]:24s} {best[key] * SCALE[key]:.3g}")
    print(f"  efficiency               {best['efficiency_percent']:.1f} %")
    print(f"  purity                   {best['purity_percent']:.1f} %")
    print(f"  enrichment               {best['enrichment_fold']:.2f} x")

    # One heatmap per remaining axis level — never an average across a third
    # parameter. Frequency changes the *number of pressure nodes*, i.e. the
    # operating regime, and averaging across regimes would report a number that
    # describes none of them.
    axes = [k for k in grid if len(grid[k]) > 1]
    if len(axes) < 2:
        return
    x_axis, y_axis = axes[-2], axes[-1]
    facet_axes = [k for k in axes if k not in (x_axis, y_axis)]

    def slices():
        if not facet_axes:
            yield "", ds
            return
        facet = facet_axes[0]
        for value in ds[facet].values:
            tag = f"_{facet}_{value * SCALE[facet]:.2f}".replace(".", "p")
            yield tag, ds.sel({facet: value})

    png_ok = True
    for tag, subset in slices():
        for metric, title in (
            ("efficiency_percent", "Recovery of MCF-7 into the collection outlet (%)"),
            ("purity_percent", "Purity of the collection outlet (%)"),
        ):
            suffix = ""
            if facet_axes:
                value = float(subset[facet_axes[0]].values)
                suffix = f" — {LABELS[facet_axes[0]]} {value * SCALE[facet_axes[0]]:.2f}"
            fig = sweep_heatmap_figure(
                subset[[metric]], metric, x=x_axis, y=y_axis,
                scale=SCALE, unit_labels=LABELS, title=title + suffix,
            )
            name = f"03_{metric}{tag}"
            fig.write_html(OUT_DIR / f"{name}.html", include_plotlyjs="cdn")
            print(f"wrote {OUT_DIR / f'{name}.html'}")
            if not png_ok:
                continue
            try:
                fig.write_image(OUT_DIR / f"{name}.png", width=1000, height=480, scale=2)
            except Exception as exc:  # noqa: BLE001 - kaleido is optional
                print(f"(PNG export unavailable: {type(exc).__name__})")
                png_ok = False


if __name__ == "__main__":
    main()
