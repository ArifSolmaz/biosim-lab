"""Two-outlet sorting: large cells leave on one side, everything else on the other.

The default chip in this package has three outlets and creams the large cells
off the middle, at the pressure node. Many real devices are two-outlet instead:
the sample enters along one wall, large cells cross the channel toward the node,
small ones do not, and a single divider separates the two streams.

The one number you have to choose is where that divider goes, and the useful
result is that it is **not** the node. Cells approach a node asymptotically and
settle a few microns short of it, so a divider placed on the node collects
nothing at all. It belongs between the two populations' landing positions, and
this script sweeps it so you can see the window.

Writes ``assets/09_two_outlet_split.csv`` and an outlet-histogram figure.

Run::

    python examples/09_two_outlet_split.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.viz.curves import outlet_histogram_figure
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BASE = dict(
    frequency="6.632 MHz",
    voltage_pp="15 V",
    channel_width="300 um",
    channel_height="50 um",
    channel_length="2 mm",
    flow_rate="5 uL/min",
    fluid="water",
    substrate="linbo3_128yx",
    mode="analytic",
    seed=20260907,
    # The sample enters along one wall, so every cell has the whole channel
    # width to migrate across and the size selectivity is as large as the
    # geometry allows.
    inlet="side",
    inlet_side="left",
    outlet_layout="lateral_split",
    collect_side="right",
    populations=[
        {"cell_type": "mcf7", "count": 300, "target": True},
        {"cell_type": "rbc", "count": 300, "target": False},
    ],
)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    width_um = 300.0

    rows = []
    best = None
    for fraction in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            sim = SAWSorterSimulation(SAWSorterParams(split_position=fraction, **BASE))
            outcome = sim.run()
        m = outcome.metrics
        row = {
            "divider_um": fraction * width_um,
            "recovery_percent": m["efficiency_percent"],
            "purity_percent": m["purity_percent"],
            "n_collected": m["n_collected"],
        }
        rows.append(row)
        # Best = highest purity among those that actually recover most cells.
        if m["efficiency_percent"] > 90.0 and (
            best is None or m["purity_percent"] > best["purity_percent"]
        ):
            best, best_outcome, best_sim = row, outcome, sim

    frame = pd.DataFrame(rows)
    print("Where to put the divider (sample enters on the left wall,")
    print(f"pressure node at {sim.node_offset * 1e6:.0f} um)\n")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    landing = best_outcome.metrics["per_population"]
    print("\nWhere each population lands:")
    for label, stats in landing.items():
        print(f"  {label:<6} {stats['mean_outlet_x_um']:6.1f} um "
              f"+/- {stats['std_outlet_x_um']:.1f}")

    print(
        f"\nBest divider: {best['divider_um']:.0f} um — "
        f"{best['recovery_percent']:.1f} % recovery at "
        f"{best['purity_percent']:.1f} % purity."
    )
    print(
        "Note the last row: a divider on the node collects nothing. Cells settle\n"
        "a few microns short of it, so the divider belongs between the two\n"
        "landing positions above, not on the node itself."
    )

    out = ASSETS / "09_two_outlet_split.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")

    fig = outlet_histogram_figure(
        best_outcome.cells,
        collection_bounds=best_sim.collection_bounds,
        channel_width=best_sim.params.channel_width,
    )
    fig.write_html(ASSETS / "09_two_outlet_split.html")
    print(f"wrote {ASSETS / '09_two_outlet_split.html'}")


if __name__ == "__main__":
    main()
