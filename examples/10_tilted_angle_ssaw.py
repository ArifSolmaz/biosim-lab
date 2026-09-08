"""Tilted-angle SSAW: a different separation mechanism, not a different outlet.

Conventional SSAW puts the standing wave straight across the channel. The node
planes then lie parallel to the flow, and a cell migrates sideways until it
reaches a node and **stops**. Its displacement is capped by the node spacing, no
matter how long the channel or how strong the field.

Tilting the IDTs (doi:10.1073/pnas.1413325111) changes the mechanism. The node
planes are no longer parallel to the flow, so a cell held in one is dragged
across the channel as it travels downstream: ``dx/dz = -tan(theta)``. The
displacement now grows with **channel length** instead of saturating, and the
separation turns on whether a cell can be held at all rather than on how fast it
migrates. That is why the technique exists.

This script shows both: that the displacement stops saturating, and what it buys
in a geometry where a straight device struggles.

Writes ``assets/10_tilted_angle_ssaw.csv``.

Run::

    python examples/10_tilted_angle_ssaw.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BASE = dict(
    frequency="6.632 MHz",
    voltage_pp="15 V",
    channel_width="600 um",
    channel_height="50 um",
    flow_rate="5 uL/min",
    fluid="water",
    substrate="linbo3_128yx",
    mode="analytic",          # a tilted field varies along the flow, so FEM cannot hold it
    seed=20260907,
    inlet="side",
    inlet_side="left",
    outlet_layout="lateral_split",
    collect_side="right",
    populations=[
        {"cell_type": "mcf7", "count": 200, "target": True},
        {"cell_type": "rbc", "count": 200, "target": False},
    ],
)
WIDTH_UM = 600.0


def run(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return SAWSorterSimulation(SAWSorterParams(**{**BASE, **overrides})).run()


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    # --- 1. saturation versus continuous drift ---------------------------
    print("Mean outlet position of the large cells, entering at ~45 um.")
    print("A straight device parks them at a node; a tilted one keeps pushing.\n")
    print(f"{'length':>8} {'straight':>12} {'tilted -10 deg':>16} {'L*tan(10)':>12}")
    drift_rows = []
    for length_mm in (1.0, 2.0, 4.0, 8.0):
        length = f"{length_mm} mm"
        straight = run(channel_length=length, tilt_angle_deg=0.0, split_position=0.5)
        tilted = run(channel_length=length, tilt_angle_deg=-10.0, split_position=0.5)
        s_x = straight.metrics["per_population"]["mcf7"]["mean_outlet_x_um"]
        t_x = tilted.metrics["per_population"]["mcf7"]["mean_outlet_x_um"]
        geometric = length_mm * 1e3 * np.tan(np.deg2rad(10.0))
        print(f"{length:>8} {s_x:11.1f}um {t_x:15.1f}um {geometric:11.0f}um")
        drift_rows.append({"length_mm": length_mm, "straight_x_um": s_x,
                           "tilted_x_um": t_x, "geometric_drift_um": geometric})

    print(
        "\nThe straight column stops moving: the cells are on a node and that is\n"
        "as far as they go. The tilted column tracks the geometric drift, short\n"
        "by the distance it takes to capture a cell in the first place."
    )

    # --- 2. what the mechanism buys --------------------------------------
    print("\n\nBest achievable split at each tilt, 4 mm channel.")
    print("The divider is swept and the highest purity above 90 % recovery kept.\n")
    print(f"{'tilt':>7} {'divider':>9} {'large':>9} {'small':>9} {'gap':>8} "
          f"{'recovery':>9} {'purity':>8}")
    best_rows = []
    for tilt in (0.0, -3.0, -5.0, -10.0, -15.0):
        best = None
        for split in np.arange(0.10, 0.95, 0.05):
            metrics = run(channel_length="4 mm", tilt_angle_deg=tilt,
                          split_position=float(split)).metrics
            if metrics["efficiency_percent"] > 90.0 and (
                best is None or metrics["purity_percent"] > best[1]["purity_percent"]
            ):
                best = (float(split), metrics)
        if best is None:
            print(f"{tilt:6.0f}o   no divider reaches 90 % recovery")
            best_rows.append({"tilt_deg": tilt, "divider_um": None,
                              "recovery_percent": None, "purity_percent": None})
            continue
        split, metrics = best
        pops = metrics["per_population"]
        gap = pops["mcf7"]["mean_outlet_x_um"] - pops["rbc"]["mean_outlet_x_um"]
        print(f"{tilt:6.0f}o {split * WIDTH_UM:8.0f}um {pops['mcf7']['mean_outlet_x_um']:8.1f} "
              f"{pops['rbc']['mean_outlet_x_um']:8.1f} {gap:7.1f} "
              f"{metrics['efficiency_percent']:8.1f}% {metrics['purity_percent']:7.1f}%")
        best_rows.append({
            "tilt_deg": tilt, "divider_um": split * WIDTH_UM,
            "large_x_um": pops["mcf7"]["mean_outlet_x_um"],
            "small_x_um": pops["rbc"]["mean_outlet_x_um"], "gap_um": gap,
            "recovery_percent": metrics["efficiency_percent"],
            "purity_percent": metrics["purity_percent"],
        })

    print(
        "\nIn this geometry the straight device cannot do the job at all: the\n"
        "channel is one wavelength wide, so both populations end up around the\n"
        "same node and no divider separates them. Tilting decouples the\n"
        "displacement from the node spacing and the separation becomes clean."
    )
    print(
        "\nMind the sign: a positive angle deflects toward -x. With the sample on\n"
        "the left wall you want a NEGATIVE angle, or every cell is pressed into\n"
        "the wall it started against. The model warns when that happens."
    )

    out = ASSETS / "10_tilted_angle_ssaw.csv"
    table = pd.concat([
        pd.DataFrame(drift_rows).assign(part="drift_vs_length"),
        pd.DataFrame(best_rows).assign(part="best_split_vs_tilt"),
    ])
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
