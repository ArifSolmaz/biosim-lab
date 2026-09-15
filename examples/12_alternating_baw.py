"""Alternating-frequency BAW: one transducer, two resonances, a sort in time.

A single piezoceramic under a hard-walled (silicon/glass) channel can excite
more than one standing wave across the channel: the half-wave resonance, with
one pressure node on the midline, and its third harmonic, with nodes at W/6,
W/2 and 5W/6. Switching between them in time sorts cells by how far they get
during the first phase (Zhang et al. 2023, doi:10.3390/ijms24043338):

* **1 MHz phase** --- everything drifts toward the midline, large and stiff cells
  fastest. A cancer cell should get more than W/6 from where it started; a PBMC
  less.
* **3 MHz phase** --- each cell falls onto the nearest of three nodes. The ones
  that crossed W/3 go to W/2 (collection), the rest back to W/6 (waste).

Repeat for at least two cycles and the populations end up on different nodes.

This script runs the device of ``benchmarks/benchmark_02_alternating_baw`` and
shows the three things worth seeing: the time traces (with the frequency phases
shaded), the design window in the 1 MHz amplitude, and the RK4 integrator
against SciPy's adaptive one. Writes ``assets/12_*.html`` and a CSV.

Run::

    python examples/12_alternating_baw.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # for the benchmark's config helpers

from benchmarks.benchmark_02_alternating_baw import cases  # noqa: E402
from biosim_lab.core.plugin import RegimeWarning  # noqa: E402
from biosim_lab.core.viz.curves import (  # noqa: E402
    force_timeline_figure,
    lateral_timeline_figure,
)
from biosim_lab.instruments.saw_sorter.design import operating_window  # noqa: E402
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterSimulation  # noqa: E402

ASSETS = ROOT / "assets"


def main() -> None:
    warnings.simplefilter("ignore", RegimeWarning)
    ASSETS.mkdir(exist_ok=True)

    # 1. One run at the paper's baseline drive, as configured.
    params = cases.make_params()
    sim = SAWSorterSimulation(params)
    out = sim.run()
    rule = out.diagnostics["switching"]["separation_rule"]
    print(f"mode {params.mode}: capture {out.metrics['capture_efficiency_percent']:.1f} %, "
          f"PBMC contamination {out.metrics['contamination_rate_percent']:.1f} %")
    for label, r in rule["populations"].items():
        print(f"  W/6 rule, {label:5s}: moves {r['displacement_um']:5.0f} um in the 1 MHz "
              f"phase (threshold {rule['threshold_um']:.0f} um) -> "
              f"{'ok' if r['satisfied'] else 'VIOLATED'}")

    # 2. The picture of Fig. 8: mean-sized cells, all entering at one point of the cycle.
    demo = cases.fig8_case(0.0)
    t_max = 3.2 * sim.schedule.period
    timeline = sim.schedule.timeline(0.0, t_max)
    lateral_timeline_figure(
        demo.tracks.trajectories, channel_width=params.channel_width, timeline=timeline,
        reference_lines=[(1 / 6, "W/6"), (1 / 3, "W/3"), (0.5, "W/2")], t_max=t_max,
        title="Cells entering at the start of the 1 MHz phase",
    ).write_html(ASSETS / "12_alternating_baw_trajectories.html", include_plotlyjs="cdn")
    labels = np.asarray(demo.forces["label"].values, dtype=str)
    pick = [int(np.flatnonzero(labels == lab)[6]) for lab in ("MCF7", "PBMC")]
    force_timeline_figure(demo.forces, particles=pick, timeline=timeline, t_max=t_max
                          ).write_html(ASSETS / "12_alternating_baw_forces.html",
                                       include_plotlyjs="cdn")

    # 3. The design window in the 1 MHz amplitude (Fig. 2c, found automatically).
    sweep = cases.sweep("V1", (6.0, 7.0, 8.0, 9.0, 10.0, 12.0))
    win = operating_window(sweep, "V1", capture_min=90.0, contamination_max=5.0)
    print(sweep[["V1", "capture_efficiency_percent", "contamination_rate_percent"]]
          .round(1).to_string(index=False))
    print(f"best 1 MHz amplitude by Youden's J: {win['best_setting']:g} Vpp "
          f"({win['best_capture_percent']:.0f} % / {win['best_contamination_percent']:.1f} %); "
          f"window at >= 90 % / <= 5 %: {win['window'] or 'none'}")

    # 4. The paper's integrator (RK4) against SciPy's adaptive LSODA, same cells.
    fixed = cases.make_params({"entry": "fixed", "entry_time_in_cycle": 0.3,
                               "time_step": 2e-3})
    rk = SAWSorterSimulation(fixed).run()
    lsoda = SAWSorterSimulation(fixed.model_copy(update={
        "switching": fixed.switching.model_copy(update={"integrator": "solve_ivp"})})).run()
    a, b = rk.tracks.trajectories, lsoda.tracks.trajectories
    common = np.intersect1d(a["time"].values, b["time"].values)[:40]
    dx = np.abs(a["position"].sel(time=common, axis="x").values
                - b["position"].sel(time=common, axis="x").values).max()
    print(f"RK4 vs LSODA: largest lateral difference {dx * 1e9:.2f} nm; same outlet for "
          f"{(rk.cells['outlet'] == lsoda.cells['outlet']).mean():.0%} of cells")

    pd.DataFrame(sweep).to_csv(ASSETS / "12_alternating_baw_V1.csv", index=False)


if __name__ == "__main__":
    main()
