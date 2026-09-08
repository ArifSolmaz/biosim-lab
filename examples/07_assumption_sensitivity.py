"""Which unsourced numbers actually change the answer?

The material library flags 38 values as ASSUMPTION: numbers that could not be
traced to a primary source. Disclosing them is necessary but not useful on its
own --- it tells a user what is unknown, not what to do about it. This script
ranks them by how much each one actually moves the sorting result, so the answer
to "what should I measure first?" is a short list rather than 38 items.

It scans **two operating points**, because the answer is different at each:

* the **design point** (15 Vpp, 5 uL/min), where the sorter recovers essentially
  every target cell. Sitting against that ceiling makes it insensitive to
  everything: the largest elasticity there is 0.007.
* a **marginal point** (15 Vpp, 10 uL/min, ~79 % efficiency), where cells only
  just reach the node. This is where the device actually discriminates, and
  where a real experiment operates when it is pushing throughput.

The same unsourced number matters **130 times more** at the second point than at
the first, which is the practical result: a sensitivity ranking is a property of
the operating point, not of the model, and quoting one without the other is
meaningless. Run this at the operating point you actually use.

What it finds: of the 38 flagged assumptions, exactly one moves the answer ---
the **MCF7 cell density** (elasticity 0.94, so a 5 % error in it costs ~5 % of
your yield). Cell radius spread is a distant second at -0.16. Everything else is
either unused by the acoustic model or below the resolution of the scan. So the
answer to "what should I measure first?" is one item, not thirty-eight.

Each assumption is perturbed by +/-5 %, paired within each of five ensembles so
the sampling noise cancels, and the result is the normalised elasticity
``(dY/Y)/(dX/X)`` with a paired t-test deciding whether it is resolved at all.

Writes ``assets/07_assumption_sensitivity.csv``.

Run::

    python examples/07_assumption_sensitivity.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.sensitivity import assumption_targets, scan
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# The materials this experiment actually touches. Scanning a cell type that is
# not in the suspension only produces a row of exact zeros.
IN_PLAY = ("water", "mcf7", "rbc", "linbo3_128yx", "pdms")

METRICS = ("efficiency_percent", "purity_percent")

BASE = dict(
    frequency="6.632 MHz",
    voltage_pp="15 V",
    channel_width="300 um",
    channel_height="50 um",
    channel_length="2 mm",
    fluid="water",
    substrate="linbo3_128yx",
    inlet="sheath_sides",
    collection_fraction=1 / 3,
    mode="analytic",
    populations=[
        {"cell_type": "mcf7", "count": 300, "target": True},
        {"cell_type": "rbc", "count": 300, "target": False},
    ],
)

OPERATING_POINTS = {
    "design": {"flow_rate": "5 uL/min"},
    "marginal": {"flow_rate": "10 uL/min"},
}


def make_runner(overrides: dict):
    """A ``run(seed)`` callable that runs the sorter and returns its metrics."""

    def _run(seed: int) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            params = SAWSorterParams(seed=seed, **{**BASE, **overrides})
            outcome = SAWSorterSimulation(params).run()
        return {m: float(outcome.metrics[m]) for m in METRICS}

    return _run


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    targets = assumption_targets(materials=IN_PLAY)
    print(f"scanning {len(targets)} assumptions in play "
          f"(of {len(assumption_targets())} flagged in the library)\n")

    frames = []
    for name, overrides in OPERATING_POINTS.items():
        # Within each seed the up- and down-perturbed runs share that seed, so
        # the ensemble is identical and the difference isolates the parameter.
        # Repeating over several seeds is what puts an error bar on it.
        report = scan(make_runner(overrides), targets, metrics=METRICS, fraction=0.05)

        header = f"=== {name} operating point ({overrides['flow_rate']}) ==="
        print(header)
        for metric in METRICS:
            print(report.summary(metric, top=12))
            print()

        frame = report.to_dataframe()
        frame.insert(0, "operating_point", name)
        frames.append(frame)

    import pandas as pd

    out = ASSETS / "07_assumption_sensitivity.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
