"""One workflow across three instruments, with the error budget carried through.

Sorting, counting and tracking are three separate instruments in this package.
A real experiment runs them in sequence, and two things only become visible when
they are chained rather than run one at a time.

**1. The sorter changes the population, not just its size.**
Acoustic radiation force scales with cell volume while Stokes drag scales with
radius, so migration speed goes as ``r^2`` and collection is size-selective. The
suspension arriving at the counter is therefore not the one that was loaded --- it
is shifted to larger diameters and much narrower. This script measures that shift
and hands the *measured* distribution to the imaging stage, which is what a
counter would actually see.

**2. The uncertainties compose, and one stage dominates.**
Sorting contributes binomial error, counting contributes Poisson error, tracking
contributes sample-to-sample error. They are independent, so they add in
quadrature and the total is dominated by whichever is largest. That tells you
where more work would actually help --- imaging more fields of view cannot fix a
sorting-limited measurement.

Run::

    python examples/08_pipeline_sort_count_track.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams
from biosim_lab.pipeline import Pipeline, Sample
from biosim_lab.stages import CountStage, SortStage, TrackStage

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def build_params() -> SAWSorterParams:
    """A marginal operating point: the sorter is discriminating, not saturated."""
    return SAWSorterParams(
        frequency="6.632 MHz",
        voltage_pp="15 V",
        channel_width="300 um",
        channel_height="50 um",
        channel_length="2 mm",
        flow_rate="10 uL/min",
        fluid="water",
        substrate="linbo3_128yx",
        inlet="sheath_sides",
        collection_fraction=1 / 3,
        mode="analytic",
        seed=20260907,
        populations=[
            {"cell_type": "mcf7", "count": 300, "target": True},
            {"cell_type": "rbc", "count": 300, "target": False},
        ],
    )


def describe(name: str, cells: pd.DataFrame) -> dict[str, float]:
    radii_um = np.asarray(cells["radius_m"], dtype=float) * 1e6
    mean = float(radii_um.mean())
    cv = float(radii_um.std(ddof=1) / mean) if radii_um.size > 1 else 0.0
    return {
        "population": name,
        "n": int(len(cells)),
        "mean_diameter_um": 2.0 * mean,
        "diameter_cv": cv,
    }


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    sort = SortStage(build_params())
    pipeline = Pipeline([
        sort,
        # n_cells is the occupancy of one field of view, not the whole sample:
        # a Neubauer square holds a small fraction of the suspension.
        CountStage(n_cells=120, image_size=640, seed=7),
        TrackStage(seed=7),
    ])

    # The first stage generates its own ensemble, so the pipeline starts empty.
    final = pipeline.run(Sample(cells=pd.DataFrame({"radius_m": [], "label": []})))

    loaded = sort.outcome.cells
    collected = loaded[loaded["outlet"] == "collect"]

    print("What the sorter does to the population")
    print("-" * 62)
    rows = [describe("loaded", loaded), describe("collected", collected)]
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    shift = rows[1]["mean_diameter_um"] / rows[0]["mean_diameter_um"] - 1.0
    narrowing = rows[0]["diameter_cv"] / rows[1]["diameter_cv"]
    print(
        f"\n  mean diameter shifts by {shift:+.0%} and the spread narrows "
        f"{narrowing:.1f}x."
    )
    print(
        "  A counter gated on the loaded distribution would be measuring the\n"
        "  wrong population. This is the reason to chain the instruments."
    )

    counted = final.history[1].metrics
    recovered = counted["mean_diameter_um"]
    handed = rows[1]["mean_diameter_um"]
    bias = recovered / handed - 1.0
    print(
        f"\n  the imaging stage recovers a mean diameter of {recovered:.2f} um "
        f"against\n  the {handed:.2f} um it was handed ({bias:+.1%}): watershed "
        "boundaries sit\n  just inside the true edge, so segmentation "
        "under-measures slightly."
    )
    print(
        f"  detection recall {counted['detection_recall']:.2f}. Recall falls in "
        "crowded fields as cells\n  touch and merge, but the size estimate is "
        "far more robust than the count\n  --- which is what the conclusion above "
        "rests on."
    )

    print("\n\nError budget")
    print("-" * 62)
    print(pipeline.summary(final))

    out = ASSETS / "08_pipeline_error_budget.csv"
    pd.DataFrame([
        {
            "stage": r.stage,
            "n_in": r.n_in,
            "n_out": r.n_out,
            "yield_fraction": r.yield_fraction,
            "relative_uncertainty": r.relative_uncertainty,
            **{k: v for k, v in r.metrics.items() if not isinstance(v, (list, dict))},
        }
        for r in final.history
    ]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
