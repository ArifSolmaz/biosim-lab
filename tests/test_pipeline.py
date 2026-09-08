"""Chaining instruments, and the error budget that only exists when they chain.

The claims under test are the two things the pipeline exists to show: that a
size-selective sort changes the population handed downstream, and that the
stages' uncertainties compose in quadrature rather than accumulating naively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biosim_lab.pipeline import Pipeline, Sample, Stage, StageReport
from biosim_lab.stages import CountStage, SortStage, TrackStage


def _cells(radii_um, alive=True) -> pd.DataFrame:
    radii = np.asarray(radii_um, dtype=float) * 1e-6
    return pd.DataFrame({
        "radius_m": radii,
        "label": ["cell"] * radii.size,
        "alive": [alive] * radii.size,
    })


class _Fake(Stage):
    """A stage with a known yield and a known uncertainty."""

    def __init__(self, name: str, keep: float, sigma: float) -> None:
        self.name = name
        self.keep = keep
        self.sigma = sigma

    def apply(self, sample: Sample) -> Sample:
        n_out = int(round(sample.n * self.keep))
        return Sample(
            cells=sample.cells.iloc[:n_out],
            volume_ml=sample.volume_ml,
            history=[*sample.history, StageReport(self.name, sample.n, n_out, self.sigma)],
        )


# -- the bookkeeping ------------------------------------------------------


def test_stages_run_in_order_and_each_leaves_one_record() -> None:
    sample = Sample(cells=_cells(np.full(100, 8.0)))
    out = Pipeline([_Fake("a", 0.5, 0.1), _Fake("b", 0.5, 0.2)]).run(sample)
    assert [r.stage for r in out.history] == ["a", "b"]
    assert out.n == 25


def test_a_stage_that_does_not_record_itself_is_an_error() -> None:
    """A silent stage would leave a plausible but incomplete error budget."""

    class Silent(Stage):
        name = "silent"

        def apply(self, sample: Sample) -> Sample:
            return sample

    with pytest.raises(RuntimeError, match="exactly one StageReport"):
        Pipeline([Silent()]).run(Sample(cells=_cells([8.0])))


def test_an_empty_pipeline_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        Pipeline([])


# -- the error budget -----------------------------------------------------


def test_independent_uncertainties_add_in_quadrature_not_linearly() -> None:
    """3 % and 4 % make 5 %, not 7 %. The difference is the whole point."""
    out = Pipeline([_Fake("a", 1.0, 0.03), _Fake("b", 1.0, 0.04)]).run(
        Sample(cells=_cells(np.full(50, 8.0)))
    )
    assert out.total_relative_uncertainty() == pytest.approx(0.05)


def test_the_dominant_stage_is_identified() -> None:
    out = Pipeline([_Fake("small", 1.0, 0.01), _Fake("big", 1.0, 0.20)]).run(
        Sample(cells=_cells(np.full(50, 8.0)))
    )
    dominant = out.dominant_uncertainty()
    assert dominant is not None
    assert dominant.stage == "big"
    assert "big" in Pipeline([_Fake("x", 1.0, 0.0)]).summary(out)


def test_a_stage_with_unmeasurable_uncertainty_does_not_poison_the_total() -> None:
    """One NaN must not turn the whole budget into NaN and hide the rest."""
    out = Pipeline([_Fake("a", 1.0, 0.03), _Fake("b", 1.0, float("nan"))]).run(
        Sample(cells=_cells(np.full(50, 8.0)))
    )
    assert out.total_relative_uncertainty() == pytest.approx(0.03)


# -- the sample itself ----------------------------------------------------


def test_radius_statistics_describe_the_population() -> None:
    sample = Sample(cells=_cells([4.0, 6.0, 8.0]))
    stats = sample.radius_stats()
    assert stats["mean_m"] == pytest.approx(6e-6)
    assert stats["cv"] == pytest.approx(2e-6 / 6e-6)


def test_concentration_uses_the_carried_volume() -> None:
    assert Sample(cells=_cells(np.full(500, 8.0)), volume_ml=0.5).concentration_per_ml == 1000.0


# -- the claim the whole pipeline exists to make --------------------------


@pytest.mark.slow
def test_sorting_shifts_the_size_distribution_handed_downstream() -> None:
    """The r^2 law means the collected population is bigger and narrower.

    This is the finding that is invisible when the instruments run separately,
    and the reason ``CountStage`` takes its size distribution from the sample
    rather than from a default.
    """
    from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams

    params = SAWSorterParams(
        frequency="6.632 MHz", voltage_pp="15 V", channel_width="300 um",
        channel_height="50 um", channel_length="2 mm", flow_rate="10 uL/min",
        fluid="water", substrate="linbo3_128yx", inlet="sheath_sides",
        collection_fraction=1 / 3, mode="analytic", seed=20260907,
        populations=[
            {"cell_type": "mcf7", "count": 200, "target": True},
            {"cell_type": "rbc", "count": 200, "target": False},
        ],
    )
    sort = SortStage(params)
    out = Pipeline([sort]).run(Sample(cells=pd.DataFrame({"radius_m": [], "label": []})))

    loaded = sort.outcome.cells["radius_m"].to_numpy()
    collected = out.cells["radius_m"].to_numpy()

    assert collected.mean() > 1.3 * loaded.mean(), "collection must enrich large cells"
    loaded_cv = loaded.std(ddof=1) / loaded.mean()
    collected_cv = collected.std(ddof=1) / collected.mean()
    assert collected_cv < 0.5 * loaded_cv, "collection must narrow the distribution"


@pytest.mark.slow
def test_the_counter_is_handed_the_sorted_distribution_not_a_default() -> None:
    """If this regresses, the counter silently images the wrong population."""
    sample = Sample(cells=_cells(np.random.default_rng(0).normal(11.0, 0.6, 120)))
    out = Pipeline([CountStage(n_cells=100, image_size=640, seed=3)]).run(sample)

    metrics = out.history[-1].metrics
    expected_px = sample.radius_stats()["mean_m"] / 0.65e-6
    assert metrics["imaged_radius_mean_px"] == pytest.approx(expected_px, rel=1e-9)
    # Default would be 9.0 px; the sample's own ~17 px must be used instead.
    assert metrics["imaged_radius_mean_px"] > 12.0
    # And the recovered diameter must land near the truth it was given.
    assert metrics["mean_diameter_um"] == pytest.approx(22.0, rel=0.15)


@pytest.mark.slow
def test_the_three_stage_workflow_runs_end_to_end() -> None:
    from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams

    params = SAWSorterParams(
        frequency="6.632 MHz", voltage_pp="15 V", channel_width="300 um",
        channel_height="50 um", channel_length="2 mm", flow_rate="10 uL/min",
        fluid="water", substrate="linbo3_128yx", inlet="sheath_sides",
        collection_fraction=1 / 3, mode="analytic", seed=20260907,
        populations=[
            {"cell_type": "mcf7", "count": 150, "target": True},
            {"cell_type": "rbc", "count": 150, "target": False},
        ],
    )
    pipeline = Pipeline([
        SortStage(params),
        CountStage(n_cells=100, image_size=640, seed=7),
        TrackStage(seed=7),
    ])
    out = pipeline.run(Sample(cells=pd.DataFrame({"radius_m": [], "label": []})))

    assert [r.stage for r in out.history] == ["sort", "count", "track"]
    assert all(np.isfinite(r.relative_uncertainty) for r in out.history)
    assert 0.0 < out.total_relative_uncertainty() < 1.0
    assert "combined" in pipeline.summary(out)
