"""Uncertainty reporting: binomial intervals and replicate spread."""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core.statistics import (
    replicate,
    summarise_replicates,
    summary_table,
    wilson_interval,
)


def test_wilson_has_width_at_a_perfect_score():
    """The whole reason for using Wilson.

    The normal approximation gives ``p ± z sqrt(p(1-p)/n)``, which is exactly
    zero at p = 1 — so 300 out of 300 would be reported as '100 % ± 0'. A sorter
    hits that case constantly.
    """
    w = wilson_interval(300, 300)
    assert w.point_percent == 100.0
    assert w.high_percent == 100.0
    assert w.low_percent < 99.5, "a perfect score still has a lower bound"
    assert w.half_width_percent > 0.0


def test_wilson_has_width_at_zero_too():
    w = wilson_interval(0, 300)
    assert w.point_percent == 0.0
    assert w.low_percent == 0.0
    assert 0.0 < w.high_percent < 5.0


def test_wilson_never_leaves_the_unit_interval():
    for trials in (1, 5, 30, 300):
        for successes in range(trials + 1):
            w = wilson_interval(successes, trials)
            assert 0.0 <= w.low_percent <= w.point_percent <= w.high_percent <= 100.0


def test_wilson_narrows_as_the_square_root_of_n():
    """Four times the cells should halve the interval."""
    small = wilson_interval(75, 100).half_width_percent
    large = wilson_interval(300, 400).half_width_percent
    assert large == pytest.approx(small / 2, rel=0.1)


def test_wilson_agrees_with_the_normal_interval_away_from_the_boundaries():
    """Where the textbook interval is valid, Wilson must not disagree with it."""
    successes, trials = 500, 1000
    w = wilson_interval(successes, trials)
    p = successes / trials
    naive = 1.96 * np.sqrt(p * (1 - p) / trials) * 100
    assert w.half_width_percent == pytest.approx(naive, rel=0.02)


def test_wilson_widens_with_confidence():
    assert (wilson_interval(80, 100, 0.99).half_width_percent
            > wilson_interval(80, 100, 0.95).half_width_percent
            > wilson_interval(80, 100, 0.80).half_width_percent)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    assert np.isnan(wilson_interval(0, 0).point_percent)


def test_replicate_summary_uses_student_t_not_z():
    """With five runs the t quantile is ~30 % wider than z; using z understates."""
    values = [10.0, 11.0, 9.0, 10.5, 9.5]
    s = summarise_replicates("x", values)
    assert s.n == 5
    assert s.mean == pytest.approx(10.0)
    half_width = 0.5 * (s.high - s.low)
    assert half_width > 1.96 * s.sem, "must be wider than the normal quantile"
    assert half_width == pytest.approx(2.776 * s.sem, rel=0.01)  # t(0.975, df=4)


def test_replicate_summary_handles_one_and_zero_runs():
    one = summarise_replicates("x", [7.0])
    assert one.mean == 7.0 and one.std == 0.0 and one.low == one.high == 7.0
    none = summarise_replicates("x", [])
    assert np.isnan(none.mean)


def test_replicate_ignores_non_finite_values():
    s = summarise_replicates("x", [1.0, np.nan, 3.0, np.inf])
    assert s.n == 2
    assert s.mean == pytest.approx(2.0)


def test_replicate_uses_a_different_seed_each_time():
    seen: list[int] = []

    def run(seed: int) -> dict[str, float]:
        seen.append(seed)
        rng = np.random.default_rng(seed)
        return {"value": float(rng.normal(10.0, 1.0))}

    result = replicate(run, n_replicates=6, base_seed=100)
    assert seen == [100, 101, 102, 103, 104, 105]
    assert len(set(seen)) == 6
    assert result["summary"]["value"].n == 6
    # Different seeds must give different draws, or the replication is fake.
    assert len({r["value"] for r in result["per_replicate"]}) == 6


def test_replicate_rejects_zero_runs():
    with pytest.raises(ValueError):
        replicate(lambda seed: {"a": 1.0}, n_replicates=0)


def test_summary_table_shape():
    result = replicate(lambda seed: {"a": float(seed), "b": 2.0}, n_replicates=4)
    table = summary_table(result)
    assert set(table["metric"]) == {"a", "b"}
    assert {"mean", "std", "sem", "ci95_low", "ci95_high", "n"} <= set(table.columns)


@pytest.mark.slow
def test_sorter_reports_both_kinds_of_uncertainty():
    """Counting error within a run, and sample-to-sample error across runs.

    They answer different questions and neither substitutes for the other.
    """
    from biosim_lab.instruments.saw_sorter.simulate import (
        SAWSorterParams,
        SAWSorterSimulation,
        replicate_sorting,
    )

    params = SAWSorterParams(
        populations=[
            {"cell_type": "mcf7", "count": 100, "target": True},
            {"cell_type": "rbc", "count": 100, "target": False},
        ],
        n_time_samples=41,
        seed=2024,
    )

    import warnings

    from biosim_lab.core.plugin import RegimeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        metrics = SAWSorterSimulation(params).run().metrics

    # Within-run counting error must bracket the point estimate.
    assert metrics["efficiency_ci_low"] <= metrics["efficiency_percent"]
    assert metrics["efficiency_percent"] <= metrics["efficiency_ci_high"]
    assert metrics["purity_ci_low"] <= metrics["purity_percent"] <= metrics["purity_ci_high"]
    # And must have real width even at a perfect efficiency.
    assert metrics["efficiency_ci_high"] - metrics["efficiency_ci_low"] > 0

    result = replicate_sorting(params, n_replicates=4)
    assert result["n_replicates"] == 4
    assert len(set(result["seeds"])) == 4
    purity = result["summary"]["purity_percent"]
    assert purity.n == 4
    assert purity.low <= purity.mean <= purity.high
    assert "table" in result
