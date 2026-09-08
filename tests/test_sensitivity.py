"""Ranking the unsourced numbers by how much they actually matter.

The library flags 38 values as ASSUMPTION. That list says what is unknown; the
sensitivity scan says what the ignorance costs, which is the part a user can act
on. These tests check the scan measures what it claims to, and --- more
importantly --- that it cannot silently report a ranking of sampling noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core import sensitivity
from biosim_lab.core.materials import CELL_TYPES, get_cell, get_fluid, perturbed

# -- the perturbation context manager ------------------------------------


def test_perturbation_reaches_the_lookup_and_is_undone() -> None:
    before = get_fluid("water").rho
    with perturbed([("fluid", "water", "density", 1100.0)]):
        assert get_fluid("water").rho == pytest.approx(1100.0)
    assert get_fluid("water").rho == pytest.approx(before)


def test_derived_properties_follow_the_perturbation() -> None:
    """kappa = 1/(rho c^2) is a property, so it must track a density change."""
    with perturbed([("fluid", "water", "density", 1100.0)]):
        fluid = get_fluid("water")
        assert fluid.kappa == pytest.approx(1.0 / (1100.0 * fluid.c**2))


def test_the_library_is_restored_even_when_the_body_raises() -> None:
    """The tables are process-wide globals; a leak would corrupt every later run."""
    before = get_cell("mcf7").rho
    with pytest.raises(RuntimeError), perturbed([("cell", "mcf7", "density", 1500.0)]):
        raise RuntimeError("boom")
    assert get_cell("mcf7").rho == pytest.approx(before)


def test_perturbation_keeps_the_provenance() -> None:
    """A perturbed assumption is still an assumption, and must still say so."""
    original = CELL_TYPES["mcf7"].density.prov
    with perturbed([("cell", "mcf7", "density", 1100.0)]):
        assert CELL_TYPES["mcf7"].density.prov is original


def test_unknown_material_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown cell"), perturbed(
        [("cell", "not_a_cell", "density", 1.0)]
    ):
        pass


# -- target discovery -----------------------------------------------------


def test_every_flagged_assumption_becomes_a_target() -> None:
    from biosim_lab.core.materials import audit

    assert len(sensitivity.assumption_targets()) == len(audit(only_assumptions=True))


def test_targets_can_be_filtered_to_what_an_experiment_touches() -> None:
    targets = sensitivity.assumption_targets(materials=("water", "mcf7"))
    assert targets
    assert {t.material for t in targets} <= {"water", "mcf7"}


def test_a_target_reads_its_current_magnitude() -> None:
    target = sensitivity.Target("cell", "mcf7", "density")
    assert target.current() == pytest.approx(float(get_cell("mcf7").density.magnitude))


# -- the elasticity itself ------------------------------------------------


def test_elasticity_of_a_known_power_law_is_its_exponent() -> None:
    """For Y = X^n the elasticity is exactly n, whatever the units.

    This is the check that the normalisation is right. Density enters as rho^2
    here, so the scan must return 2.0 --- not the raw slope, which would carry
    units and could not be compared across inputs.
    """
    rho0 = float(get_cell("mcf7").density.magnitude)

    def run(seed: int) -> dict[str, float]:
        return {"y": float(get_cell("mcf7").rho) ** 2}

    report = sensitivity.scan(
        run, [sensitivity.Target("cell", "mcf7", "density")],
        metrics=("y",), fraction=0.01,
    )
    row = report.ranked("y")[0]
    assert row.elasticity == pytest.approx(2.0, rel=1e-3)
    assert row.baseline_metric == pytest.approx(rho0**2)


def test_an_input_the_model_ignores_ranks_at_zero() -> None:
    def run(seed: int) -> dict[str, float]:
        return {"y": 42.0}

    report = sensitivity.scan(
        run, [sensitivity.Target("cell", "mcf7", "density")], metrics=("y",),
    )
    assert report.ranked("y")[0].elasticity == pytest.approx(0.0)


def test_ranking_puts_the_biggest_lever_first() -> None:
    def run(seed: int) -> dict[str, float]:
        # Strongly dependent on cell density, weakly on fluid viscosity.
        return {"y": float(get_cell("mcf7").rho) ** 3 * float(get_fluid("water").mu) ** 0.1}

    targets = [
        sensitivity.Target("fluid", "water", "viscosity"),
        sensitivity.Target("cell", "mcf7", "density"),
    ]
    ranked = sensitivity.scan(run, targets, metrics=("y",), fraction=0.01).ranked("y")
    assert ranked[0].label == "mcf7.density"
    assert abs(ranked[0].elasticity) > abs(ranked[1].elasticity)


# -- the two ways this analysis goes wrong --------------------------------


def test_sampling_noise_produces_no_resolved_findings() -> None:
    """A model whose output is pure noise must produce NO resolved findings.

    Without a resolution test the scan would happily rank random numbers and
    hand back a confident, meaningless ordering.
    """
    def run(seed: int) -> dict[str, float]:
        return {"y": 100.0 + np.random.default_rng(seed).normal(0.0, 5.0)}

    targets = sensitivity.assumption_targets(materials=("mcf7",))
    report = sensitivity.scan(run, targets, metrics=("y",))
    assert report.noise_floor["y"] > 0.0
    assert not any(r.resolved for r in report.rows)


def test_a_saturated_metric_is_named_rather_than_reported_as_a_table_of_zeros() -> None:
    """At 100 % efficiency every derivative is genuinely zero.

    That is a real result --- the operating point is robust --- but it looks
    identical to a broken scan, so the report must say which it is.
    """
    def run(seed: int) -> dict[str, float]:
        return {"efficiency_percent": 100.0}

    report = sensitivity.scan(
        run, sensitivity.assumption_targets(materials=("mcf7",)),
        metrics=("efficiency_percent",),
    )
    assert report.is_saturated("efficiency_percent")
    assert "SATURATED" in report.summary("efficiency_percent")


def test_central_difference_cancels_the_quadratic_term() -> None:
    """One-sided differencing would carry an O(h) error; central is O(h^2)."""
    rho0 = float(get_cell("mcf7").density.magnitude)

    def run(seed: int) -> dict[str, float]:
        rho = float(get_cell("mcf7").rho)
        return {"y": rho + 3.0 * (rho - rho0) ** 2 / rho0}

    # True elasticity at the baseline is 1.0; the quadratic term must cancel.
    report = sensitivity.scan(
        run, [sensitivity.Target("cell", "mcf7", "density")],
        metrics=("y",), fraction=0.05,
    )
    assert report.ranked("y")[0].elasticity == pytest.approx(1.0, rel=1e-9)


# -- the real thing -------------------------------------------------------


@pytest.mark.slow
def test_cell_density_is_the_assumption_that_moves_the_sorter() -> None:
    """The scan's headline claim, checked end to end on the real instrument.

    At a marginal operating point the assumed MCF7 density is the one unsourced
    number that changes what gets collected. If this ever stops being true the
    documentation in ``examples/07_assumption_sensitivity.py`` is stale.
    """
    import warnings

    from biosim_lab.core.plugin import RegimeWarning
    from biosim_lab.instruments.saw_sorter.simulate import (
        SAWSorterParams,
        SAWSorterSimulation,
    )

    def run(seed: int) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            params = SAWSorterParams(
                frequency="6.632 MHz", voltage_pp="15 V", channel_width="300 um",
                channel_height="50 um", channel_length="2 mm", flow_rate="10 uL/min",
                fluid="water", substrate="linbo3_128yx", inlet="sheath_sides",
                collection_fraction=1 / 3, mode="analytic", seed=seed,
                populations=[
                    {"cell_type": "mcf7", "count": 150, "target": True},
                    {"cell_type": "rbc", "count": 150, "target": False},
                ],
            )
            outcome = SAWSorterSimulation(params).run()
        return {"efficiency_percent": float(outcome.metrics["efficiency_percent"])}

    report = sensitivity.scan(
        run, sensitivity.assumption_targets(materials=("water", "mcf7", "rbc")),
        metrics=("efficiency_percent",), fraction=0.05,
    )
    top = report.ranked("efficiency_percent")[0]
    assert top.label == "mcf7.density"
    assert abs(top.elasticity) > 0.3
    assert top.resolved, "the headline finding must clear the paired-difference test"


def test_fewer_than_three_seeds_is_refused() -> None:
    """With two seeds there is one degree of freedom and no usable error bar."""
    def run(seed: int) -> dict[str, float]:
        return {"y": 1.0}

    with pytest.raises(ValueError, match="at least three seeds"):
        sensitivity.scan(
            run, [sensitivity.Target("cell", "mcf7", "density")],
            metrics=("y",), seeds=(1, 2),
        )


def test_common_random_numbers_make_a_small_effect_visible() -> None:
    """The paired design is what lets a 5 % effect be seen through 30 % noise.

    The same measurement with independent ensembles is swamped. This is the
    reason the perturbed runs reuse the baseline's seed rather than drawing
    fresh ones.
    """
    rho0 = float(get_cell("mcf7").density.magnitude)

    def paired(seed: int) -> dict[str, float]:
        # Ensemble noise depends only on the seed, so it cancels in the pair.
        noise = np.random.default_rng(seed).normal(0.0, 30.0)
        return {"y": float(get_cell("mcf7").rho) / rho0 * 100.0 + noise}

    report = sensitivity.scan(
        paired, [sensitivity.Target("cell", "mcf7", "density")], metrics=("y",),
    )
    row = report.ranked("y")[0]

    # The pairing cancels the ensemble noise EXACTLY in the difference, which is
    # the whole point: a +/-5 % perturbation of a metric whose level is 100 must
    # give a difference of exactly 10, through noise six times larger.
    assert row.delta_metric == pytest.approx(10.0, rel=1e-9)
    assert row.resolved
    assert report.noise_floor["y"] > 10.0

    # The elasticity divides by the baseline level, which is NOT paired and so
    # keeps its own sampling error. The numerator is the precise part.
    assert row.elasticity == pytest.approx(1.0, rel=0.15)
