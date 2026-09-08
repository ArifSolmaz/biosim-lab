"""Real sample composition: rare cells, and suspensions too dense to model.

A 300-vs-300 test bench says nothing about a tube holding one tumour cell per
billion blood cells at 45 % cells by volume. These cover the two things that
have to be right before such a separation can be simulated at all: composing
per-population probabilities onto real abundances, and refusing to pretend a
non-dilute suspension is one.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core import samples

# -- how crowded is the sample -------------------------------------------


def test_whole_blood_reproduces_the_haematocrit() -> None:
    """A sanity check on the composition: the volume fraction IS haematocrit.

    If this drifts, the cell radii or the reference counts are wrong.
    """
    fraction = samples.volume_fraction(samples.WHOLE_BLOOD)
    assert 0.38 < fraction < 0.52, f"got {fraction:.3f}, not a plausible haematocrit"


def test_whole_blood_is_not_a_dilute_suspension() -> None:
    """The headline reason no protocol runs whole blood through such a device."""
    report = samples.dilution_report(samples.WHOLE_BLOOD)
    assert not report.is_dilute
    assert report.mean_separation_radii < 3.0, "cells are ~2 radii apart"
    assert report.required_dilution > 50.0


def test_separation_scales_as_the_cube_root_of_dilution() -> None:
    """Spacing goes as phi^(-1/3); an 8x dilution must double it."""
    blood = samples.WHOLE_BLOOD
    assert samples.mean_separation_radii(samples.dilute(blood, 8.0)) == pytest.approx(
        2.0 * samples.mean_separation_radii(blood), rel=1e-9
    )


def test_the_required_dilution_actually_reaches_the_limit() -> None:
    blood = samples.WHOLE_BLOOD
    factor = samples.required_dilution(blood)
    assert samples.dilution_report(samples.dilute(blood, factor)).is_dilute


def test_diluting_by_less_than_one_is_refused() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        samples.dilute(samples.WHOLE_BLOOD, 0.5)


# -- the lysis step every published protocol uses -------------------------


def test_lysis_removes_the_red_cells_and_leaves_everything_else() -> None:
    lysed = samples.rbc_lysis(samples.WHOLE_BLOOD)
    assert lysed["rbc"] == pytest.approx(samples.WHOLE_BLOOD["rbc"] * 1e-3)
    assert lysed["wbc"] == pytest.approx(samples.WHOLE_BLOOD["wbc"])


def test_lysis_is_what_makes_the_sample_modellable() -> None:
    """45 % by volume down to under 1 %: the whole point of the step."""
    before = samples.dilution_report(samples.WHOLE_BLOOD)
    after = samples.dilution_report(samples.rbc_lysis(samples.WHOLE_BLOOD))
    assert before.volume_fraction > 0.4
    assert after.volume_fraction < 0.01
    assert after.mean_separation_radii > 4.0 * before.mean_separation_radii


def test_resuspension_volume_dilutes_the_survivors() -> None:
    """Li et al. resuspend 1 mL in 1 mL, so leukocytes are unchanged at ratio 1."""
    same = samples.rbc_lysis(samples.WHOLE_BLOOD, final_volume_ratio=1.0)
    halved = samples.rbc_lysis(samples.WHOLE_BLOOD, final_volume_ratio=2.0)
    assert same["wbc"] == pytest.approx(samples.WHOLE_BLOOD["wbc"])
    assert halved["wbc"] == pytest.approx(samples.WHOLE_BLOOD["wbc"] / 2.0)


# -- composing probabilities onto a real tube -----------------------------


def _composition() -> dict[str, float]:
    return {"mcf7": 10.0, "wbc": 1.0e6}


def test_recovery_and_depletion_are_computed_at_the_real_ratio() -> None:
    result = samples.compose(
        {"mcf7": 0.9, "wbc": 0.01}, _composition(), targets={"mcf7"}
    )
    assert result["recovery_percent"] == pytest.approx(90.0)
    assert result["background_depletion_log10"] == pytest.approx(2.0)
    assert result["target_collected"] == pytest.approx(9.0)


def test_purity_is_hopeless_at_real_ratios_even_when_the_device_works() -> None:
    """The number the 50:50 bench hides: a good device still gives ~1e-3 purity."""
    result = samples.compose(
        {"mcf7": 1.0, "wbc": 0.001}, _composition(), targets={"mcf7"}
    )
    assert result["recovery_percent"] == pytest.approx(100.0)
    assert result["purity_out"] < 0.02
    assert result["enrichment_fold"] > 100.0


# -- the trap this exists to close ----------------------------------------


def test_zero_simulated_hits_is_not_infinite_depletion() -> None:
    """Losing every simulated background cell means unmeasured, not eliminated.

    At 1e6 cells per mL a collection probability of 0.1 % still puts a thousand
    of them in the outlet, and 200 simulated cells cannot see that.
    """
    naive = samples.compose(
        {"mcf7": 1.0, "wbc": 0.0}, _composition(), targets={"mcf7"}
    )
    assert naive["background_depletion_log10"] == float("inf")

    bounded = samples.compose(
        {"mcf7": 1.0, "wbc": 0.0}, _composition(), targets={"mcf7"},
        simulated_counts={"mcf7": 200, "wbc": 200},
    )
    demonstrated = bounded["background_depletion_log10_demonstrated"]
    assert np.isfinite(demonstrated)
    assert demonstrated < 3.0, "200 cells cannot demonstrate 3 logs of depletion"
    assert bounded["background_collected_upper_bound"] > 0.0


def test_more_simulated_cells_demonstrate_more_depletion() -> None:
    """The bound must be a statement about the ensemble, not about the device."""
    def demonstrated(n: int) -> float:
        return samples.compose(
            {"mcf7": 1.0, "wbc": 0.0}, _composition(), targets={"mcf7"},
            simulated_counts={"mcf7": n, "wbc": n},
        )["background_depletion_log10_demonstrated"]

    assert demonstrated(2000) > demonstrated(200)


def test_the_bound_is_never_more_optimistic_than_the_point_estimate() -> None:
    result = samples.compose(
        {"mcf7": 0.9, "wbc": 0.05}, _composition(), targets={"mcf7"},
        simulated_counts={"mcf7": 300, "wbc": 300},
    )
    assert (
        result["background_depletion_log10_demonstrated"]
        <= result["background_depletion_log10"] + 1e-12
    )
