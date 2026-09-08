"""Real sample composition: what is actually in the tube, and at what ratio.

An instrument test usually simulates equal numbers of two populations, because
that is what gives good statistics on both. A real sample is nothing like that.
Whole blood holds about 5e9 red cells per mL and, in metastatic carcinoma,
somewhere between 1 and 10 circulating tumour cells --- a ratio near 1 : 1e9.

Two things follow, and both change how a separation must be simulated.

**You cannot simulate the real ratio directly.** Seeing ten tumour cells at that
abundance means tracking ten billion red cells. The way round it is to simulate a
statistically useful number of each population, measure each one's *probability*
of reaching the collection outlet, and then compose those probabilities back onto
the real abundances. :func:`compose` does that. It assumes cells behave
independently, which is the same assumption the force model already makes.

**Whole blood is not a dilute suspension.** At 45 % cells by volume the particles
are about two radii apart, so particle-particle scattering and acoustic streaming
are not small corrections --- they are the dominant physics, and none of it is in
this model. Real devices dilute, lyse the red cells, or pre-enrich for exactly
this reason. :func:`dilution_report` says how far from the dilute limit a given
composition sits and what dilution would fix it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .materials import Provenance, get_cell

#: Mean centre-to-centre spacing, in particle radii, below which a suspension
#: stops behaving as a set of independent particles. Derived from the volume
#: fraction as ``(4 pi / 3 phi)^(1/3)``.
DILUTE_SEPARATION_RADII = 10.0

DILUTE_LIMIT_PROVENANCE = Provenance(
    assumption="Ten particle radii of mean separation is taken as the point below "
    "which particle-particle acoustic interaction and streaming stop being small "
    "corrections. The underlying physics is continuous, not a threshold: treat it "
    "as the scale at which this model's non-interacting assumption starts to cost "
    "accuracy, not a line beyond which it fails."
)

#: Adult peripheral blood, cells per mL.
WHOLE_BLOOD: dict[str, float] = {
    "rbc": 5.0e9,
    "wbc": 7.0e6,
    "platelet": 3.0e8,
}

BLOOD_PROVENANCE: dict[str, Provenance] = {
    "rbc": Provenance(
        doi="10.1182/blood-2013-06-508325",
        citation="Blood reference intervals; adult erythrocyte count 4.5-5.9e12/L, "
        "midpoint taken",
    ),
    "wbc": Provenance(
        doi="10.1182/blood-2013-06-508325",
        citation="Adult leukocyte count 4.5-11e9/L, midpoint taken",
    ),
    "platelet": Provenance(
        doi="10.1182/blood-2013-06-508325",
        citation="Adult platelet count 150-400e9/L, midpoint taken",
    ),
}

#: Circulating tumour cells per mL of whole blood in metastatic carcinoma.
CTC_PER_ML_METASTATIC = 5.0

CTC_PROVENANCE = Provenance(
    doi="10.1158/1078-0432.CCR-04-0378",
    citation="Allard et al. (2004), Clin Cancer Res 10:6897 — CTCs detected in "
    "7.5 mL of blood across major carcinomas; counts are typically single digits "
    "per mL and span orders of magnitude between patients",
)


def cell_volume(cell_type: str) -> float:
    """Volume of one cell of *cell_type* [m^3], from its equivalent radius."""
    radius = float(get_cell(cell_type).r)
    return 4.0 / 3.0 * np.pi * radius**3


def volume_fraction(composition: dict[str, float]) -> float:
    """Fraction of the sample volume occupied by cells, from ``cells per mL``."""
    per_ml_in_m3 = 1e-6
    return float(
        sum(n * cell_volume(k) for k, n in composition.items()) / per_ml_in_m3
    )


def mean_separation_radii(composition: dict[str, float]) -> float:
    """Mean centre-to-centre spacing, in units of the mean particle radius.

    ``(4 pi / 3 phi)^(1/3)``: the spacing a random packing at volume fraction
    ``phi`` implies. More useful than the volume fraction itself, because it is
    the distance that decides whether particles interact.
    """
    phi = volume_fraction(composition)
    if phi <= 0.0:
        return float("inf")
    return float((4.0 * np.pi / (3.0 * phi)) ** (1.0 / 3.0))


def dilute(composition: dict[str, float], factor: float) -> dict[str, float]:
    """Dilute every population by *factor* (2.0 = one part sample, one part buffer)."""
    if factor < 1.0:
        raise ValueError(f"dilution factor must be >= 1, got {factor}")
    return {k: v / float(factor) for k, v in composition.items()}


def _ceil_significant(value: float, digits: int = 2) -> float:
    """Round *value* UP to *digits* significant figures."""
    if value <= 0.0:
        return value
    scale = 10.0 ** (math.floor(math.log10(value)) - digits + 1)
    return math.ceil(value / scale) * scale


def required_dilution(composition: dict[str, float]) -> float:
    """Dilution needed to reach :data:`DILUTE_SEPARATION_RADII`.

    Separation goes as ``phi^(-1/3)`` and dilution scales ``phi`` directly, so
    the factor is the cube of the shortfall in separation.

    Rounded UP to two significant figures. Partly because nobody dilutes by
    108.4678, and partly because the exact factor lands on the limit rather than
    past it --- in floating point that reads as still-not-dilute, so the number
    this returns would fail its own test.
    """
    separation = mean_separation_radii(composition)
    if separation >= DILUTE_SEPARATION_RADII:
        return 1.0
    return _ceil_significant((DILUTE_SEPARATION_RADII / separation) ** 3)


@dataclass
class DilutionReport:
    """How far a composition sits from a dilute suspension."""

    volume_fraction: float
    mean_separation_radii: float
    required_dilution: float
    is_dilute: bool

    def __str__(self) -> str:
        if self.is_dilute:
            return (
                f"dilute: {self.volume_fraction * 100:.3g} % cells by volume, "
                f"{self.mean_separation_radii:.1f} radii apart"
            )
        return (
            f"NOT dilute: {self.volume_fraction * 100:.3g} % cells by volume, only "
            f"{self.mean_separation_radii:.1f} radii apart. Dilute about "
            f"{self.required_dilution:.0f}x to reach "
            f"{DILUTE_SEPARATION_RADII:.0f} radii"
        )


def dilution_report(composition: dict[str, float]) -> DilutionReport:
    """Whether *composition* is dilute enough for a non-interacting model."""
    separation = mean_separation_radii(composition)
    return DilutionReport(
        volume_fraction=volume_fraction(composition),
        mean_separation_radii=separation,
        required_dilution=required_dilution(composition),
        is_dilute=bool(separation >= DILUTE_SEPARATION_RADII),
    )


#: Fraction of red cells removed by isotonic ammonium-chloride lysis.
RBC_LYSIS_EFFICIENCY = 0.999

RBC_LYSIS_PROVENANCE = Provenance(
    assumption="Isotonic ammonium-chloride lysis (e.g. BD Pharm Lyse) is quoted as "
    "removing essentially all erythrocytes, but published acoustophoresis protocols "
    "state the step without a residual count. 99.9 % is taken as a working figure; "
    "the residual matters because it sets the background this device must then "
    "deplete, so measure it for your own buffer if the result is marginal."
)

LYSIS_PROTOCOL_PROVENANCE = Provenance(
    doi="10.1073/pnas.1504484112",
    citation="Li et al. (2015), PNAS 112:4970 — 'The concentration of WBCs is "
    "obtained by directly lysing 1 mL of human whole blood and resuspending the "
    "cells into 1 mL solution of PBS.' The same step appears in Anal Chem "
    "93:17076 (doi:10.1021/acs.analchem.1c04050), which lyses with BD Pharm Lyse "
    "and then dilutes 1:1",
)


def rbc_lysis(
    composition: dict[str, float],
    *,
    efficiency: float = RBC_LYSIS_EFFICIENCY,
    final_volume_ratio: float = 1.0,
) -> dict[str, float]:
    """Remove red cells by isotonic lysis, the way every published protocol does.

    No acoustophoretic CTC protocol runs whole blood. Both the tilted-angle SAW
    device of Li et al. and the two-step bulk device of Anal Chem 93:17076 lyse
    the erythrocytes first, because they are 99.9 % of the cells and 45 % of the
    sample volume --- which is both an impossible background to deplete and far
    outside any dilute-suspension model.

    Parameters
    ----------
    efficiency:
        Fraction of red cells removed. See :data:`RBC_LYSIS_PROVENANCE`.
    final_volume_ratio:
        Volume after resuspension divided by the volume before. Li et al. lyse
        1 mL and resuspend in 1 mL, so 1.0 and the leukocyte count is unchanged;
        a 1:1 dilution afterwards is 2.0.

    Everything that is not a red cell survives and is then diluted by the
    resuspension, which is why leukocytes barely move at ``final_volume_ratio=1``.
    """
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError(f"lysis efficiency must be in [0, 1], got {efficiency}")
    if final_volume_ratio <= 0.0:
        raise ValueError("final_volume_ratio must be positive")
    out = {}
    for key, value in composition.items():
        surviving = value * (1.0 - efficiency) if key == "rbc" else value
        out[key] = surviving / final_volume_ratio
    return out


def whole_blood_with_ctc(
    ctc_type: str = "mcf7",
    ctc_per_ml: float = CTC_PER_ML_METASTATIC,
) -> dict[str, float]:
    """Whole blood plus a realistic circulating-tumour-cell burden, per mL."""
    return {**WHOLE_BLOOD, ctc_type: float(ctc_per_ml)}


def compose(
    collection_probability: dict[str, float],
    composition: dict[str, float],
    *,
    targets: set[str],
    volume_ml: float = 1.0,
    simulated_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Apply per-population outlet probabilities to a real sample.

    This is what makes a rare-cell separation answerable. The simulation measures
    ``P(collected)`` for each cell type using however many cells give a usable
    estimate; this then asks what happens to a tube containing the real numbers.

    Assumes each cell's fate is independent of the others --- the same assumption
    the force model makes, and the reason :func:`dilution_report` matters.

    Returns recovery of the targets, depletion of everything else expressed in
    log10 (the way a rare-cell paper reports it), and the purity and enrichment
    that follow.

    Pass *simulated_counts* --- how many of each population were actually
    tracked. Without it a background population that happened to lose every
    simulated cell reports **infinite** depletion and perfect purity, which is
    never true: it means the collection probability is below what that many cells
    can resolve, not that it is zero. With 250 simulated cells nothing under
    about 1 % is measurable, while at 5e7 red cells per mL a probability of 0.1 %
    still puts fifty thousand of them in the collection outlet.

    Given the counts, each probability also gets a Wilson upper bound, and the
    depletion computed from those bounds is reported as
    ``background_depletion_log10_demonstrated``: the most the simulation can
    honestly claim at this ensemble size. Raise the counts to claim more.
    """
    from .statistics import wilson_interval
    collected = {
        k: composition.get(k, 0.0) * volume_ml * collection_probability.get(k, 0.0)
        for k in composition
    }
    upper = dict(collection_probability)
    if simulated_counts:
        for key, n in simulated_counts.items():
            if n > 0:
                hits = int(round(collection_probability.get(key, 0.0) * n))
                upper[key] = wilson_interval(hits, int(n)).high_percent / 100.0
    loaded = {k: composition[k] * volume_ml for k in composition}

    target_in = sum(v for k, v in loaded.items() if k in targets)
    target_out = sum(v for k, v in collected.items() if k in targets)
    background_in = sum(v for k, v in loaded.items() if k not in targets)
    background_out = sum(v for k, v in collected.items() if k not in targets)

    total_out = target_out + background_out
    total_in = target_in + background_in
    purity_in = target_in / total_in if total_in else float("nan")
    purity_out = target_out / total_out if total_out else float("nan")

    with np.errstate(divide="ignore"):
        depletion_log10 = (
            float(np.log10(background_in / background_out))
            if background_out > 0
            else float("inf")
        )

    background_out_upper = sum(
        composition[k] * volume_ml * upper.get(k, 0.0)
        for k in composition
        if k not in targets
    )
    with np.errstate(divide="ignore"):
        demonstrated = (
            float(np.log10(background_in / background_out_upper))
            if background_out_upper > 0
            else float("inf")
        )

    return {
        "volume_ml": volume_ml,
        "loaded": loaded,
        "collected": collected,
        "target_loaded": target_in,
        "target_collected": target_out,
        "background_loaded": background_in,
        "background_collected": background_out,
        "recovery_percent": 100.0 * target_out / target_in if target_in else float("nan"),
        "background_depletion_log10": depletion_log10,
        "background_depletion_log10_demonstrated": demonstrated,
        "background_collected_upper_bound": background_out_upper,
        "purity_in": purity_in,
        "purity_out": purity_out,
        "enrichment_fold": purity_out / purity_in if purity_in else float("nan"),
    }


__all__ = [
    "BLOOD_PROVENANCE",
    "CTC_PER_ML_METASTATIC",
    "CTC_PROVENANCE",
    "DILUTE_LIMIT_PROVENANCE",
    "DILUTE_SEPARATION_RADII",
    "WHOLE_BLOOD",
    "DilutionReport",
    "cell_volume",
    "compose",
    "dilute",
    "dilution_report",
    "mean_separation_radii",
    "required_dilution",
    "volume_fraction",
    "whole_blood_with_ctc",
]
