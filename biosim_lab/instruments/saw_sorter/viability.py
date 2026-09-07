"""Do the cells survive the device, and which ones were already dead?

The selling point of acoustic sorting is that it is gentle: no antibodies, no
labels, no contact. ``Gentle`` is a claim, and a claim is something to compute.
This module computes the three ways a cell can be harmed in this device, using
published damage thresholds, and reports where the operating point sits relative
to each.

The three mechanisms
--------------------
**Heat.** Damage accumulates non-linearly with temperature, so ``2 minutes at
45 °C`` is not comparable to ``4 minutes at 44 °C`` by any simple ratio. The
standard currency is the *cumulative equivalent minutes at 43 °C*:

    CEM43 = sum over time of  R^(43 - T) * dt      R = 0.25 below 43 °C, 0.5 above

Reference: Sapareto & Dewey (1984), *Thermal dose determination in cancer
therapy*, Int. J. Radiat. Oncol. Biol. Phys. 10:787,
doi:10.1016/0360-3016(84)90379-1. Review of thresholds across cell types:
van Rhoon et al. (2013), Eur. Radiol. 23:2215, doi:10.1007/s00330-013-2825-y.

**Shear.** The flow drags on the membrane. Erythrocytes are the classic
reference case and begin to haemolyse above roughly 150 Pa of sustained shear;
nucleated cells are more fragile in long exposures but tolerate brief peaks.
Reference: Leverett et al. (1972), *Red blood cell damage by shear stress*,
Biophys. J. 12:257, doi:10.1016/S0006-3495(72)86085-5.

**Cavitation.** If the negative half-cycle of the pressure is strong enough it
can tear the liquid apart around a nucleus, and the collapse destroys anything
nearby. The standard indicator is the mechanical index,
``MI = p_negative[MPa] / sqrt(f[MHz])``; diagnostic ultrasound is capped at 1.9
by regulators. Reference: Apfel & Holland (1991), *Gauging the likelihood of
cavitation from short-pulse, low-duty cycle diagnostic ultrasound*, Ultrasound
Med. Biol. 17:179, doi:10.1016/0301-5629(91)90125-Q.

What this module does not model
-------------------------------
Membrane poration below the lysis threshold (sonoporation), which can let dye
in without killing the cell and therefore corrupts a trypan-blue readout; and
any change in a cell's *acoustic* properties once it dies. Dead cells are
tracked through the device with the same density and compressibility as live
ones --- an explicit assumption, flagged below --- so that the question ``do
dead cells contaminate my collection outlet?`` can still be asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CEM43_REFERENCE_C = 43.0
"""Reference temperature of the thermal-dose scale [°C]."""

CEM43_DAMAGE_THRESHOLD_MIN = 15.0
"""Thermal dose at which about half of a cultured population is lost [CEM43 min].

**ASSUMPTION** within a sourced range: van Rhoon et al. (doi:10.1007/s00330-013-2825-y)
collate thresholds from a few CEM43 minutes for the most sensitive tissues to
several hundred for the most resistant. 15 minutes is a mid-range value for
cultured mammalian cells. Nothing in the default operating point comes close to
it, so the result is insensitive to the exact figure --- but change the number
if you are deliberately running hot.
"""

SHEAR_LYSIS_THRESHOLD_PA = 150.0
"""Sustained shear stress at which erythrocytes begin to haemolyse [Pa].

Leverett et al. (1972), doi:10.1016/S0006-3495(72)86085-5. Nucleated cells
differ; see :func:`shear_survival`.
"""

MECHANICAL_INDEX_LIMIT = 1.9
"""Regulatory cap on the mechanical index for diagnostic ultrasound.

US FDA guidance for diagnostic ultrasound devices. A microfluidic chip is not a
diagnostic scanner, but the index is the standard way to state how close a field
is to the cavitation threshold, so it is the right yardstick to report.
"""

DEAD_CELL_ACOUSTICS_ASSUMPTION = (
    "Dead cells are propagated with the same density and compressibility as live "
    "ones. Necrotic cells lose membrane integrity and shrink, so their real "
    "acoustic contrast differs, but no measurement was found to quantify it. "
    "The consequence is that the predicted destination of dead cells is less "
    "trustworthy than that of live ones."
)


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------


def cem43(temperature_c: float | np.ndarray, seconds: float | np.ndarray) -> np.ndarray:
    """Cumulative equivalent minutes at 43 °C for a constant-temperature exposure.

    ``CEM43 = R^(43 - T) * t_minutes`` with ``R = 0.25`` below 43 °C and
    ``R = 0.5`` at or above it. The break in ``R`` is empirical: above 43 °C
    protein denaturation dominates and the dose accumulates twice as fast per
    degree.

    doi:10.1016/0360-3016(84)90379-1
    """
    t = np.asarray(temperature_c, dtype=float)
    minutes = np.asarray(seconds, dtype=float) / 60.0
    r = np.where(t >= CEM43_REFERENCE_C, 0.5, 0.25)
    return np.asarray(r ** (CEM43_REFERENCE_C - t) * minutes, dtype=float)


def thermal_survival(
    dose_cem43: float | np.ndarray, *, threshold: float = CEM43_DAMAGE_THRESHOLD_MIN
) -> np.ndarray:
    """Surviving fraction from a thermal dose, as an exponential kill curve.

    ``S = exp(-ln(2) * dose / threshold)``, i.e. *threshold* CEM43 minutes
    halves the population. An exponential is the simplest form consistent with
    single-hit killing and does not pretend to a shoulder the data would not
    support.
    """
    dose = np.asarray(dose_cem43, dtype=float)
    return np.asarray(np.exp(-np.log(2.0) * dose / float(threshold)), dtype=float)


# ---------------------------------------------------------------------------
# shear
# ---------------------------------------------------------------------------


def wall_shear_stress(flow: Any, *, n_samples: int = 401) -> dict[str, float]:
    """Peak and mean wall shear stress in the channel [Pa].

    ``tau = mu * du/dn`` at the wall, evaluated by differentiating the exact
    Poiseuille series along both wall normals.

    Parameters
    ----------
    flow:
        A :class:`~biosim_lab.instruments.saw_sorter.flow.RectangularPoiseuille`.
    """
    mu = flow.viscosity
    w, h = flow.width, flow.height
    eps = min(w, h) * 1e-4

    # Along the long walls (y = 0 and y = h), differentiate in y.
    xs = np.linspace(0.05 * w, 0.95 * w, n_samples)
    du_dy = (flow.velocity(xs, np.full_like(xs, eps))
             - flow.velocity(xs, np.zeros_like(xs))) / eps
    # Along the short walls (x = 0 and x = w), differentiate in x.
    ys = np.linspace(0.05 * h, 0.95 * h, n_samples)
    du_dx = (flow.velocity(np.full_like(ys, eps), ys)
             - flow.velocity(np.zeros_like(ys), ys)) / eps

    tau_long = mu * np.abs(du_dy)
    tau_short = mu * np.abs(du_dx)
    return {
        "peak_wall_shear_Pa": float(max(tau_long.max(), tau_short.max())),
        "mean_wall_shear_Pa": float(0.5 * (tau_long.mean() + tau_short.mean())),
    }


def shear_at(flow: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Local shear-rate magnitude times viscosity [Pa] at cell positions.

    A cell experiences the shear where it actually is, which for a focused cell
    in the middle of the channel is far below the wall value.
    """
    w, h = flow.width, flow.height
    eps = min(w, h) * 1e-4
    x = np.clip(np.asarray(x, dtype=float), eps, w - eps)
    y = np.clip(np.asarray(y, dtype=float), eps, h - eps)
    du_dx = (flow.velocity(x + eps, y) - flow.velocity(x - eps, y)) / (2 * eps)
    du_dy = (flow.velocity(x, y + eps) - flow.velocity(x, y - eps)) / (2 * eps)
    return np.asarray(flow.viscosity * np.hypot(du_dx, du_dy), dtype=float)


def shear_survival(
    stress_pa: float | np.ndarray,
    seconds: float | np.ndarray,
    *,
    threshold_pa: float = SHEAR_LYSIS_THRESHOLD_PA,
    reference_seconds: float = 120.0,
) -> np.ndarray:
    """Surviving fraction after a shear exposure.

    Damage is treated as a stress--time product relative to the erythrocyte
    reference point (150 Pa sustained for 2 minutes halves the population,
    doi:10.1016/S0006-3495(72)86085-5), with a quadratic stress dependence,
    which is the form usually fitted to haemolysis data.

    **ASSUMPTION**: the exponent and the reference exposure time are taken from
    the erythrocyte literature and applied to every cell type. Nucleated cells
    are generally more fragile in long exposures. In this device the shear is
    two to three orders of magnitude below the threshold, so the model returns
    essentially 1 regardless --- which is the honest answer, and the reason this
    approximation is tolerable here and would not be in a pump or a needle.
    """
    stress = np.asarray(stress_pa, dtype=float)
    exposure = np.asarray(seconds, dtype=float)
    dose = (stress / float(threshold_pa)) ** 2 * (exposure / float(reference_seconds))
    return np.asarray(np.exp(-np.log(2.0) * dose), dtype=float)


# ---------------------------------------------------------------------------
# acoustic exposure
# ---------------------------------------------------------------------------


def mechanical_index(pressure_amplitude: float, frequency: float) -> float:
    """``MI = p_negative[MPa] / sqrt(f[MHz])`` (dimensionless).

    Below ~0.7 cavitation is not expected in the absence of contrast agents;
    diagnostic ultrasound is capped at 1.9.
    doi:10.1016/0301-5629(91)90125-Q
    """
    return float((pressure_amplitude / 1e6) / np.sqrt(frequency / 1e6))


def acoustic_intensity(pressure_amplitude: float, fluid: Any) -> float:
    """Equivalent travelling-wave intensity of the standing field [W/m^2].

    A standing wave carries no *net* intensity --- the two counter-propagating
    components cancel. What matters for absorption and heating is the sum of
    their magnitudes, ``I = p0^2 / (4 * rho * c)``, which is what is reported.
    """
    return float(pressure_amplitude**2 / (4.0 * fluid.rho * fluid.c))


# ---------------------------------------------------------------------------
# putting it together
# ---------------------------------------------------------------------------


@dataclass
class ViabilityReport:
    """Per-cell survival plus the indicators that produced it."""

    alive_at_inlet: np.ndarray
    alive_at_outlet: np.ndarray
    survival_probability: np.ndarray
    thermal_dose_cem43: np.ndarray
    shear_stress_pa: np.ndarray
    indicators: dict[str, Any]

    @property
    def viability_in_percent(self) -> float:
        n = self.alive_at_inlet.size
        return float(100.0 * self.alive_at_inlet.sum() / n) if n else float("nan")

    @property
    def viability_out_percent(self) -> float:
        n = self.alive_at_outlet.size
        return float(100.0 * self.alive_at_outlet.sum() / n) if n else float("nan")

    @property
    def killed_by_device_percent(self) -> float:
        """Percentage points of viability lost between inlet and outlet."""
        return self.viability_in_percent - self.viability_out_percent


def assess(
    *,
    temperature_c: float,
    residence_time_s: np.ndarray,
    shear_stress_pa: np.ndarray,
    pressure_amplitude: float,
    frequency: float,
    fluid: Any,
    inlet_viability: float = 0.95,
    rng: np.random.Generator | None = None,
    thermal_threshold: float = CEM43_DAMAGE_THRESHOLD_MIN,
) -> ViabilityReport:
    """Decide which cells arrive alive and which the device kills.

    Parameters
    ----------
    temperature_c:
        Fluid temperature during transit.
    residence_time_s, shear_stress_pa:
        Per-cell arrays; cells near a wall move slowly and shear harder, so both
        vary across the population.
    inlet_viability:
        Fraction of the sample that was already alive before it reached the
        device. A freshly prepared suspension is typically 90-97 % viable, so
        the honest baseline is not 100 %.

    Notes
    -----
    Killing is applied stochastically from the computed survival probability,
    using the run's own seeded generator, so a cell's fate is reproducible and
    the population statistics carry the right binomial noise.
    """
    rng = rng or np.random.default_rng()
    residence = np.asarray(residence_time_s, dtype=float)
    shear = np.asarray(shear_stress_pa, dtype=float)
    n = residence.size

    # Cells that never reached the outlet have a NaN residence time; charge them
    # the longest observed transit rather than silently exempting them.
    finite = np.isfinite(residence)
    residence = np.where(finite, residence, np.nanmax(residence) if finite.any() else 0.0)

    dose = cem43(temperature_c, residence)
    survival = thermal_survival(dose, threshold=thermal_threshold) * shear_survival(
        shear, residence
    )
    survival = np.clip(survival, 0.0, 1.0)

    alive_in = rng.random(n) < float(inlet_viability)
    alive_out = alive_in & (rng.random(n) < survival)

    indicators = {
        "temperature_C": float(temperature_c),
        "thermal_dose_cem43_max": float(dose.max()) if n else float("nan"),
        "thermal_dose_threshold_cem43": float(thermal_threshold),
        "peak_shear_on_a_cell_Pa": float(shear.max()) if n else float("nan"),
        "shear_lysis_threshold_Pa": SHEAR_LYSIS_THRESHOLD_PA,
        "mechanical_index": mechanical_index(pressure_amplitude, frequency),
        "mechanical_index_limit": MECHANICAL_INDEX_LIMIT,
        "acoustic_intensity_W_cm2": acoustic_intensity(pressure_amplitude, fluid) / 1e4,
        "inlet_viability_percent": 100.0 * float(inlet_viability),
        "dead_cell_acoustics": DEAD_CELL_ACOUSTICS_ASSUMPTION,
    }
    indicators["thermal_margin"] = (
        float(thermal_threshold / dose.max()) if n and dose.max() > 0 else float("inf")
    )
    indicators["shear_margin"] = (
        float(SHEAR_LYSIS_THRESHOLD_PA / shear.max()) if n and shear.max() > 0
        else float("inf")
    )
    indicators["cavitation_margin"] = (
        MECHANICAL_INDEX_LIMIT / indicators["mechanical_index"]
        if indicators["mechanical_index"] > 0 else float("inf")
    )

    return ViabilityReport(
        alive_at_inlet=alive_in,
        alive_at_outlet=alive_out,
        survival_probability=survival,
        thermal_dose_cem43=dose,
        shear_stress_pa=shear,
        indicators=indicators,
    )


__all__ = [
    "CEM43_REFERENCE_C",
    "CEM43_DAMAGE_THRESHOLD_MIN",
    "SHEAR_LYSIS_THRESHOLD_PA",
    "MECHANICAL_INDEX_LIMIT",
    "DEAD_CELL_ACOUSTICS_ASSUMPTION",
    "ViabilityReport",
    "cem43",
    "thermal_survival",
    "wall_shear_stress",
    "shear_at",
    "shear_survival",
    "mechanical_index",
    "acoustic_intensity",
    "assess",
]
