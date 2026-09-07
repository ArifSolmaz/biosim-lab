"""Environmental conditions, and how they change the physics.

Why this module exists
----------------------
Every material constant in :mod:`biosim_lab.core.materials` is quoted at a
single temperature (25 °C for the aqueous media). That is fine for a look-up
table and wrong for a simulation, because the two properties the sorter depends
on most are strongly temperature dependent:

======================  ==========  ==========  =========
Property                 25 °C       37 °C       change
======================  ==========  ==========  =========
viscosity  ``mu``        0.890 mPa·s 0.692 mPa·s  −22 %
speed of sound ``c``     1496.7 m/s  1523.7 m/s   +1.8 %
density ``rho``          997.0 kg/m³ 993.3 kg/m³  −0.4 %
======================  ==========  ==========  =========

Acoustophoretic velocity is ``F / (6 pi mu r)``, so that viscosity drop alone
makes every cell migrate **29 % faster** at body temperature than on the bench.
A device tuned at room temperature and then run in an incubator is not the same
device. Before this module existed the platform silently assumed 25 °C
everywhere; now temperature is an explicit input and its consequences propagate.

Correlations
------------
All three are the standard pure-water correlations, and all three reproduce the
values in :mod:`~biosim_lab.core.materials` at 25 °C to four significant
figures --- which is the check that they were transcribed correctly
(``tests/test_environment.py``).

``density``
    Kell (1975), *Density, thermal expansivity, and compressibility of liquid
    water from 0° to 150 °C*, J. Chem. Eng. Data 20:97,
    doi:10.1021/je60064a005.
``speed_of_sound``
    Marczak (1997), *Water as a standard in the measurements of speed of sound
    in liquids*, JASA 102:2776, doi:10.1121/1.420332.
``viscosity``
    The Swindells/CRC correlation relative to 20 °C, consistent with the
    reference formulation of Kestin, Sokolov & Wakeham (1978), JPCRD 7:941,
    doi:10.1063/1.555581.

Solutes shift these slightly. Cell-culture medium is ~0.15 M salt plus protein,
which raises viscosity by a few percent and the speed of sound by ~10 m/s; the
correlations here are for **pure water** and that offset is carried as a
constant ratio from the medium's tabulated 25 °C values.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

from biosim_lab.core.materials import Fluid, Provenance, Value
from biosim_lab.core.plugin import RegimeWarning

#: Valid range of the pure-water correlations [°C].
WATER_T_MIN, WATER_T_MAX = 0.0, 95.0

KELVIN = 273.15
WATER_SPECIFIC_HEAT = 4180.0
"""Specific heat capacity of water [J/(kg*K)] near room temperature."""

ABSORPTION_DB_CM_MHZ2 = 2.2e-3
"""Ultrasonic absorption of pure water, ``alpha / f^2`` [dB/(cm*MHz^2)] at 20 °C.

Water is exceptionally transparent to ultrasound --- soft tissue is about 200
times more absorbing. Reference: Pinkerton (1949), *The absorption of
ultrasonic waves in liquids and its relation to molecular constitution*,
Proc. Phys. Soc. B 62:129, doi:10.1088/0370-1301/62/2/307.
"""

NEPER_PER_DB = 1.0 / 8.685889638


def _check_range(temperature_c: float) -> None:
    if not (WATER_T_MIN <= temperature_c <= WATER_T_MAX):
        warnings.warn(
            f"temperature {temperature_c:.1f} °C is outside the range "
            f"{WATER_T_MIN:.0f}-{WATER_T_MAX:.0f} °C over which these water "
            "correlations were fitted; the values returned are extrapolations.",
            RegimeWarning,
            stacklevel=3,
        )


def water_density(temperature_c: float) -> float:
    """Density of pure water [kg/m^3] at 1 atm.

    Kell (1975), doi:10.1021/je60064a005.
    """
    _check_range(temperature_c)
    t = float(temperature_c)
    numerator = (
        999.83952
        + 16.945176 * t
        - 7.9870401e-3 * t**2
        - 46.170461e-6 * t**3
        + 105.56302e-9 * t**4
        - 280.54253e-12 * t**5
    )
    return float(numerator / (1.0 + 16.87985e-3 * t))


def water_sound_speed(temperature_c: float) -> float:
    """Speed of sound in pure water [m/s] at 1 atm.

    Marczak (1997), doi:10.1121/1.420332.
    """
    _check_range(temperature_c)
    t = float(temperature_c)
    return float(
        1.402385e3
        + 5.038813 * t
        - 5.799136e-2 * t**2
        + 3.287156e-4 * t**3
        - 1.398845e-6 * t**4
        + 2.787860e-9 * t**5
    )


def water_viscosity(temperature_c: float) -> float:
    """Dynamic viscosity of pure water [Pa*s].

    ``log10(mu_T / mu_20) = [1.1709(20-T) - 0.001827(T-20)^2] / (T + 89.93)``
    with ``mu_20 = 1.002 mPa*s``. Consistent with Kestin, Sokolov & Wakeham
    (1978), doi:10.1063/1.555581, to better than 0.5 % over 0-100 °C.

    This is the single most consequential temperature dependence in the whole
    platform: acoustophoretic velocity is inversely proportional to it.
    """
    _check_range(temperature_c)
    t = float(temperature_c)
    exponent = (1.1709 * (20.0 - t) - 0.001827 * (t - 20.0) ** 2) / (t + 89.93)
    return float(1.002e-3 * 10.0**exponent)


def water_compressibility(temperature_c: float) -> float:
    """Isentropic compressibility of pure water [1/Pa], ``1/(rho c^2)``."""
    return 1.0 / (water_density(temperature_c) * water_sound_speed(temperature_c) ** 2)


def fluid_at(fluid: Fluid, temperature_c: float) -> Fluid:
    """Return *fluid* with its properties moved to *temperature_c*.

    Pure water is evaluated directly from the correlations. Any other medium is
    scaled by the *ratio* of the water correlation at the target temperature to
    its value at the medium's own tabulated temperature, which preserves the
    solute offset (salt and protein raise viscosity and sound speed) while
    still capturing the temperature dependence.

    The returned :class:`~biosim_lab.core.materials.Fluid` carries provenance
    recording that it was derived, not measured.
    """
    reference_c = fluid.temperature_K - KELVIN
    prov = Provenance(
        assumption=(
            f"derived from the {fluid.key!r} entry at {reference_c:.1f} °C by scaling "
            f"with the pure-water correlation to {temperature_c:.1f} °C; the solute "
            "offset is assumed temperature independent"
        )
    )

    scale_rho = water_density(temperature_c) / water_density(reference_c)
    scale_c = water_sound_speed(temperature_c) / water_sound_speed(reference_c)
    scale_mu = water_viscosity(temperature_c) / water_viscosity(reference_c)

    return replace(
        fluid,
        density=Value(fluid.rho * scale_rho, "kg/m**3", prov),
        speed_of_sound=Value(fluid.c * scale_c, "m/s", prov),
        viscosity=Value(fluid.mu * scale_mu, "Pa*s", prov),
        temperature_K=temperature_c + KELVIN,
    )


# ---------------------------------------------------------------------------
# acoustic absorption and heating
# ---------------------------------------------------------------------------


def absorption_coefficient(frequency: float, temperature_c: float = 25.0) -> float:
    """Pressure absorption coefficient of water ``alpha`` [Np/m].

    ``alpha = 2.2e-3 * (f/MHz)^2`` dB/cm, converted to nepers per metre. The
    quadratic frequency dependence is why megahertz ultrasound is fine in a
    300 um channel and gigahertz ultrasound is not.

    At 6.6 MHz this is 1.1 Np/m, so a wave loses 0.03 % of its amplitude
    crossing a 300 um channel --- which is why the Helmholtz solver treats the
    fluid as lossless and takes its damping parameter from the *device* quality
    factor instead.
    """
    f_mhz = float(frequency) / 1e6
    # Absorption falls with temperature in water (unusually); ~1 %/K near 25 °C.
    thermal_factor = 1.0 - 0.01 * (temperature_c - 20.0)
    db_per_cm = ABSORPTION_DB_CM_MHZ2 * f_mhz**2 * max(thermal_factor, 0.3)
    return float(db_per_cm * 100.0 * NEPER_PER_DB)


def bulk_heating_power_density(
    pressure_amplitude: float, frequency: float, fluid: Fluid
) -> float:
    """Heat deposited in the liquid by sound absorption [W/m^3].

    For a standing wave of amplitude ``p0`` the dissipated power density is

        q = 2 * alpha * c * E_ac = alpha * p0^2 / (2 * rho * c)

    with ``E_ac = p0^2 * kappa / 4`` the acoustic energy density.

    Reference for the energy-density convention: Bruus (2012),
    doi:10.1039/c2lc21068a, eq. 33.
    """
    alpha = absorption_coefficient(frequency, fluid.temperature_K - KELVIN)
    return float(alpha * float(pressure_amplitude) ** 2 / (2.0 * fluid.rho * fluid.c))


#: Transducer heating per watt of applied RF power, **ASSUMPTION**.
#:
#: Real SAW devices warm by several to tens of kelvin, and almost none of it
#: comes from the water absorbing sound (see :func:`thermal_budget`, which
#: computes both and shows the bulk term is a thousand times smaller). The heat
#: comes from resistive loss in the IDT fingers and viscoelastic loss in the
#: substrate and bonding layer, which this project does not model --- that needs
#: the piezoelectric solve that is Stage 4.
#:
#: The coefficient below is an order-of-magnitude stand-in for a chip on a
#: passive heat spreader. Measure your own device with a thermocouple or an
#: infrared camera and set ``temperature`` explicitly instead of relying on it.
TRANSDUCER_HEATING_K_PER_W = 12.0


def thermal_budget(
    *,
    pressure_amplitude: float,
    frequency: float,
    fluid: Fluid,
    channel_width: float,
    channel_height: float,
    channel_length: float,
    flow_rate: float,
    rf_power: float | None = None,
) -> dict[str, float]:
    """Where the heat comes from and how much the liquid warms.

    Two independent contributions are reported:

    ``bulk_absorption_K``
        Computed from first principles: the sound absorbed by the water itself,
        divided by the thermal capacity of the liquid flowing through. In a
        microchannel this is essentially always negligible, and the number is
        reported so that it can be *seen* to be negligible rather than assumed.

    ``transducer_K``
        The term that actually matters in a real device, and the one this
        project cannot compute. Estimated from *rf_power* through
        :data:`TRANSDUCER_HEATING_K_PER_W`, which is an explicit assumption.
        ``None`` for *rf_power* omits it.

    Returns
    -------
    dict
        Power density, both temperature rises, and their sum.
    """
    q = bulk_heating_power_density(pressure_amplitude, frequency, fluid)
    volume = channel_width * channel_height * channel_length
    power = q * volume
    mass_flow = fluid.rho * flow_rate
    bulk_rise = power / (mass_flow * WATER_SPECIFIC_HEAT) if mass_flow > 0 else float("inf")

    transducer_rise = 0.0 if rf_power is None else TRANSDUCER_HEATING_K_PER_W * rf_power

    return {
        "absorption_coefficient_Np_m": absorption_coefficient(
            frequency, fluid.temperature_K - KELVIN
        ),
        "power_density_W_m3": q,
        "absorbed_power_W": power,
        "bulk_absorption_K": bulk_rise,
        "transducer_K": transducer_rise,
        "total_rise_K": bulk_rise + transducer_rise,
    }


# ---------------------------------------------------------------------------
# the environment as a configurable object
# ---------------------------------------------------------------------------

IncubatorPreset = Literal["bench", "incubator", "cold_room", "custom"]

PRESETS: dict[str, dict[str, float]] = {
    #: A microscope bench in an air-conditioned laboratory.
    "bench": {"temperature_c": 25.0},
    #: A standard mammalian-cell incubator.
    "incubator": {"temperature_c": 37.0},
    #: A cold room or a refrigerated stage.
    "cold_room": {"temperature_c": 4.0},
}


def describe_temperature_effect(fluid: Fluid, from_c: float, to_c: float) -> dict[str, float]:
    """Quantify what moving from one temperature to another does.

    Returns the ratios a reader needs to interpret a result taken at a different
    temperature from the one they will run at --- in particular
    ``acoustophoretic_speed_ratio``, which is simply the inverse viscosity
    ratio, and is the number that decides whether a device tuned on the bench
    still works in an incubator.
    """
    cold = fluid_at(fluid, from_c)
    warm = fluid_at(fluid, to_c)
    return {
        "viscosity_ratio": warm.mu / cold.mu,
        "sound_speed_ratio": warm.c / cold.c,
        "density_ratio": warm.rho / cold.rho,
        "compressibility_ratio": warm.kappa / cold.kappa,
        # Force scales with kappa_f; drag scales with mu. Speed is the ratio.
        "acoustophoretic_speed_ratio": (warm.kappa / cold.kappa) * (cold.mu / warm.mu),
    }


__all__ = [
    "KELVIN",
    "WATER_T_MIN",
    "WATER_T_MAX",
    "WATER_SPECIFIC_HEAT",
    "ABSORPTION_DB_CM_MHZ2",
    "TRANSDUCER_HEATING_K_PER_W",
    "PRESETS",
    "water_density",
    "water_sound_speed",
    "water_viscosity",
    "water_compressibility",
    "fluid_at",
    "absorption_coefficient",
    "bulk_heating_power_density",
    "thermal_budget",
    "describe_temperature_effect",
]
