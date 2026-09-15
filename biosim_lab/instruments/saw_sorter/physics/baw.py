"""Bulk-acoustic-wave (BAW) standing field across a hard-walled channel.

The alternating-frequency sorter (``mode="alternating_baw"``) is a *bulk*
device: a piezoceramic under a silicon/Pyrex chip excites the transverse
half-wave resonances of the channel itself. Unlike an SSAW device, whose node
spacing is imposed by the substrate wavelength, here the fluid resonance sets
it: the ``n``-th mode of a channel of width ``W`` has

    p(y) = p_a * cos(n * pi * y / W),     k_n = n * pi / W,

a pressure antinode on each wall and nodes at ``y = (2m + 1) W / (2n)``. For
``n = 1`` that is the single central node; for ``n = 3`` the three nodes at
``W/6, W/2, 5W/6`` that the alternating-frequency scheme of Zhang et al.
(2023, doi:10.3390/ijms24043338, Sec. 4.2) is built on.

Force and trajectory
--------------------
The primary radiation force on a small sphere in that field is (Bruus 2012,
doi:10.1039/c2lc21068a, eq. 32)

    F_y = 4 * pi * Phi_B * a^3 * k_n * E_ac * sin(2 * k_n * y)

with ``E_ac = p_a^2 / (4 rho_f c_f^2)`` and ``Phi_B = f1/3 + f2/2`` (Bruus'
normalisation, a third of the classical ``Phi``). This is the same law as the
project-wide ``F = -(pi p0^2 V beta_f / 2 lambda) Phi sin(2ky)`` with
``lambda = 2W/n`` and the origin moved from the node to the wall; the tests
check the two agree.

Balancing it against Stokes drag gives a closed-form trajectory (Barnkob et
al. 2010, doi:10.1039/b920376a, eq. 6), used to verify the integrators:

    tan(k_n y(t)) = tan(k_n y0) * exp(t / tau),
    tau = 3 * mu / (4 * Phi_B * (k_n a)^2 * E_ac).

Drive calibration
-----------------
``E_ac`` scales with the square of the transducer voltage at a fixed
frequency (Barnkob et al. 2010, doi:10.1039/b920376a, Fig. 7: ``E_ac ~ U_pp^2``).
The *proportionality constant* is device-specific --- it depends on the
transducer, the glue layer and how close the drive is to resonance --- and has
to be measured or calibrated. See :func:`energy_density_from_voltage`.
"""

from __future__ import annotations

import warnings

import numpy as np

from biosim_lab.core.plugin import RegimeWarning


def mode_number(frequency: float, width: float, sound_speed: float) -> int:
    """Index ``n`` of the transverse half-wave resonance closest to *frequency*.

    Resonance ``n`` sits at ``f_n = n c / (2 W)`` for a channel whose walls are
    acoustically hard (silicon, glass). A drive between two resonances still
    excites the nearer mode, only more weakly --- which the energy-density
    calibration absorbs --- so this warns when the detuning exceeds 15 %.
    """
    if min(frequency, width, sound_speed) <= 0:
        raise ValueError("frequency, width and sound_speed must be positive")
    exact = 2.0 * float(width) * float(frequency) / float(sound_speed)
    n = max(1, int(round(exact)))
    if abs(exact - n) / n > 0.15:
        warnings.warn(
            f"{frequency / 1e6:.3g} MHz is {100 * abs(exact - n) / n:.0f} % away from "
            f"the nearest half-wave resonance (n = {n}, f_n = "
            f"{n * sound_speed / (2 * width) / 1e6:.3g} MHz) of a {width * 1e6:.0f} um "
            "channel; the field model assumes a clean resonance.",
            RegimeWarning,
            stacklevel=2,
        )
    return n


def resonance_frequency(n: int, width: float, sound_speed: float) -> float:
    """``f_n = n c / (2 W)`` [Hz]."""
    return int(n) * float(sound_speed) / (2.0 * float(width))


def wavenumber(n: int, width: float) -> float:
    """``k_n = n pi / W`` [1/m]."""
    return int(n) * np.pi / float(width)


def node_positions(width: float, n: int) -> np.ndarray:
    """Pressure nodes of mode *n*: ``(2m + 1) W / (2n)`` for ``m = 0 .. n-1`` [m]."""
    return (2.0 * np.arange(int(n)) + 1.0) * float(width) / (2.0 * int(n))


def energy_density_from_pressure(p_a: float, rho_f: float, c_f: float) -> float:
    """``E_ac = p_a^2 / (4 rho_f c_f^2)`` [J/m^3] (doi:10.1039/c2lc21068a, eq. 33)."""
    return float(p_a) ** 2 / (4.0 * float(rho_f) * float(c_f) ** 2)


def pressure_from_energy_density(e_ac: float, rho_f: float, c_f: float) -> float:
    """Inverse of :func:`energy_density_from_pressure` [Pa]."""
    return float(np.sqrt(4.0 * float(e_ac) * float(rho_f) * float(c_f) ** 2))


def energy_density_from_voltage(
    voltage_pp: float, *, reference_voltage_pp: float, reference_energy_density: float
) -> float:
    """``E_ac(U) = E_ref * (U / U_ref)^2`` [J/m^3].

    The quadratic law is measured (Barnkob et al. 2010, doi:10.1039/b920376a,
    Fig. 7); the reference pair ``(U_ref, E_ref)`` is a property of one
    particular chip and frequency and must come from a measurement or an
    explicit calibration.
    """
    if reference_voltage_pp <= 0:
        raise ValueError("reference_voltage_pp must be positive")
    return float(reference_energy_density) * (float(voltage_pp) / reference_voltage_pp) ** 2


def radiation_force(
    y: np.ndarray | float,
    *,
    n: int,
    width: float,
    energy_density: np.ndarray | float,
    radius: np.ndarray | float,
    phi_bruus: np.ndarray | float,
) -> np.ndarray:
    """Primary radiation force across the channel [N], positive toward ``+y``.

    ``F_y = 4 pi Phi_B a^3 k_n E_ac sin(2 k_n y)`` (doi:10.1039/c2lc21068a,
    eq. 32), ``y`` measured from the wall. *n* may be an array (one mode per
    particle), which is how a time-switched ensemble with staggered entry times
    is evaluated in one call.
    """
    k = np.asarray(n, dtype=float) * np.pi / float(width)
    a = np.asarray(radius, dtype=float)
    return (
        4.0 * np.pi * np.asarray(phi_bruus, dtype=float) * a**3 * k
        * np.asarray(energy_density, dtype=float)
        * np.sin(2.0 * k * np.asarray(y, dtype=float))
    )


def relaxation_time(
    *, n: int, width: float, energy_density: float, radius: float | np.ndarray,
    phi_bruus: float | np.ndarray, viscosity: float,
) -> np.ndarray:
    """``tau = 3 mu / (4 Phi_B (k_n a)^2 E_ac)`` [s] (doi:10.1039/b920376a, eq. 6)."""
    k = wavenumber(n, width)
    a = np.asarray(radius, dtype=float)
    return 3.0 * float(viscosity) / (
        4.0 * np.asarray(phi_bruus, dtype=float) * (k * a) ** 2 * float(energy_density)
    )


def analytic_position(
    y0: np.ndarray | float,
    t: np.ndarray | float,
    *,
    n: int,
    width: float,
    energy_density: float,
    radius: float | np.ndarray,
    phi_bruus: float | np.ndarray,
    viscosity: float,
) -> np.ndarray:
    """Exact overdamped trajectory in a single mode (doi:10.1039/b920376a, eq. 6).

    Within one antinode-to-antinode cell ``[m W/n, (m+1) W/n]`` the local
    coordinate ``s = y - m W / n`` obeys ``tan(k s) = tan(k s0) exp(t / tau)``,
    which carries the particle monotonically onto that cell's node. A particle
    starting exactly on an antinode stays there (unstable equilibrium).
    """
    y0 = np.asarray(y0, dtype=float)
    t = np.asarray(t, dtype=float)
    k = wavenumber(n, width)
    cell = width / n
    m = np.floor(y0 / cell)
    s0 = y0 - m * cell
    tau = relaxation_time(
        n=n, width=width, energy_density=energy_density, radius=radius,
        phi_bruus=phi_bruus, viscosity=viscosity,
    )
    arg = np.tan(k * s0) * np.exp(t / tau)
    # arctan returns (-pi/2, pi/2); a start past the node (tan < 0) maps back
    # into (pi/2, pi) so the particle approaches the node from above.
    ks = np.arctan(arg)
    ks = np.where(ks < 0.0, ks + np.pi, ks)
    return m * cell + ks / k


def displacement_in(
    duration: float,
    y0: float | np.ndarray,
    *,
    n: int,
    width: float,
    energy_density: float,
    radius: float | np.ndarray,
    phi_bruus: float | np.ndarray,
    viscosity: float,
) -> np.ndarray:
    """Distance travelled across the channel in *duration* seconds [m].

    The quantity in the separation rule of Zhang et al. 2023
    (doi:10.3390/ijms24043338, Sec. 4.2): during the 1 MHz phase a target cell
    must move more than ``W/6`` and a background cell less.
    """
    y1 = analytic_position(
        y0, duration, n=n, width=width, energy_density=energy_density, radius=radius,
        phi_bruus=phi_bruus, viscosity=viscosity,
    )
    return np.abs(y1 - np.asarray(y0, dtype=float))


__all__ = [
    "mode_number",
    "resonance_frequency",
    "wavenumber",
    "node_positions",
    "energy_density_from_pressure",
    "pressure_from_energy_density",
    "energy_density_from_voltage",
    "radiation_force",
    "relaxation_time",
    "analytic_position",
    "displacement_in",
]
