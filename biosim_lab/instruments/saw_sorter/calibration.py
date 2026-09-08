"""Measuring the acoustic field from bead trajectories.

The largest unsourced input in this project is the step from IDT drive voltage
to acoustic pressure. ``PRESSURE_PER_VOLT_ASSUMPTION`` in
:mod:`~biosim_lab.instruments.saw_sorter.fem_model` is a straight line anchored
at 0.45 MPa for 15 Vpp, flagged in capitals, and it scales *every* acoustic
force the platform computes.

This module is how you get rid of it. Put calibration beads of known size and
material through your own chip, film them, and fit the acoustic energy density
from how they move. That is the standard experimental method, and it needs no
knowledge of the transducer at all.

Reference: Barnkob, Augustsson, Laurell & Bruus (2010), *Measuring the local
pressure amplitude in microchannel acoustophoresis*, Lab Chip 10:563,
doi:10.1039/b920376a.

The trajectory has a closed form
--------------------------------
In the overdamped limit a particle in a one-dimensional standing wave obeys

    dx/dt = -u_max sin(2k(x - x_node))

which is separable. Substituting ``xi = k(x - x_node)`` and integrating
``d(xi)/(sin xi cos xi) = d ln|tan xi|`` gives

    tan(k(x(t) - x_node)) = tan(k(x_0 - x_node)) * exp(-2 k u_max t)

so the whole trajectory is an exact expression with **one** free parameter,
``u_max``. Fitting it to measured positions gives the acoustic energy density
directly:

    u_max = (2/9) * a^2 * Phi * E_ac * k / mu

This matters twice over. It is the calibration route, and it is also a second,
completely independent analytic solution to check the numerical trajectory
integrator against --- one that exercises the *whole* path, not a single point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def energy_density(pressure_amplitude: float, kappa_f: float) -> float:
    """Acoustic energy density ``E_ac = p0^2 kappa_f / 4`` [J/m^3]."""
    return 0.25 * float(pressure_amplitude) ** 2 * float(kappa_f)


def pressure_amplitude(energy_density_value: float, kappa_f: float) -> float:
    """Inverse of :func:`energy_density`: ``p0 = sqrt(4 E_ac / kappa_f)`` [Pa].

    Raises on a negative energy density rather than returning ``nan``. A
    negative value only arises from a fit that has failed, and a silent ``nan``
    propagates into plots and summaries as a blank rather than as an error.
    """
    value = float(energy_density_value)
    if value < 0.0:
        raise ValueError(
            f"energy density is negative ({value:.4g} J/m^3); a pressure "
            "amplitude cannot be recovered from it"
        )
    return float(np.sqrt(4.0 * value / float(kappa_f)))


def acoustophoretic_velocity(
    *,
    radius: float,
    phi: float,
    energy_density_value: float,
    wavelength: float,
    viscosity: float,
) -> float:
    """Peak lateral migration speed ``u_max`` [m/s].

    ``u_max = (2/9) a^2 Phi E_ac k / mu`` with ``k = 2 pi / lambda`` and *phi*
    the **classical** contrast factor (three times the Bruus normalisation ---
    in Bruus' own convention the prefactor is 2/3 rather than 2/9, and the two
    forms are identical).

    Derivation: the peak radiation force is ``V Phi E_ac k``; divide by the
    Stokes drag coefficient ``6 pi mu a``.
    """
    k = 2.0 * np.pi / float(wavelength)
    return float(
        (2.0 / 9.0) * float(radius) ** 2 * float(phi)
        * float(energy_density_value) * k / float(viscosity)
    )


def analytic_trajectory(
    times: np.ndarray,
    *,
    x_initial: float,
    u_max: float,
    wavelength: float,
    node_offset: float = 0.0,
) -> np.ndarray:
    """Exact particle path in a one-dimensional standing wave [m].

    ``x(t) = x_node + arctan( tan(k (x_0 - x_node)) exp(-2 k u_max t) ) / k``

    Exact for the overdamped equation of motion, with no discretisation of any
    kind --- which is what makes it useful as a reference for the numerical
    integrator over a whole trajectory rather than at one instant.
    """
    k = 2.0 * np.pi / float(wavelength)
    times = np.asarray(times, dtype=float)
    xi0 = k * (float(x_initial) - float(node_offset))
    # Wrap into (-pi/2, pi/2): tan has period pi and the particle cannot cross a
    # pressure antinode, so this identifies which half-wavelength cell it is in.
    cell = np.round(xi0 / np.pi)
    xi0_wrapped = xi0 - cell * np.pi
    xi = np.arctan(np.tan(xi0_wrapped) * np.exp(-2.0 * k * float(u_max) * times))
    return float(node_offset) + (xi + cell * np.pi) / k


@dataclass
class CalibrationResult:
    """Acoustic field strength fitted from an observed trajectory."""

    u_max: float
    energy_density: float
    pressure_amplitude: float
    u_max_stderr: float
    r_squared: float
    n_points: int

    def __str__(self) -> str:  # pragma: no cover - reporting aid
        return (f"E_ac = {self.energy_density:.3g} J/m^3, "
                f"p0 = {self.pressure_amplitude / 1e6:.3f} MPa "
                f"(u_max = {self.u_max * 1e6:.1f} um/s, R^2 = {self.r_squared:.4f})")


def fit_energy_density(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    radius: float,
    phi: float,
    wavelength: float,
    kappa_f: float,
    viscosity: float,
    node_offset: float = 0.0,
    fit_initial_position: bool = True,
) -> CalibrationResult:
    """Fit the acoustic energy density to one measured bead trajectory.

    This is the calibration that replaces the voltage assumption. Track a bead
    of known radius and material, feed the ``(t, x)`` pairs in, and read the
    field strength out.

    Parameters
    ----------
    times, positions:
        Observed times [s] and lateral positions [m], one bead.
    radius, phi:
        The bead's radius [m] and **classical** contrast factor. Polystyrene in
        water is the usual choice because both are known precisely.
    wavelength, node_offset:
        Geometry of the standing wave [m].
    kappa_f, viscosity:
        Fluid compressibility [1/Pa] and dynamic viscosity [Pa*s], **at the
        temperature of the experiment**. The fitted energy density is directly
        proportional to the viscosity, and water's moves 22 % between a bench
        and an incubator, so a calibration quoted without its temperature is
        incomplete.
    fit_initial_position:
        Also fit ``x_0`` rather than trusting the first frame, which carries the
        same tracking noise as every other frame.

    How the fit works, and why not the obvious way
    ----------------------------------------------
    Rearranging the closed form gives a straight line,

        ln|tan(k(x - x_node))| = ln|tan(k(x_0 - x_node))| - 2 k u_max t

    which is tempting because it needs no starting guess. It is also **biased**,
    badly, on real data. A bead reaches the node in a fraction of the recording,
    and every frame after that sits within tracking noise of it --- where
    ``d ln|tan| / dx`` diverges. In a typical track, two thirds of the frames
    are in that region and a 0.3 um tracking error is amplified by five orders
    of magnitude. Fitting them equally underestimates the field by ~40 %.

    So the linear form is used only for a starting guess, from the frames that
    actually carry information, and the fit itself is ordinary least squares on
    ``x(t)`` --- the space the noise is actually in, where it is homoscedastic
    and every frame is worth the same.
    """
    from scipy.optimize import curve_fit

    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if times.size != positions.size:
        raise ValueError("times and positions must have the same length")
    if times.size < 3:
        raise ValueError("need at least three points to fit a trajectory")

    k = 2.0 * np.pi / float(wavelength)
    xi = k * (positions - float(node_offset))
    cell = np.round(xi[0] / np.pi)
    tangent = np.tan(xi - cell * np.pi)

    # Starting guess from the frames that are neither on the node (tan -> 0) nor
    # on an antinode (tan -> infinity); both are uninformative about the speed.
    informative = np.isfinite(tangent) & (np.abs(tangent) > 1e-3) & (np.abs(tangent) < 1e3)
    if informative.sum() < 2:
        raise ValueError(
            "fewer than two informative points: the bead sits at the node or the "
            "antinode for the whole track, so its speed is unconstrained"
        )
    slope = np.polyfit(times[informative], np.log(np.abs(tangent[informative])), 1)[0]
    u_guess = float(-slope / (2.0 * k))

    def model(t: np.ndarray, u: float, x0: float) -> np.ndarray:
        return analytic_trajectory(
            t, x_initial=x0, u_max=u, wavelength=wavelength, node_offset=node_offset
        )

    x0_guess = float(positions[0])
    if fit_initial_position:
        popt, pcov = curve_fit(model, times, positions, p0=[u_guess, x0_guess],
                               maxfev=20000)
        u_max, _ = float(popt[0]), float(popt[1])
        u_stderr = float(np.sqrt(abs(pcov[0, 0])))
    else:
        popt, pcov = curve_fit(
            lambda t, u: model(t, u, x0_guess), times, positions, p0=[u_guess],
            maxfev=20000,
        )
        u_max = float(popt[0])
        u_stderr = float(np.sqrt(abs(pcov[0, 0])))

    fitted = model(times, u_max, float(popt[1]) if fit_initial_position else x0_guess)
    residual = positions - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((positions - positions.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if u_max <= 0.0:
        # The bead drifted away from the node instead of toward it. Physically
        # that means the assumed node position is wrong, the field was off, or
        # the track is pure noise -- never a negative energy density.
        raise ValueError(
            f"fitted migration speed is not positive ({u_max * 1e6:.4g} um/s): "
            "the bead does not move toward the assumed node, so no energy "
            "density can be recovered from this track"
        )

    e_ac = 9.0 * float(viscosity) * u_max / (2.0 * float(radius) ** 2 * float(phi) * k)
    return CalibrationResult(
        u_max=u_max,
        energy_density=e_ac,
        pressure_amplitude=pressure_amplitude(e_ac, kappa_f),
        u_max_stderr=u_stderr,
        r_squared=r_squared,
        n_points=int(times.size),
    )


def calibrate_from_tracks(
    tracks: Any,
    *,
    radius: float,
    phi: float,
    wavelength: float,
    kappa_f: float,
    viscosity: float,
    node_offset: float = 0.0,
    time_column: str = "time",
    position_column: str = "x",
    particle_column: str = "particle",
) -> Any:
    """Fit every bead in a track table and return a DataFrame of results.

    Accepts the long-format table ``cell_tracker`` produces, so a real
    calibration video can go straight from the tracker into a field
    measurement. The spread across beads is the honest error bar on the
    calibration --- a single bead's fit looks far more precise than the
    calibration actually is.
    """
    import pandas as pd

    rows = []
    rejected = 0
    for particle, group in tracks.groupby(particle_column):
        ordered = group.sort_values(time_column)
        try:
            fit = fit_energy_density(
                ordered[time_column].to_numpy(),
                ordered[position_column].to_numpy(),
                radius=radius, phi=phi, wavelength=wavelength,
                kappa_f=kappa_f, viscosity=viscosity, node_offset=node_offset,
            )
        except ValueError:
            rejected += 1
            continue
        rows.append({
            "particle": particle,
            "u_max_um_s": fit.u_max * 1e6,
            "energy_density_J_m3": fit.energy_density,
            "pressure_amplitude_MPa": fit.pressure_amplitude / 1e6,
            "r_squared": fit.r_squared,
            "n_points": fit.n_points,
        })
    table = pd.DataFrame(rows)
    # Beads whose fit failed are dropped, but silently dropping them would make
    # a calibration built on two usable tracks look as solid as one built on
    # fifty, so the count travels with the table.
    table.attrs["n_rejected"] = rejected
    table.attrs["n_accepted"] = len(rows)
    return table


__all__ = [
    "CalibrationResult",
    "energy_density",
    "pressure_amplitude",
    "acoustophoretic_velocity",
    "analytic_trajectory",
    "fit_energy_density",
    "calibrate_from_tracks",
]
