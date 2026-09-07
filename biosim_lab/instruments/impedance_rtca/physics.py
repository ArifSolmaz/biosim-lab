"""Electrical models for cell-substrate impedance sensing (ECIS / RTCA).

Giaever-Keese model
-------------------
A confluent monolayer on a small gold electrode adds impedance by three
independent routes, and the classical treatment solves for them together:

* ``Rb``  — the paracellular (tight-junction) resistance between neighbouring
  cells [Ohm*cm^2];
* ``alpha = r_c * sqrt(rho / h)`` — the constrained current flow in the thin
  electrolyte gap of height ``h`` under a cell of radius ``r_c``
  [Ohm^0.5 * cm];
* ``Cm`` — the specific membrane capacitance [F/cm^2].

The specific impedance of the cell-covered electrode is

    1/Zc = 1/Zn * [ Z_n/(Z_n + Z_m)
                  + (Z_m/(Z_n+Z_m)) * ( gamma*I0(gamma)/... ) ]

with the Bessel-function form given in the original paper.  This module
implements that solution.

Reference: Giaever & Keese (1991), *Micromotion of mammalian cells measured
electrically*, PNAS 88:7896, doi:10.1073/pnas.88.17.7896.  The single-cell
shell dielectric model used by :func:`shell_model_permittivity` follows
Asami (2002), *Characterization of biological cells by dielectric
spectroscopy*, J. Non-Cryst. Solids 305:268, doi:10.1016/S0022-3093(02)01110-9.

Cell Index
----------
The xCELLigence Cell Index is a dimensionless, background-referenced impedance
change,

    CI(t) = ( |Z_cell(t)| - |Z_background| ) / |Z_reference|

where ``Z_reference`` is a nominal 15 Ohm normalisation constant in the RTCA
firmware.  Because the constant is instrument-specific, the value used here is
configurable and its default is flagged ASSUMPTION.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import iv

EPS0 = 8.8541878128e-12
"""Vacuum permittivity [F/m], CODATA 2018 (doi:10.1103/RevModPhys.93.025010)."""

RTCA_REFERENCE_IMPEDANCE_OHM = 15.0
"""ASSUMPTION: the Cell Index normalisation constant.

ACEA/Agilent describe the Cell Index as an impedance change divided by a fixed
nominal impedance; 15 Ohm is the value commonly quoted for the RTCA E-Plate at
10 kHz. Set ``reference_impedance`` explicitly when comparing against a specific
instrument, because this constant only rescales the y-axis.
"""


@dataclass
class ElectrodeGeometry:
    """Gold electrode geometry for one well.

    Attributes
    ----------
    area_cm2:
        Total electrode area [cm^2]. The default 8e-3 cm^2 (0.8 mm^2) is the
        order of magnitude of an RTCA E-Plate 96 gold interdigitated electrode
        — **ASSUMPTION**, since the exact active area is not published.
    cell_radius:
        Radius of an adherent cell footprint [m].
    gap_height:
        Electrolyte gap between the cell's ventral membrane and the gold [m].
        Values of 15-150 nm are reported for adherent lines; the default 100 nm
        is an ASSUMPTION unless fitted.
    """

    area_cm2: float = 8.0e-3
    cell_radius: float = 8.0e-6
    gap_height: float = 100e-9


def electrode_specific_impedance(
    frequency: np.ndarray, *, cpe_q: float = 3.0e-5, cpe_n: float = 0.92
) -> np.ndarray:
    """Specific impedance of the gold/electrolyte interface [Ohm*cm^2].

    ``Z_spec = 1 / (Q * (i*omega)^n)`` with ``Q`` in S*s^n/cm^2.  A
    constant-phase element rather than an ideal capacitor is used because gold
    microelectrodes show ``n ~ 0.85-0.95`` from surface roughness and
    non-uniform current density.

    Reference: Franks et al. (2005), *Impedance characterization and modeling
    of electrodes for biomedical applications*, IEEE TBME 52:1295,
    doi:10.1109/TBME.2005.847523.

    **ASSUMPTION**: the default ``cpe_q`` and ``cpe_n`` sit in the range that
    paper reports for sputtered gold. Fit them to your own blank wells — they
    set the absolute impedance level and therefore the Cell Index scale.

    Everything in this module works in the Giaever-Keese unit system: specific
    impedances in Ohm*cm^2, areas in cm^2, capacitances in F/cm^2.
    """
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    return 1.0 / (cpe_q * (1j * omega) ** cpe_n)


def solution_resistance(area_cm2: float, *, conductivity: float) -> float:
    """Bulk electrolyte spreading resistance of a disc electrode [Ohm].

    For a disc of area ``A`` in an insulating plane, ``R = 1 / (4 * sigma * a)``
    with ``a = sqrt(A/pi)``.

    Reference: Newman (1966), *Resistance for flow of current to a disk*,
    J. Electrochem. Soc. 113:501, doi:10.1149/1.2424009.

    Parameters
    ----------
    area_cm2:
        Electrode area [cm^2].
    conductivity:
        Medium conductivity [S/m].
    """
    a_m = np.sqrt(float(area_cm2) * 1e-4 / np.pi)
    return float(1.0 / (4.0 * conductivity * a_m))


def naked_electrode_impedance(
    frequency: np.ndarray,
    *,
    area_cm2: float,
    conductivity: float,
    cpe_q: float = 3.0e-5,
    cpe_n: float = 0.92,
) -> np.ndarray:
    """Total impedance of a cell-free electrode [Ohm].

    Interface (specific impedance divided by area) in series with the
    electrolyte spreading resistance.
    """
    z_interface = electrode_specific_impedance(frequency, cpe_q=cpe_q, cpe_n=cpe_n)
    return z_interface / float(area_cm2) + solution_resistance(
        area_cm2, conductivity=conductivity
    )


def membrane_impedance(
    frequency: np.ndarray, *, specific_capacitance: float
) -> np.ndarray:
    """Specific impedance of the two cell membranes in series [Ohm*cm^2].

    ``Z_m = 2 / (i * omega * Cm)`` — dorsal and ventral membranes in series, so
    twice the single-membrane impedance.  Membrane conductance is neglected:
    the resting membrane conductivity (~1e-7 S/m) contributes below 1 % above
    1 kHz.
    """
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    return 2.0 / (1j * omega * float(specific_capacitance))


def alpha_parameter(cell_radius: float, gap_height: float, resistivity: float) -> float:
    """Giaever-Keese ventral-gap parameter ``alpha = r_c * sqrt(rho / h)``.

    Parameters
    ----------
    cell_radius, gap_height:
        In metres.
    resistivity:
        Medium resistivity [Ohm*m], i.e. ``1 / conductivity``.

    Returns
    -------
    float
        ``alpha`` in Ohm^0.5 * cm, the unit the Giaever-Keese solution expects.
    """
    r_cm = cell_radius * 1e2
    h_cm = gap_height * 1e2
    rho_ohm_cm = resistivity * 1e2
    return float(r_cm * np.sqrt(rho_ohm_cm / h_cm))


def giaever_keese_impedance(
    frequency: np.ndarray,
    *,
    z_naked_specific: np.ndarray,
    rb: float,
    alpha: float,
    specific_capacitance: float,
    coverage: float = 1.0,
) -> np.ndarray:
    """Specific impedance of a cell-covered electrode [Ohm*cm^2].

    Implements the Giaever-Keese solution (doi:10.1073/pnas.88.17.7896):

        1/Zc = (1/Zn) * [ Zn/(Zn + Zm)
               + (Zm/(Zn + Zm)) / ( gamma*I0(gamma)/(2*I1(gamma))
                                    + Rb*(1/Zn + 1/Zm) ) ]

    with ``gamma = alpha * sqrt(1/Zn + 1/Zm)``.

    Parameters
    ----------
    z_naked_specific:
        Specific impedance of the cell-free electrode [Ohm*cm^2].
    rb:
        Junctional (paracellular) resistance [Ohm*cm^2]. Leaky lines are
        ~1 Ohm*cm^2; tight epithelia reach 10-20.
    alpha:
        Ventral-gap parameter [Ohm^0.5*cm], see :func:`alpha_parameter`.
    specific_capacitance:
        Membrane capacitance [F/cm^2]; ~1 uF/cm^2 for a lipid bilayer.
    coverage:
        Fraction of the electrode covered by cells, 0-1. Uncovered area is
        added in parallel as bare electrode, which is what turns a step into a
        growth curve.
    """
    zn = np.asarray(z_naked_specific, dtype=complex)
    zm = membrane_impedance(frequency, specific_capacitance=specific_capacitance)

    inv_sum = 1.0 / zn + 1.0 / zm
    gamma = alpha * np.sqrt(inv_sum)
    # gamma*I0/(2*I1) -> 1 as gamma -> 0; guard the 0/0.
    with np.errstate(invalid="ignore", divide="ignore"):
        bessel = gamma * iv(0, gamma) / (2.0 * iv(1, gamma))
    bessel = np.where(np.isfinite(bessel), bessel, 1.0)

    inv_zc = (1.0 / zn) * (zn / (zn + zm) + (zm / (zn + zm)) / (bessel + rb * inv_sum))
    z_covered = 1.0 / inv_zc

    if coverage >= 1.0:
        return z_covered
    if coverage <= 0.0:
        return zn
    return 1.0 / (coverage / z_covered + (1.0 - coverage) / zn)


def well_impedance(
    frequency: np.ndarray,
    *,
    coverage: float,
    geometry: ElectrodeGeometry,
    conductivity: float = 1.4,
    rb: float = 2.0,
    specific_capacitance: float = 1.0e-6,
    cpe_q: float = 3.0e-5,
    cpe_n: float = 0.92,
) -> np.ndarray:
    """Total measured impedance of one well [Ohm] at the given cell coverage.

    Chains the pieces: interface specific impedance -> Giaever-Keese cell layer
    -> divide by electrode area -> add the electrolyte spreading resistance.
    """
    z_spec_naked = electrode_specific_impedance(frequency, cpe_q=cpe_q, cpe_n=cpe_n)
    alpha = alpha_parameter(
        geometry.cell_radius, geometry.gap_height, 1.0 / conductivity
    )
    z_spec = giaever_keese_impedance(
        frequency,
        z_naked_specific=z_spec_naked,
        rb=rb,
        alpha=alpha,
        specific_capacitance=specific_capacitance,
        coverage=coverage,
    )
    return z_spec / geometry.area_cm2 + solution_resistance(
        geometry.area_cm2, conductivity=conductivity
    )


def shell_model_permittivity(
    frequency: np.ndarray,
    *,
    radius: float,
    membrane_capacitance: float,
    cytoplasm_conductivity: float,
    cytoplasm_permittivity_rel: float = 60.0,
    medium_conductivity: float = 1.5,
    medium_permittivity_rel: float = 78.0,
    volume_fraction: float = 0.1,
) -> np.ndarray:
    """Effective complex permittivity of a cell suspension (single-shell model).

    A cell is a conductive sphere wrapped in a thin insulating membrane; the
    Maxwell-Wagner mixture rule then gives the suspension's effective complex
    permittivity, which is what an impedance analyser actually measures.

    Reference: Asami (2002), doi:10.1016/S0022-3093(02)01110-9.

    Returns
    -------
    ndarray
        Complex permittivity ``eps* = eps' - i*sigma/(omega*eps0)`` (F/m,
        absolute).
    """
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    eps_med = medium_permittivity_rel * EPS0 - 1j * medium_conductivity / omega
    eps_cyt = cytoplasm_permittivity_rel * EPS0 - 1j * cytoplasm_conductivity / omega
    # Thin-shell limit: the membrane enters only through its capacitance.
    cm = membrane_capacitance
    eps_cell = cm * radius * eps_cyt / (cm * radius + eps_cyt)

    # Maxwell-Wagner mixture for a dilute suspension of spheres.
    f = volume_fraction
    ratio = (eps_cell - eps_med) / (eps_cell + 2.0 * eps_med)
    return eps_med * (1.0 + 2.0 * f * ratio) / (1.0 - f * ratio)


def cell_index(
    z_cell: np.ndarray,
    z_background: float | np.ndarray,
    *,
    reference_impedance: float = RTCA_REFERENCE_IMPEDANCE_OHM,
) -> np.ndarray:
    """Cell Index ``CI = (|Z_cell| - |Z_background|) / Z_reference`` (dimensionless).

    See :data:`RTCA_REFERENCE_IMPEDANCE_OHM` for the caveat on the constant.
    """
    return (np.abs(np.asarray(z_cell)) - np.abs(np.asarray(z_background))) / reference_impedance


# ---------------------------------------------------------------------------
# dose-response
# ---------------------------------------------------------------------------


def four_parameter_logistic(
    c: np.ndarray, bottom: float, top: float, ic50: float, hill: float
) -> np.ndarray:
    """Four-parameter logistic (Hill) dose-response curve.

    ``y = bottom + (top - bottom) / (1 + (c / IC50)^hill)``

    Reference: Sebaugh (2011), *Guidelines for accurate EC50/IC50 estimation*,
    Pharm. Stat. 10:128, doi:10.1002/pst.426.
    """
    c = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return bottom + (top - bottom) / (1.0 + (c / ic50) ** hill)


@dataclass
class DoseResponseFit:
    """Result of a four-parameter logistic fit."""

    bottom: float
    top: float
    ic50: float
    hill: float
    covariance: np.ndarray
    r_squared: float

    @property
    def ic50_stderr(self) -> float:
        """Standard error of the IC50 estimate, from the covariance diagonal."""
        return float(np.sqrt(np.abs(self.covariance[2, 2])))

    def predict(self, c: np.ndarray) -> np.ndarray:
        """Evaluate the fitted curve."""
        return four_parameter_logistic(c, self.bottom, self.top, self.ic50, self.hill)


def fit_dose_response(
    concentration: np.ndarray, response: np.ndarray, *, maxfev: int = 20000
) -> DoseResponseFit:
    """Fit a 4PL curve and return the IC50 with its standard error.

    Concentrations must be positive (the model is defined on a log scale);
    zero-dose controls should be passed as the *top* plateau instead of as
    ``c = 0``.
    """
    c = np.asarray(concentration, dtype=float)
    y = np.asarray(response, dtype=float)
    good = np.isfinite(c) & np.isfinite(y) & (c > 0)
    c, y = c[good], y[good]
    if c.size < 4:
        raise ValueError("a 4-parameter fit needs at least 4 positive-concentration points")

    p0 = [float(np.min(y)), float(np.max(y)), float(np.median(c)), 1.0]
    bounds = (
        [-np.inf, -np.inf, c.min() * 1e-3, 0.05],
        [np.inf, np.inf, c.max() * 1e3, 20.0],
    )
    popt, pcov = curve_fit(
        four_parameter_logistic, c, y, p0=p0, bounds=bounds, maxfev=maxfev
    )
    residual = y - four_parameter_logistic(c, *popt)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return DoseResponseFit(*popt, covariance=np.asarray(pcov), r_squared=r2)


__all__ = [
    "EPS0",
    "RTCA_REFERENCE_IMPEDANCE_OHM",
    "ElectrodeGeometry",
    "electrode_specific_impedance",
    "solution_resistance",
    "naked_electrode_impedance",
    "membrane_impedance",
    "alpha_parameter",
    "well_impedance",
    "giaever_keese_impedance",
    "shell_model_permittivity",
    "cell_index",
    "four_parameter_logistic",
    "DoseResponseFit",
    "fit_dose_response",
]
