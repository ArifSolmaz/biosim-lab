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
from scipy.special import ellipk, iv

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


# ---------------------------------------------------------------------------
# interdigitated electrodes
# ---------------------------------------------------------------------------


@dataclass
class IDEGeometry:
    """Coplanar gold interdigitated electrode (IDE): two combs of N fingers in total.

    RTCA plates measure between two interdigitated combs of equal size, not
    between a small working electrode and a large counter electrode as classic
    ECIS does. Two things follow, and both are in :func:`ide_well_impedance`:
    the two interfaces sit **in series**, and the bulk resistance is set by the
    finger pattern, not by the spreading resistance of a disc.

    Attributes
    ----------
    finger_width, finger_spacing:
        Metal width ``w`` and gap ``s`` [m].
    finger_length:
        Length of each finger [m].
    n_fingers:
        Total fingers, both combs together (``N``; ``N - 1`` gaps).

    **ASSUMPTION** in the defaults: a generic 50/50 um gold IDE filling a
    3 x 3 mm patch, the scale of a 96-well plate bottom. The dimensions of the
    commercial E-Plate electrodes are not published; replace these with your
    own chip's before comparing absolute impedances.
    """

    finger_width: float = 50e-6
    finger_spacing: float = 50e-6
    finger_length: float = 3e-3
    n_fingers: int = 30

    @property
    def period(self) -> float:
        """Distance between neighbouring fingers of opposite combs, ``w + s`` [m]."""
        return self.finger_width + self.finger_spacing

    @property
    def metallisation_ratio(self) -> float:
        """``eta = w / (w + s)``."""
        return self.finger_width / self.period

    @property
    def comb_area_cm2(self) -> float:
        """Metal area of ONE comb [cm^2] (half the fingers)."""
        return 0.5 * self.n_fingers * self.finger_width * self.finger_length * 1e4

    @property
    def active_comb_area_cm2(self) -> float:
        """Metal area of one comb that faces the other across a gap [cm^2].

        ``(N - 1) w L / 2``: the outer halves of the two end fingers have no
        neighbour. The same ``N - 1`` convention the cell constant uses, so the
        interface and bulk terms describe the same electrode; the difference
        from :attr:`comb_area_cm2` is ``1/N`` (3 % for 30 fingers).
        """
        return 0.5 * (self.n_fingers - 1) * self.finger_width * self.finger_length * 1e4

    @property
    def footprint_cm2(self) -> float:
        """Area the finger pattern spans [cm^2]."""
        return self.n_fingers * self.period * self.finger_length * 1e4


def ide_cell_constant(geometry: IDEGeometry) -> float:
    """Cell constant ``K_cell`` [1/m] of a coplanar IDE under a deep electrolyte.

    ``K_cell = 2 / ((N - 1) L) * K(k) / K(k')``,
    ``k = cos(pi/2 * w / (w + s))``, ``k' = sqrt(1 - k^2)``,

    with ``K`` the complete elliptic integral of the first kind, from the
    conformal map of the periodic finger pattern (Olthuis et al. 1995,
    *Theoretical and experimental determination of cell constants of
    planar-interdigitated electrolyte conductivity sensors*, Sens. Actuators B
    24-25:252, doi:10.1016/0925-4005(95)85053-8). The bulk resistance is
    ``R = K_cell / sigma``. Valid when the electrolyte is several periods
    deep, which a well of medium always is. The electro-quasistatic FEM on
    :func:`~biosim_lab.core.geometry.ide_unit_cell_2d` reproduces it to 0.2 %.
    """
    g = geometry
    if g.n_fingers < 2:
        raise ValueError("an interdigitated electrode needs at least two fingers")
    k = np.cos(0.5 * np.pi * g.metallisation_ratio)
    k_prime = np.sqrt(1.0 - k**2)
    # scipy's ellipk takes the parameter m = k^2, not the modulus k.
    ratio = ellipk(k**2) / ellipk(k_prime**2)
    return float(2.0 / ((g.n_fingers - 1) * g.finger_length) * ratio)


def ide_solution_resistance(geometry: IDEGeometry, *, conductivity: float) -> float:
    """Bulk electrolyte resistance between the two combs, ``K_cell / sigma`` [Ohm]."""
    return ide_cell_constant(geometry) / float(conductivity)


def ide_bulk_impedance(
    frequency: np.ndarray, geometry: IDEGeometry, *, conductivity: float,
    permittivity_rel: float = 78.0,
) -> np.ndarray:
    """``K_cell / sigma*`` [Ohm], with ``sigma* = sigma + i omega eps0 eps_r``.

    The electrolyte is a leaky capacitor, and above a few MHz the displacement
    current is no longer negligible (3 % of the conduction current at 10 MHz
    in saline).
    """
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=float)
    sigma_star = float(conductivity) + 1j * omega * EPS0 * float(permittivity_rel)
    return ide_cell_constant(geometry) / sigma_star


def ide_well_impedance(
    frequency: np.ndarray,
    *,
    coverage: float | np.ndarray,
    geometry: IDEGeometry,
    conductivity: float = 1.4,
    rb: float = 2.0,
    specific_capacitance: float = 1.0e-6,
    cell_radius: float = 8.0e-6,
    gap_height: float = 100e-9,
    cpe_q: float = 3.0e-5,
    cpe_n: float = 0.92,
    permittivity_rel: float = 78.0,
) -> np.ndarray:
    """Impedance of a cell-covered IDE [Ohm], lumped (uniform-current) model.

    ``Z = 2 z_c / A_comb + K_cell / sigma*``: each comb's interface --- the
    CPE double layer, covered by the Giaever-Keese cell layer
    (doi:10.1073/pnas.88.17.7896) at the given coverage --- in series with the
    other comb's and with the bulk resistance of the finger pattern
    (:func:`ide_cell_constant`).

    It assumes current crosses each finger uniformly. That fails when the
    interface impedance is small next to the electrolyte resistance across a
    finger (a small :func:`wagner_number`), where current crowds onto the
    finger edges; the FEM model (``fem_model.IDEFieldModel``) solves that
    case, and the two agree when the Wagner number is large.
    *coverage* may be an array (one impedance per coverage, broadcast against
    *frequency*).
    """
    z_naked = electrode_specific_impedance(frequency, cpe_q=cpe_q, cpe_n=cpe_n)
    alpha = alpha_parameter(cell_radius, gap_height, 1.0 / conductivity)
    z_covered = giaever_keese_impedance(
        frequency, z_naked_specific=z_naked, rb=rb, alpha=alpha,
        specific_capacitance=specific_capacitance, coverage=1.0,
    )
    cov = np.asarray(coverage, dtype=float)
    z_spec = _parallel_coverage(z_covered, z_naked, cov)
    return 2.0 * z_spec / geometry.active_comb_area_cm2 + ide_bulk_impedance(
        frequency, geometry, conductivity=conductivity, permittivity_rel=permittivity_rel
    )


def _parallel_coverage(z_covered: np.ndarray, z_naked: np.ndarray,
                       coverage: np.ndarray) -> np.ndarray:
    """Covered and bare patches in parallel, ``1/z = c/z_cov + (1-c)/z_bare``, vectorised."""
    c = np.clip(coverage, 0.0, 1.0)
    return 1.0 / (c / z_covered + (1.0 - c) / z_naked)


def wagner_number(
    specific_impedance_ohm_cm2: complex | np.ndarray, *, conductivity: float,
    length: float,
) -> np.ndarray:
    """``Wa = |z_s| sigma / L``: interface against electrolyte resistance over *length*.

    Large ``Wa`` means the interface dominates and current spreads uniformly
    over the electrode (a lumped model is exact); small ``Wa`` means it crowds
    at the edges. Newman, *Electrochemical Systems*, 3rd ed., section 18.3
    (ISBN 978-0-471-47756-3; the Wagner number is the ratio of polarisation to
    ohmic resistance).
    """
    z_si = np.abs(np.asarray(specific_impedance_ohm_cm2)) * 1e-4  # Ohm*m^2
    return z_si * float(conductivity) / float(length)


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
    "IDEGeometry",
    "ide_cell_constant",
    "ide_solution_resistance",
    "ide_bulk_impedance",
    "ide_well_impedance",
    "wagner_number",
    "cell_index",
    "four_parameter_logistic",
    "DoseResponseFit",
    "fit_dose_response",
]
