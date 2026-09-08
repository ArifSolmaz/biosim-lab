"""Acoustic radiation force in a standing surface-acoustic-wave (SSAW) field.

Conventions used throughout this module
---------------------------------------
**Time factor** ``exp(-i omega t)``.

**Contrast factor.** Two definitions circulate in the literature and differ by
a factor of three.  This module uses the *classical* one,

    Phi = (5*rho_p - 2*rho_f) / (2*rho_p + rho_f)  -  kappa_p / kappa_f

as introduced by Yosioka & Kawasima (1955) and reproduced in most
cell-sorting papers.  Bruus (doi:10.1039/c2lc21068a) writes the same physics as
``Phi_Bruus = f1/3 + f2/2 = Phi / 3``.  Mixing the two is the single most
common factor-3 error in acoustophoresis code, so :func:`contrast_factor`
returns the classical value and :func:`bruus_phi` is provided explicitly.

**Force law.** For a one-dimensional standing wave with a *pressure node* at
``x_node``,

    F_r(x) = -(pi * p0^2 * V_c * kappa_f / (2*lambda)) * Phi * sin(2*k*(x - x_node))

which is exactly the expression in the project specification with the origin
placed at the node.  The sign is such that a **positive-contrast** particle
(``Phi > 0``, e.g. every mammalian cell in water) is pushed *towards* the
pressure node, and a negative-contrast particle (lipid droplet) towards the
antinode.  Derivation and equivalence to ``F = 4*pi*Phi_Bruus*a^3*k*E_ac*sin(2kx)``
are shown in the module tests.

References
----------
* Gor'kov (1962), *On the forces acting on a small particle in an acoustical
  field in an ideal fluid*, Sov. Phys. Dokl. 6:773. (No DOI: pre-DOI Soviet
  journal; the result is reproduced with derivation in the Bruus tutorial
  below.)
* Bruus (2012), *Acoustofluidics 7: The acoustic radiation force on small
  particles*, Lab Chip 12:1014, doi:10.1039/c2lc21068a.
* Yosioka & Kawasima (1955), *Acoustic radiation pressure on a compressible
  sphere*, Acustica 5:167. (Pre-DOI.)
* Shi et al. (2009), *Focusing microparticles in a microfluidic channel with
  standing surface acoustic waves*, Lab Chip 9:3354, doi:10.1039/b910595f —
  establishes that the pressure-node spacing in an SSAW device is
  ``lambda_SAW / 2``, set by the substrate periodicity, **not** by the sound
  wavelength in the liquid.
* Ding et al. (2012), *On-chip manipulation of single microparticles, cells and
  organisms using surface acoustic waves*, PNAS 109:11105,
  doi:10.1073/pnas.1209288109.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from biosim_lab.core.plugin import RegimeWarning

# ---------------------------------------------------------------------------
# contrast factor
# ---------------------------------------------------------------------------


def f1_monopole(kappa_p: float | np.ndarray, kappa_f: float) -> np.ndarray:
    """Monopole scattering coefficient ``f1 = 1 - kappa_p / kappa_f``.

    doi:10.1039/c2lc21068a, eq. 20.
    """
    return 1.0 - np.asarray(kappa_p, dtype=float) / float(kappa_f)


def f2_dipole(rho_p: float | np.ndarray, rho_f: float) -> np.ndarray:
    """Dipole scattering coefficient ``f2 = 2*(rho_p - rho_f) / (2*rho_p + rho_f)``.

    doi:10.1039/c2lc21068a, eq. 20.
    """
    rho_p = np.asarray(rho_p, dtype=float)
    return 2.0 * (rho_p - rho_f) / (2.0 * rho_p + rho_f)


def bruus_phi(
    rho_p: float | np.ndarray, rho_f: float, kappa_p: float | np.ndarray, kappa_f: float
) -> np.ndarray:
    """Acoustophoretic contrast factor in Bruus' normalisation, ``f1/3 + f2/2``.

    doi:10.1039/c2lc21068a, eq. 24.
    """
    return f1_monopole(kappa_p, kappa_f) / 3.0 + f2_dipole(rho_p, rho_f) / 2.0


def contrast_factor(
    rho_p: float | np.ndarray, rho_f: float, kappa_p: float | np.ndarray, kappa_f: float
) -> np.ndarray:
    """Classical acoustic contrast factor ``Phi`` (three times :func:`bruus_phi`).

    Parameters
    ----------
    rho_p, rho_f:
        Particle and fluid density [kg/m^3].
    kappa_p, kappa_f:
        Particle and fluid isentropic compressibility [1/Pa].

    Returns
    -------
    ndarray
        ``Phi = (5*rho_p - 2*rho_f)/(2*rho_p + rho_f) - kappa_p/kappa_f``.
        ``Phi > 0`` moves the particle to the pressure node.

    Notes
    -----
    Every mammalian cell in an aqueous buffer has ``Phi > 0`` (denser and
    stiffer than water); lipid droplets and air-filled contrast agents have
    ``Phi < 0``.  ``tests/test_acoustics.py`` asserts both.
    """
    rho_p = np.asarray(rho_p, dtype=float)
    kappa_p = np.asarray(kappa_p, dtype=float)
    return (5.0 * rho_p - 2.0 * rho_f) / (2.0 * rho_p + rho_f) - kappa_p / float(kappa_f)


def effective_contrast_factor(
    rho_p: float | np.ndarray,
    rho_f: float,
    kappa_p: float | np.ndarray,
    kappa_f: float,
    *,
    k_transverse: float,
    k_fluid: float,
) -> np.ndarray:
    """Contrast factor corrected for a standing wave that is **not** one-dimensional.

    Why this exists
    ---------------
    The textbook force law assumes a plane standing wave whose transverse wave
    number equals the fluid wave number, ``k_x = k_f = omega / c_f``.  In an
    **SSAW device that is not true**: the lateral periodicity is imposed by the
    substrate (``k_x = 2*pi / lambda_SAW``) while the fluid still obeys the
    Helmholtz equation, so the field carries a vertical component
    ``k_y = sqrt(k_f^2 - k_x^2)`` — exactly the leaky-wave refraction at the
    Rayleigh angle.

    Writing the field as ``p = p0 * sin(k_x*x) * cos(k_y*y)`` and inserting it
    into the Gor'kov potential gives, in a plane where ``cos^2(k_y*y) = 1``
    (where the pressure amplitude peaks),

        F_x = -(V*p0^2*kappa_f*k_x / 4) * sin(2*k_x*x) * [ f1 + (3/2)*f2*(k_x/k_f)^2 ]

    so the *monopole* term is unchanged while the *dipole* term is scaled by
    ``(k_x / k_f)^2 = sin^2(theta_R_complement)``.  Setting ``k_x = k_f``
    recovers ``Phi = f1 + (3/2)*f2`` and the classical expression exactly, which
    ``tests/test_acoustics.py`` asserts.

    Parameters
    ----------
    k_transverse:
        Wave number of the standing pattern along the acoustic axis [1/m],
        ``2*pi / lambda_SAW`` for an SSAW device.
    k_fluid:
        ``omega / c_f`` [1/m].

    Returns
    -------
    ndarray
        Effective contrast factor to feed to :func:`primary_radiation_force_1d`
        together with ``wavelength = 2*pi / k_transverse``.
    """
    if k_fluid <= 0:
        raise ValueError("k_fluid must be positive")
    ratio2 = (float(k_transverse) / float(k_fluid)) ** 2
    if ratio2 > 1.0 + 1e-12:
        raise ValueError(
            "k_transverse exceeds k_fluid: the pattern would be evanescent in the fluid "
            "(this happens if the SAW velocity is below the sound speed in the liquid)"
        )
    return f1_monopole(kappa_p, kappa_f) + 1.5 * f2_dipole(rho_p, rho_f) * ratio2


def acoustic_energy_density(p0: float, kappa_f: float) -> float:
    """Time- and space-averaged acoustic energy density ``E_ac = p0^2 * kappa_f / 4`` [J/m^3].

    For a standing wave of pressure amplitude *p0*.  doi:10.1039/c2lc21068a, eq. 33.
    """
    return 0.25 * float(p0) ** 2 * float(kappa_f)


# ---------------------------------------------------------------------------
# geometry of the SSAW field
# ---------------------------------------------------------------------------


def saw_wavelength(frequency: float, saw_velocity: float) -> float:
    """SAW wavelength on the substrate ``lambda_SAW = c_SAW / f`` [m].

    In an SSAW device this — not the wavelength in the liquid — sets the
    pressure-node spacing, because the fluid field inherits the periodicity of
    the substrate displacement (doi:10.1039/b910595f).
    """
    if frequency <= 0 or saw_velocity <= 0:
        raise ValueError("frequency and saw_velocity must be positive")
    return float(saw_velocity) / float(frequency)


def rayleigh_angle(sound_speed_fluid: float, saw_velocity: float) -> float:
    """Rayleigh (leaky-wave) refraction angle into the fluid [rad].

    ``theta_R = arcsin(c_fluid / c_SAW)``; the leaky SAW radiates into the
    liquid at this angle from the surface normal.  For water on 128 deg YX
    LiNbO3 it is about 22 degrees.

    Raises
    ------
    ValueError
        If ``c_fluid >= c_SAW``, in which case no leaky wave exists.
    """
    ratio = float(sound_speed_fluid) / float(saw_velocity)
    if ratio >= 1.0:
        raise ValueError(
            "no leaky Rayleigh wave: the fluid sound speed must be below the SAW velocity "
            f"(got c_f={sound_speed_fluid:g} >= c_SAW={saw_velocity:g} m/s)"
        )
    return float(np.arcsin(ratio))


def node_spacing(wavelength: float) -> float:
    """Distance between adjacent pressure nodes, ``lambda / 2`` [m]."""
    return 0.5 * float(wavelength)


def node_positions(
    channel_width: float, wavelength: float, *, node_offset: float = 0.0
) -> np.ndarray:
    """Pressure-node positions inside ``[0, channel_width]`` [m].

    Parameters
    ----------
    channel_width:
        Channel width along the acoustic axis [m].
    wavelength:
        SSAW wavelength [m] (use :func:`saw_wavelength`).
    node_offset:
        Position of one node [m]; the rest follow at ``lambda/2`` spacing.
        The default puts a node at ``x = 0`` (the wall).  To centre a node,
        pass ``channel_width / 2``.
    """
    spacing = node_spacing(wavelength)
    first = node_offset - np.ceil(node_offset / spacing) * spacing
    nodes = np.arange(first, channel_width + 0.5 * spacing, spacing)
    return nodes[(nodes >= -1e-15) & (nodes <= channel_width + 1e-15)]


# ---------------------------------------------------------------------------
# primary radiation force
# ---------------------------------------------------------------------------


def check_gorkov_validity(radius: np.ndarray | float, wavelength: float) -> float:
    """Warn when the long-wavelength (Rayleigh) assumption ``a << lambda`` is stretched.

    Gor'kov's potential is the leading term of an expansion in ``k*a``; the
    usual practical limit is ``k*a < 0.1``, i.e. ``a < lambda / 63``.
    Returns the worst-case ``k*a``.
    """
    ka = float(np.max(np.asarray(radius, dtype=float))) * 2.0 * np.pi / float(wavelength)
    if ka > 0.1:
        warnings.warn(
            f"k*a = {ka:.3g} > 0.1: the Gor'kov long-wavelength approximation is being "
            "stretched; the true force is smaller than predicted and higher multipoles "
            "matter (doi:10.1039/c2lc21068a, section 5).",
            RegimeWarning,
            stacklevel=2,
        )
    return ka


def primary_radiation_force_1d(
    x: np.ndarray | float,
    *,
    p0: float,
    volume: np.ndarray | float,
    kappa_f: float,
    wavelength: float,
    phi: np.ndarray | float,
    node_offset: float = 0.0,
) -> np.ndarray:
    """Primary acoustic radiation force along the acoustic axis [N].

    Implements

        F_r = -(pi * p0^2 * V_c * kappa_f / (2*lambda)) * Phi * sin(2*k*(x - x_node))

    with ``k = 2*pi/lambda``.  This is the specification's formula with the
    origin shifted to the pressure node at *node_offset*.

    Parameters
    ----------
    x:
        Position(s) along the acoustic axis [m].
    p0:
        Pressure amplitude of the standing wave [Pa].
    volume:
        Particle volume [m^3].
    kappa_f:
        Fluid compressibility ``1/(rho_f c_f^2)`` [1/Pa].  (The specification
        writes this as ``beta_f``.)
    wavelength:
        Standing-wave wavelength [m] — for an SSAW device this is the *SAW*
        wavelength, see :func:`saw_wavelength`.
    phi:
        Classical contrast factor from :func:`contrast_factor`.
    node_offset:
        Position of the pressure node [m].

    Returns
    -------
    ndarray
        Force in newtons, positive along ``+x``.
    """
    x = np.asarray(x, dtype=float)
    k = 2.0 * np.pi / float(wavelength)
    amplitude = (
        np.pi * float(p0) ** 2 * np.asarray(volume, dtype=float) * float(kappa_f)
        / (2.0 * float(wavelength))
    )
    return -amplitude * np.asarray(phi, dtype=float) * np.sin(2.0 * k * (x - node_offset))


def ssaw_pressure_field(
    x: np.ndarray | float,
    *,
    p0: float,
    wavelength: float,
    node_offset: float = 0.0,
    y: np.ndarray | float | None = None,
    decay_length: float | None = None,
) -> np.ndarray:
    """Standing pressure field ``p(x) = p0 * sin(k*(x - x_node))`` [Pa].

    This is the field whose Gor'kov force is :func:`primary_radiation_force_1d`
    (a *sine* because the node sits at *node_offset*).

    Parameters
    ----------
    y, decay_length:
        Optional exponential attenuation with height above the substrate,
        ``exp(-y / decay_length)``.  Leaky-SAW devices lose amplitude with
        distance from the transducer surface; the decay length is device
        specific and should be fitted, so it is off by default.
    """
    k = 2.0 * np.pi / float(wavelength)
    field = float(p0) * np.sin(k * (np.asarray(x, dtype=float) - node_offset))
    if y is not None and decay_length:
        field = field * np.exp(-np.asarray(y, dtype=float) / float(decay_length))
    return field


# ---------------------------------------------------------------------------
# general Gor'kov potential from a computed field
# ---------------------------------------------------------------------------


def gorkov_potential(
    p: np.ndarray,
    grad_p: tuple[np.ndarray, ...],
    *,
    volume: float,
    rho_f: float,
    c_f: float,
    rho_p: float,
    kappa_p: float,
    omega: float,
) -> np.ndarray:
    """Gor'kov potential ``U`` [J] for a complex time-harmonic pressure field.

    ``U = V * ( (f1/2) * kappa_f * <p^2>  -  (3/4) * f2 * rho_f * <v^2> )``

    with, for the ``exp(-i omega t)`` convention and complex amplitudes,
    ``<p^2> = |p|^2 / 2`` and ``<v^2> = |grad p|^2 / (2 * (rho_f * omega)^2)``
    (from ``v = grad(p) / (i * omega * rho_f)``).

    doi:10.1039/c2lc21068a, eq. 17-19.

    Parameters
    ----------
    p:
        Complex pressure amplitude on a grid [Pa].
    grad_p:
        Tuple of complex pressure gradients, one array per axis [Pa/m].
    volume:
        Particle volume [m^3].
    rho_f, c_f:
        Fluid density [kg/m^3] and sound speed [m/s].
    rho_p, kappa_p:
        Particle density [kg/m^3] and compressibility [1/Pa].
    omega:
        Angular frequency [rad/s].
    """
    kappa_f = 1.0 / (rho_f * c_f**2)
    f1 = float(f1_monopole(kappa_p, kappa_f))
    f2 = float(f2_dipole(rho_p, rho_f))
    p2 = 0.5 * np.abs(p) ** 2
    grad_sq = sum(np.abs(g) ** 2 for g in grad_p)
    v2 = 0.5 * grad_sq / (rho_f * omega) ** 2
    return volume * (0.5 * f1 * kappa_f * p2 - 0.75 * f2 * rho_f * v2)


def gorkov_force_on_grid(
    field: xr.Dataset,
    *,
    volume: float,
    rho_f: float,
    c_f: float,
    rho_p: float,
    kappa_p: float,
    frequency: float,
) -> xr.Dataset:
    """Gor'kov force field from a gridded complex pressure solution.

    Takes the output of
    :meth:`~biosim_lab.core.fem.helmholtz.HelmholtzSolver.sample_on_grid`
    (variables ``p_real``/``p_imag`` on coordinates ``x``, ``y``) and returns
    ``U``, ``F_x`` and ``F_y`` on the same grid.

    The gradients are evaluated by second-order central differences
    (``numpy.gradient``): the potential is a *quadratic* functional of ``p``, so
    differentiating P2 shape functions twice would be noisier than
    differentiating the interpolated field once.
    """
    xs = field["x"].values
    ys = field["y"].values
    p = field["p_real"].values + 1j * field["p_imag"].values
    dpdx, dpdy = np.gradient(p, xs, ys, edge_order=2)
    omega = 2.0 * np.pi * frequency

    U = gorkov_potential(
        p,
        (dpdx, dpdy),
        volume=volume,
        rho_f=rho_f,
        c_f=c_f,
        rho_p=rho_p,
        kappa_p=kappa_p,
        omega=omega,
    )
    dUdx, dUdy = np.gradient(U, xs, ys, edge_order=2)

    out = xr.Dataset(
        {
            "gorkov_potential": (("x", "y"), U),
            "F_x": (("x", "y"), -dUdx),
            "F_y": (("x", "y"), -dUdy),
            "p_abs": (("x", "y"), np.abs(p)),
        },
        coords={"x": xs, "y": ys},
    )
    out["gorkov_potential"].attrs["units"] = "J"
    out["F_x"].attrs["units"] = out["F_y"].attrs["units"] = "N"
    out["p_abs"].attrs["units"] = "Pa"
    out.attrs.update(
        frequency_Hz=frequency,
        particle_volume_m3=volume,
        rho_f_kg_m3=rho_f,
        c_f_m_s=c_f,
        method="Gor'kov potential, central-difference gradients",
        reference="doi:10.1039/c2lc21068a",
    )
    return out


__all__ = [
    "f1_monopole",
    "f2_dipole",
    "bruus_phi",
    "contrast_factor",
    "effective_contrast_factor",
    "acoustic_energy_density",
    "saw_wavelength",
    "rayleigh_angle",
    "node_spacing",
    "node_positions",
    "check_gorkov_validity",
    "primary_radiation_force_1d",
    "ssaw_pressure_field",
    "gorkov_potential",
    "gorkov_force_on_grid",
    "max_trappable_tilt",
    "cutoff_radius",
]


def _force_to_drag_ratio(
    radius: float | np.ndarray,
    *,
    p0: float,
    kappa_f: float,
    wavelength: float,
    phi: float | np.ndarray,
    viscosity: float,
    flow_speed: float,
) -> np.ndarray:
    """Peak radiation force divided by the drag of moving at the flow speed.

    Substituting the Gor'kov amplitude and Stokes drag into
    ``F_amp / (6 pi mu a u)`` and cancelling gives

        R = pi * p0^2 * kappa_f * Phi * a^2 / (9 * mu * lambda * u)

    which is dimensionless and, note, scales with ``a^2`` --- the same ``r^2``
    that governs migration speed, so trapping inherits the size selectivity.
    """
    a = np.asarray(radius, dtype=float)
    return (
        np.pi * float(p0) ** 2 * float(kappa_f) * np.abs(phi) * a**2
        / (9.0 * float(viscosity) * float(wavelength) * float(flow_speed))
    )


def max_trappable_tilt(
    radius: float | np.ndarray,
    *,
    p0: float,
    kappa_f: float,
    wavelength: float,
    phi: float | np.ndarray,
    viscosity: float,
    flow_speed: float,
) -> np.ndarray:
    """Largest IDT tilt [rad] at which a particle is still carried by a node.

    In a tilted-angle device (doi:10.1073/pnas.1413325111) the node planes cross
    the flow, so holding a particle on one requires dragging it sideways at
    ``u * tan(theta)`` for the whole channel. Writing the projected coordinate as
    ``xi`` and setting ``d(xi)/dt = 0`` for a particle advected at ``u``,

        |F| = 6 * pi * mu * a * u * sin(theta) / cos^2(theta)

    so a particle stays trapped only while the available force covers that.
    With ``R`` from :func:`_force_to_drag_ratio` the condition is
    ``sin(theta) / cos^2(theta) <= R``, and substituting ``s = sin(theta)`` makes
    it the quadratic ``R s^2 + s - R = 0``, giving the closed form below.

    **This is the design number for a tilted device.** Past this angle the
    particle slips across node planes, the average force cancels, and it flows
    straight through: the device silently does nothing rather than failing
    loudly. Because ``R`` scales with ``a^2``, small cells lose their grip first
    --- which is exactly the separation mechanism, and why the useful tilt sits
    between the two populations' limits.
    """
    ratio = _force_to_drag_ratio(
        radius, p0=p0, kappa_f=kappa_f, wavelength=wavelength, phi=phi,
        viscosity=viscosity, flow_speed=flow_speed,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_theta = (-1.0 + np.sqrt(1.0 + 4.0 * ratio**2)) / (2.0 * ratio)
    sin_theta = np.where(np.isfinite(sin_theta), sin_theta, 0.0)
    return np.arcsin(np.clip(sin_theta, 0.0, 1.0))


def cutoff_radius(
    tilt_angle: float,
    *,
    p0: float,
    kappa_f: float,
    wavelength: float,
    phi: float,
    viscosity: float,
    flow_speed: float,
) -> float:
    """Smallest particle radius [m] still carried by a node at *tilt_angle* [rad].

    The inverse of :func:`max_trappable_tilt`, and the more directly useful form:
    it is the **cutoff size of a tilted-angle sorter**. Everything larger is
    deflected across the channel, everything smaller flows straight through.

    Inverting ``R >= sin(theta)/cos^2(theta)`` for the radius gives

        a_cutoff = sqrt( 9 * mu * lambda * u * T / (pi * p0^2 * kappa_f * Phi) )

    with ``T = sin(theta)/cos^2(theta)``. Tilt harder and the cutoff rises;
    raise the field or slow the flow and it falls.
    """
    theta = abs(float(tilt_angle))
    t = np.sin(theta) / np.cos(theta) ** 2
    numerator = 9.0 * float(viscosity) * float(wavelength) * float(flow_speed) * t
    denominator = np.pi * float(p0) ** 2 * float(kappa_f) * abs(float(phi))
    if denominator <= 0.0:
        return float("inf")
    return float(np.sqrt(numerator / denominator))
