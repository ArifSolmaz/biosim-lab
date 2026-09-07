"""FEM mode: Helmholtz pressure field driven by a leaky surface acoustic wave.

Physical model
--------------
Two counter-propagating SAWs on a 128 deg YX LiNbO3 substrate form a standing
SAW.  The substrate surface displacement is imposed as a **prescribed normal
velocity** on the bottom boundary of the fluid domain,

    v_n(x) = -i * omega * u0 * sin(k_SAW * (x - x_node))

(``exp(-i omega t)`` convention, so ``v = -i*omega*u`` for a displacement
amplitude ``u0``).  Because ``c_SAW > c_fluid`` the wave leaks into the liquid
at the Rayleigh angle ``theta_R = arcsin(c_f / c_SAW) ~ 22 deg`` — that
refraction is not imposed by hand, it *emerges* from the Helmholtz solution:
the imposed lateral wavenumber ``k_SAW`` is smaller than ``k_fluid``, so the
vertical wavenumber ``sqrt(k_f^2 - k_SAW^2)`` is real and the field propagates
into the fluid at exactly that angle.

Scope of this stage
-------------------
The **piezoelectric problem is not solved here**.  The substrate surface
displacement ``u0`` is an *input*: converting an IDT drive voltage into ``u0``
requires the full electro-mechanical solution, which is Stage 4
(``solvers/solver_elmer``).  Until then :func:`displacement_from_voltage`
provides a clearly labelled empirical calibration.

References
----------
* Shi et al. (2009), Lab Chip 9:3354, doi:10.1039/b910595f — SSAW node spacing.
* Ding et al. (2012), PNAS 109:11105, doi:10.1073/pnas.1209288109 — SSAW
  cell manipulation, device geometry.
* Pierce, *Acoustics*, doi:10.1007/978-3-030-11214-1 — the
  ``dp/dn = i*omega*rho*v_n`` boundary condition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from biosim_lab.core.fem.helmholtz import HelmholtzSolver
from biosim_lab.core.geometry import MeshBundle, straight_channel_2d
from biosim_lab.core.materials import Fluid, Substrate
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.solver import BoundaryCondition
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    node_positions,
    rayleigh_angle,
    saw_wavelength,
)

#: Empirical IDT drive calibration, **ASSUMPTION**.
#:
#: Converting IDT peak-to-peak voltage into surface displacement depends on the
#: IDT aperture, finger count, electrical matching and substrate cut. Published
#: SSAW devices operating at 10-30 MHz and 10-20 Vpp report acoustic pressure
#: amplitudes of a few hundred kPa in the channel. This module therefore uses a
#: *linear* voltage-to-pressure calibration anchored at 0.45 MPa for 15 Vpp,
#: which sits inside the 0.3-0.6 MPa range quoted for that operating point.
#:
#: Treat it as a fitted device constant, not as physics: measure p0 for your own
#: chip (e.g. by tracking calibration beads and fitting the acoustophoretic
#: velocity) and set ``pressure_amplitude`` explicitly in the config.
PRESSURE_PER_VOLT_ASSUMPTION = 0.45e6 / 15.0
"""Pa per Vpp. ASSUMPTION — see module docstring."""


def pressure_from_voltage(voltage_pp: float) -> float:
    """Acoustic pressure amplitude [Pa] from IDT drive voltage [Vpp].

    **ASSUMPTION** — a linear device calibration, see
    :data:`PRESSURE_PER_VOLT_ASSUMPTION`.  Prefer measuring ``p0`` directly.
    """
    return float(voltage_pp) * PRESSURE_PER_VOLT_ASSUMPTION


def displacement_from_pressure(p0: float, fluid: Fluid) -> float:
    """Substrate displacement amplitude [m] that produces pressure ``p0`` [Pa].

    Uses the plane-wave impedance relation ``p = rho_f * c_f * v`` with
    ``v = omega * u0``, i.e. ``u0 = p0 / (rho_f * c_f * omega)``.  This is only
    an order-of-magnitude seed for the FEM drive: the actual amplitude in a
    resonant channel depends on the quality factor, which is why
    :class:`SAWFieldModel` renormalises the solved field to the requested
    ``p0``.
    """
    return float(p0) / (fluid.rho * fluid.c)


@dataclass
class SAWFieldResult:
    """Solved acoustic field plus the diagnostics needed to interpret it."""

    nodal: xr.Dataset
    grid: xr.Dataset
    mesh: MeshBundle
    diagnostics: dict[str, Any]


class SAWFieldModel:
    """Build and solve the Helmholtz problem for an SSAW sorter cross-section.

    Parameters
    ----------
    frequency:
        Drive frequency [Hz].
    channel_width, channel_height:
        Cross-section [m].
    fluid:
        Suspending medium.
    substrate:
        Piezoelectric substrate (its ``saw_velocity`` sets the node spacing).
    pressure_amplitude:
        Target pressure amplitude ``p0`` [Pa].  The solved field is rescaled so
        that its maximum equals this value — see :meth:`solve` for why.
    node_offset:
        Position of a pressure node along ``x`` [m].
    damping:
        Effective loss factor.  A lossless closed resonator driven exactly on
        resonance has an unbounded response, so a non-zero value is required
        for the amplitude to be meaningful; the default 0.02 corresponds to
        ``Q = 50``, typical of a PDMS-walled SAW channel.  **ASSUMPTION** —
        fit ``Q`` from your device's frequency response.
    wall:
        Side- and top-wall model.

        ``"ssaw"`` (default)
            Rigid side walls, PDMS-impedance ceiling.  This is the physically
            consistent choice for a standing-SAW device: the lateral field
            pattern is imposed by the substrate with period ``lambda_SAW``, and
            SSAW channels are designed with a width that is an integer multiple
            of ``lambda_SAW / 2``, which puts both side walls exactly on
            pressure antinodes where ``dp/dx = 0`` — the rigid condition — while
            the PDMS ceiling is nearly impedance-matched to water and lets the
            leaky wave out instead of trapping it.
            :meth:`check_wall_consistency` warns when the width is not
            commensurate.
        ``"rigid"``
            Hard walls everywhere. The channel becomes a closed resonator and
            the field grows towards the ceiling; useful for bulk-acoustic-wave
            (BAW) devices, misleading for SAW ones.
        ``"soft"``
            Pressure release (``p = 0``) on the sides and top: a fluid/air
            interface.
        ``"pdms"``
            Impedance boundary on sides and top. Appropriate when the channel
            width is *not* commensurate with the SAW wavelength, so the side
            walls genuinely terminate the field.
    """

    def __init__(
        self,
        *,
        frequency: float,
        channel_width: float,
        channel_height: float,
        fluid: Fluid,
        substrate: Substrate,
        pressure_amplitude: float,
        node_offset: float | None = None,
        damping: float = 0.02,
        wall: str = "ssaw",
        resolution: int = 40,
        element_order: int = 2,
    ) -> None:
        self.frequency = float(frequency)
        self.channel_width = float(channel_width)
        self.channel_height = float(channel_height)
        self.fluid = fluid
        self.substrate = substrate
        self.pressure_amplitude = float(pressure_amplitude)
        self.damping = float(damping)
        self.wall = wall
        self.resolution = int(resolution)
        self.element_order = int(element_order)

        if substrate.saw_velocity is None:
            raise ValueError(
                f"substrate {substrate.key!r} has no SAW velocity; pick a piezoelectric cut"
            )
        self.saw_velocity = float(substrate.saw_velocity)
        self.wavelength = saw_wavelength(self.frequency, self.saw_velocity)
        self.node_offset = (
            0.5 * self.channel_width if node_offset is None else float(node_offset)
        )
        self._result: SAWFieldResult | None = None

    # -- diagnostics ------------------------------------------------------
    @property
    def rayleigh_angle_rad(self) -> float:
        """Leaky-wave refraction angle into the fluid [rad]."""
        return rayleigh_angle(self.fluid.c, self.saw_velocity)

    @property
    def node_positions_m(self) -> np.ndarray:
        """Pressure-node positions inside the channel [m]."""
        return node_positions(self.channel_width, self.wavelength,
                              node_offset=self.node_offset)

    @property
    def n_nodes_in_channel(self) -> int:
        """How many pressure nodes actually fall inside the channel width."""
        return int(self.node_positions_m.size)

    def check_single_node(self) -> None:
        """Warn when the channel holds more than one pressure node.

        A multi-node channel does not sort into two outlets: cells collect at
        whichever node they start nearest, so the outlet histogram becomes
        multi-modal.  Either narrow the channel to ``lambda_SAW / 2`` or lower
        the frequency to ``c_SAW / (2 * W)``.
        """
        if self.n_nodes_in_channel > 1:
            suggested = self.saw_velocity / (2.0 * self.channel_width)
            warnings.warn(
                f"{self.n_nodes_in_channel} pressure nodes fit across the "
                f"{self.channel_width * 1e6:.0f} um channel "
                f"(SAW wavelength {self.wavelength * 1e6:.1f} um, node spacing "
                f"{self.wavelength * 5e5:.1f} um). Two-outlet sorting assumes a single "
                f"central node: use f = {suggested / 1e6:.2f} MHz for this width, or narrow "
                f"the channel to {self.wavelength * 5e5:.0f} um at this frequency.",
                RegimeWarning,
                stacklevel=2,
            )

    # -- solve ------------------------------------------------------------
    def build_mesh(self, wall_thickness: float = 0.0) -> MeshBundle:
        """Mesh the channel cross-section (optionally with a PDMS wall layer)."""
        return straight_channel_2d(
            self.channel_width,
            self.channel_height,
            resolution=self.resolution,
            wall_thickness=wall_thickness,
        )

    def boundary_conditions(self, u0: float) -> list[BoundaryCondition]:
        """Leaky-SAW drive on the bottom plus the chosen side/top wall model."""
        omega = 2.0 * np.pi * self.frequency
        k_saw = 2.0 * np.pi / self.wavelength
        x_node = self.node_offset

        def saw_velocity_profile(x: np.ndarray) -> np.ndarray:
            # Standing SAW from two counter-propagating waves. A piston drives pressure
            # in phase with its normal velocity, so the *pressure* node forms above the
            # substrate displacement node: the drive must be sin(), not cos(), for the
            # node to land on `node_offset`. This matches ssaw_pressure_field(), which
            # is also a sine about the node.
            return -1j * omega * u0 * np.sin(k_saw * (x[0] - x_node))

        bcs = [BoundaryCondition("bottom", "velocity", saw_velocity_profile,
                                 meta={"k_saw": k_saw, "u0": u0})]

        from biosim_lab.core.materials import PDMS

        z_pdms = PDMS.rho * PDMS.c

        if self.wall == "ssaw":
            # Sides rigid (they sit on pressure antinodes), ceiling absorbing.
            self.check_wall_consistency()
            bcs.append(BoundaryCondition("top", "impedance", z_pdms))
        elif self.wall == "rigid":
            pass  # natural BC everywhere
        elif self.wall == "soft":
            bcs += [BoundaryCondition(side, "pressure", 0.0)
                    for side in ("left", "right", "top")]
        elif self.wall == "pdms":
            bcs += [BoundaryCondition(side, "impedance", z_pdms)
                    for side in ("left", "right", "top")]
        else:
            raise ValueError(
                f"unknown wall model {self.wall!r}; use ssaw, rigid, soft or pdms"
            )
        return bcs

    def check_wall_consistency(self) -> None:
        """Warn if rigid side walls are being used on a non-commensurate channel.

        Rigid side walls are only consistent with the substrate-imposed pattern
        when the channel width is an integer multiple of ``lambda_SAW / 2``, so
        that both walls fall on pressure antinodes. Otherwise the walls
        genuinely terminate the field and ``wall="pdms"`` is the right model.
        """
        half = 0.5 * self.wavelength
        ratio = self.channel_width / half
        if abs(ratio - round(ratio)) > 0.05:
            warnings.warn(
                f"channel width {self.channel_width * 1e6:.0f} um is "
                f"{ratio:.2f} x (lambda_SAW/2 = {half * 1e6:.0f} um), so the side walls do "
                "not sit on pressure antinodes and the rigid condition of wall='ssaw' is "
                "inconsistent. Use wall='pdms', or pick a commensurate width/frequency.",
                RegimeWarning,
                stacklevel=3,
            )

    def solve(self, *, nx: int = 241, ny: int = 41, normalise: bool = True) -> SAWFieldResult:
        """Solve the Helmholtz problem and return nodal + gridded fields.

        Parameters
        ----------
        normalise:
            Rescale the solved field so ``max|p| == pressure_amplitude``.  The
            absolute amplitude of a driven resonator depends on the quality
            factor, which is a fitted device property rather than a first-
            principles output of this model; renormalising keeps the *mode
            shape* from the FEM while taking the *amplitude* from the
            (measurable) ``p0``.  Set ``False`` to inspect the raw response.
        """
        self.check_single_node()
        mesh = self.build_mesh()
        u0 = displacement_from_pressure(self.pressure_amplitude, self.fluid)

        solver = HelmholtzSolver(
            self.frequency,
            self.fluid.rho,
            self.fluid.c,
            damping=self.damping,
            element_order=self.element_order,
        )
        solver.setup(mesh, self.boundary_conditions(u0))
        solver.run()
        nodal = solver.fields()
        grid = solver.sample_on_grid(nx=nx, ny=ny)

        raw_max = float(np.max(np.hypot(grid["p_real"].values, grid["p_imag"].values)))
        scale = 1.0
        if normalise and raw_max > 0:
            scale = self.pressure_amplitude / raw_max
            for ds in (nodal, grid):
                for name in ds.data_vars:
                    if name.startswith("p_"):
                        ds[name] = ds[name] * scale

        diagnostics = {
            "saw_wavelength_m": self.wavelength,
            "fluid_wavelength_m": self.fluid.c / self.frequency,
            "node_spacing_m": 0.5 * self.wavelength,
            "n_pressure_nodes": self.n_nodes_in_channel,
            "rayleigh_angle_deg": float(np.degrees(self.rayleigh_angle_rad)),
            "surface_displacement_m": u0,
            "raw_max_pressure_Pa": raw_max,
            "normalisation_scale": scale,
            "damping_loss_factor": self.damping,
            "wall_model": self.wall,
            "element_order": self.element_order,
        }
        self._result = SAWFieldResult(
            nodal=nodal, grid=grid, mesh=mesh, diagnostics=diagnostics
        )
        return self._result

    def results(self) -> SAWFieldResult:
        """Return the last solve."""
        if self._result is None:
            raise RuntimeError("call solve() first")
        return self._result

    def force_field(
        self, *, radius: float, density: float, compressibility: float
    ) -> xr.Dataset:
        """Gor'kov force field for one particle species, on the sampling grid.

        Returns a dataset with ``F_x``, ``F_y`` [N] and ``gorkov_potential`` [J].
        Because the force is linear in particle *volume* at fixed contrast, the
        result for another radius is obtained by scaling with ``(r'/r)^3``.
        """
        from biosim_lab.instruments.saw_sorter.physics.acoustics import gorkov_force_on_grid

        res = self.results()
        volume = 4.0 / 3.0 * np.pi * float(radius) ** 3
        out = gorkov_force_on_grid(
            res.grid,
            volume=volume,
            rho_f=self.fluid.rho,
            c_f=self.fluid.c,
            rho_p=float(density),
            kappa_p=float(compressibility),
            frequency=self.frequency,
        )
        out.attrs["particle_radius_m"] = float(radius)
        return out


__all__ = [
    "SAWFieldModel",
    "SAWFieldResult",
    "pressure_from_voltage",
    "displacement_from_pressure",
    "PRESSURE_PER_VOLT_ASSUMPTION",
]
