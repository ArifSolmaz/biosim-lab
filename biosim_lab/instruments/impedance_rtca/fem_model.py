"""Electro-quasistatic FEM of a cell-covered interdigitated electrode.

The lumped model (:func:`~.physics.ide_well_impedance`) puts the two comb
interfaces and the bulk resistance in series, which assumes the current
crosses every finger uniformly. On a coplanar electrode it does not: the
current density is singular at the finger edges, and when the interface
impedance is small next to the electrolyte resistance across a finger (a
small Wagner number, i.e. at high frequency) the current crowds there. This
module solves for that directly:

    div( sigma* grad(phi) ) = 0          in the electrolyte,
    sigma* dphi/dn = (phi_e - phi) / z   on each comb,

on the symmetry cell of an infinite IDE
(:func:`~biosim_lab.core.geometry.ide_unit_cell_2d`), with ``sigma* = sigma +
i omega eps0 eps_r`` and ``z`` the specific impedance of the comb surface ---
the CPE double layer, covered by the Giaever-Keese cell layer
(doi:10.1073/pnas.88.17.7896) at the given coverage. The whole electrode is
``N - 1`` such cells in parallel along the finger length.

What the cell layer does here: it is a *boundary* impedance on the metal, as
in Giaever and Keese's own derivation. Cells lying over the insulating gaps
also narrow the electrolyte there; that is not included, and is small next to
the constriction the model does include (the ventral gap, via ``alpha``).

Checked in ``tests/test_impedance_rtca.py`` against two closed forms: pure
conduction gives the conformal-map cell constant of Olthuis et al. (1995,
doi:10.1016/0925-4005(95)85053-8), and a large interface impedance gives the
lumped series model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from biosim_lab.core.fem.electroquasistatic import ElectroQuasistaticSolver
from biosim_lab.core.geometry import MeshBundle, ide_unit_cell_2d
from biosim_lab.core.materials import RTCA_MODEL
from biosim_lab.core.solver import BoundaryCondition
from biosim_lab.instruments.impedance_rtca.physics import (
    IDEGeometry,
    _parallel_coverage,
    alpha_parameter,
    electrode_specific_impedance,
    giaever_keese_impedance,
    ide_well_impedance,
    wagner_number,
)

# Defaults mirror ide_well_impedance, and both read the material library so
# the FEM and the lumped model cannot drift apart or lose their provenance.
_M = RTCA_MODEL


@dataclass
class IDEFieldModel:
    """Solve the IDE unit cell and scale it to the whole electrode.

    Parameters mirror :func:`~.physics.ide_well_impedance`, so the FEM and
    the lumped model can be compared argument for argument.
    """

    geometry: IDEGeometry
    conductivity: float = float(_M.medium_conductivity)
    permittivity_rel: float = float(_M.medium_permittivity_rel)
    rb: float = float(_M.junctional_resistance)
    specific_capacitance: float = float(_M.membrane_specific_capacitance)
    cell_radius: float = float(_M.adherent_cell_radius)
    gap_height: float = float(_M.ventral_gap_height)
    cpe_q: float = float(_M.cpe_magnitude)
    cpe_n: float = float(_M.cpe_exponent)
    # Same default as ImpedanceRTCAParams.fem_resolution, so instantiating this
    # directly gives the mesh the instrument and the app actually use.
    resolution: int = 24
    _mesh: MeshBundle | None = field(default=None, init=False, repr=False)
    _last: ElectroQuasistaticSolver | None = field(default=None, init=False, repr=False)

    @property
    def mesh(self) -> MeshBundle:
        if self._mesh is None:
            self._mesh = ide_unit_cell_2d(
                self.geometry.finger_width, self.geometry.finger_spacing,
                resolution=self.resolution,
            )
        return self._mesh

    @property
    def n_cells(self) -> float:
        """Unit cells in parallel: ``(N - 1)`` gaps times the finger length [m]."""
        return (self.geometry.n_fingers - 1) * self.geometry.finger_length

    def surface_impedance(self, frequency: float, coverage: float) -> complex:
        """Specific impedance of a comb surface [Ohm*cm^2] at *coverage*."""
        f = np.array([float(frequency)])
        z_naked = electrode_specific_impedance(f, cpe_q=self.cpe_q, cpe_n=self.cpe_n)
        alpha = alpha_parameter(self.cell_radius, self.gap_height, 1.0 / self.conductivity)
        z_covered = giaever_keese_impedance(
            f, z_naked_specific=z_naked, rb=self.rb, alpha=alpha,
            specific_capacitance=self.specific_capacitance, coverage=1.0,
        )
        return complex(_parallel_coverage(z_covered, z_naked, np.array(coverage))[0])

    def impedance(self, frequency: float, coverage: float) -> complex:
        """Total electrode impedance [Ohm] from one FEM solve."""
        z_s = self.surface_impedance(frequency, coverage) * 1e-4  # Ohm*cm^2 -> Ohm*m^2
        solver = ElectroQuasistaticSolver(
            float(frequency), conductivity=self.conductivity,
            permittivity_rel=self.permittivity_rel,
        )
        solver.setup(self.mesh, [
            BoundaryCondition("electrode_a", "contact_impedance", z_s,
                              meta={"z_s": z_s, "potential": 1.0}),
            BoundaryCondition("electrode_b", "contact_impedance", z_s,
                              meta={"z_s": z_s, "potential": 0.0}),
        ])
        solver.run()
        self._last = solver
        # Per unit length of one cell, then N - 1 cells times the finger length in parallel.
        current = solver.terminal_current("electrode_a", depth=self.n_cells)
        return 1.0 / current

    def spectrum(self, frequencies: np.ndarray, coverage: float) -> np.ndarray:
        """``Z(f)`` [Ohm] at one coverage."""
        return np.array([self.impedance(f, coverage) for f in np.asarray(frequencies)])

    def coverage_table(self, frequency: float, coverages: np.ndarray) -> np.ndarray:
        """``Z(coverage)`` [Ohm] at the readout frequency, for interpolating a time course."""
        return np.array([self.impedance(frequency, c) for c in np.asarray(coverages)])

    def lumped(self, frequencies: np.ndarray, coverage: float) -> np.ndarray:
        """The lumped series model with exactly the same inputs, for comparison."""
        return ide_well_impedance(
            np.asarray(frequencies, dtype=float), coverage=coverage, geometry=self.geometry,
            conductivity=self.conductivity, rb=self.rb,
            specific_capacitance=self.specific_capacitance, cell_radius=self.cell_radius,
            gap_height=self.gap_height, cpe_q=self.cpe_q, cpe_n=self.cpe_n,
            permittivity_rel=self.permittivity_rel,
        )

    def wagner(self, frequency: float, coverage: float) -> float:
        """Wagner number over half a finger, the length current spreads across."""
        return float(wagner_number(
            self.surface_impedance(frequency, coverage), conductivity=self.conductivity,
            length=0.5 * self.geometry.finger_width,
        ))

    def potential_map(self) -> xr.Dataset:
        """``|phi|`` of the last solve on its tensor grid, for plotting."""
        if self._last is None:
            raise RuntimeError("solve something first")
        ds = self._last.fields()
        x = np.unique(np.round(ds["x"].values, 15))
        y = np.unique(np.round(ds["y"].values, 15))
        order = np.lexsort((ds["x"].values, ds["y"].values))
        grid = ds["phi_abs"].values[order].reshape(len(y), len(x)).T
        out = xr.Dataset({"phi_abs": (("x", "y"), grid)}, coords={"x": x, "y": y})
        out["phi_abs"].attrs["units"] = "V"
        out.attrs["drive"] = "1 V on electrode_a (left half-finger), 0 V on electrode_b"
        return out

    def diagnostics(self, frequency: float, coverage: float) -> dict[str, Any]:
        """FEM against the lumped model at one operating point."""
        z_fem = self.impedance(frequency, coverage)
        z_lump = complex(self.lumped(np.array([frequency]), coverage)[0])
        return {
            "frequency_Hz": float(frequency),
            "coverage": float(coverage),
            "z_fem_ohm": abs(z_fem),
            "z_lumped_ohm": abs(z_lump),
            "lumped_error_percent": 100.0 * (abs(z_lump) - abs(z_fem)) / abs(z_fem),
            "wagner_number": self.wagner(frequency, coverage),
            "mesh_nodes": int(self.mesh.mesh.p.shape[1]),
        }


__all__ = ["IDEFieldModel"]
