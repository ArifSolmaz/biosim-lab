"""Stage 2: the interdigitated electrode, its closed forms, and the FEM that checks them.

Two independent routes to the same numbers:

* the conformal-map cell constant of a coplanar IDE (Olthuis et al. 1995,
  doi:10.1016/0925-4005(95)85053-8) against a pure-conduction FEM solve on the
  IDE symmetry cell, and
* the lumped series model (interface - bulk - interface) against the FEM with
  contact-impedance electrodes, in the two limits where each must be exact:
  a dominant interface (large Wagner number, uniform current) and a vanishing
  one (the bulk alone).
"""

import numpy as np
import pytest

from biosim_lab.core.fem.electroquasistatic import ElectroQuasistaticSolver
from biosim_lab.core.geometry import ide_unit_cell_2d
from biosim_lab.core.solver import BoundaryCondition
from biosim_lab.instruments.impedance_rtca.fem_model import IDEFieldModel
from biosim_lab.instruments.impedance_rtca.physics import (
    IDEGeometry,
    electrode_specific_impedance,
    ide_bulk_impedance,
    ide_cell_constant,
    ide_well_impedance,
)

SIGMA = 1.4


def _cell_conductance(eta: float, resolution: int) -> float:
    """FEM conductance per unit length of one IDE gap, electrodes at fixed potential."""
    w = eta * 100e-6
    s = 100e-6 - w
    solver = ElectroQuasistaticSolver(10.0, conductivity=SIGMA, permittivity_rel=0.0)
    solver.setup(ide_unit_cell_2d(w, s, resolution=resolution), [
        BoundaryCondition("electrode_a", "dirichlet", 1.0),
        BoundaryCondition("electrode_b", "dirichlet", 0.0),
    ])
    solver.run()
    return abs(solver.terminal_current("electrode_a", depth=1.0))


def _olthuis_per_gap(eta: float) -> float:
    """Olthuis conductance for ONE gap of unit length, from the device cell constant."""
    g = IDEGeometry(finger_width=eta * 100e-6, finger_spacing=(1 - eta) * 100e-6,
                    finger_length=1.0, n_fingers=2)
    return SIGMA / ide_cell_constant(g)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.8])
def test_fem_reproduces_the_olthuis_cell_constant(eta):
    assert _cell_conductance(eta, 48) == pytest.approx(_olthuis_per_gap(eta), rel=3e-3)


def test_fem_conductance_converges_from_above():
    """A Dirichlet FE solution over-estimates conductance and closes in monotonically."""
    exact = _olthuis_per_gap(0.5)
    errors = [_cell_conductance(0.5, n) / exact - 1.0 for n in (8, 16, 32)]
    assert all(e > 0 for e in errors)
    assert errors[0] > errors[1] > errors[2]


def test_equal_fingers_and_gaps_give_a_unit_elliptic_ratio():
    """At w = s the modulus is cos(pi/4) = its complement, so K(k)/K(k') = 1."""
    g = IDEGeometry(finger_width=50e-6, finger_spacing=50e-6, finger_length=2e-3, n_fingers=11)
    assert ide_cell_constant(g) == pytest.approx(2.0 / (10 * 2e-3))


def test_cell_constant_limits():
    wide = IDEGeometry(finger_width=95e-6, finger_spacing=5e-6)
    thin = IDEGeometry(finger_width=5e-6, finger_spacing=95e-6)
    assert ide_cell_constant(wide) < ide_cell_constant(IDEGeometry()) < ide_cell_constant(thin)


def test_ide_cell_has_no_degenerate_elements():
    """Segments are joined at exact breakpoints; near-duplicates made 1e-27 m^2 slivers."""
    mesh = ide_unit_cell_2d(50e-6, 50e-6, resolution=32).mesh
    p, t = mesh.p, mesh.t
    area = 0.5 * np.abs((p[0, t[1]] - p[0, t[0]]) * (p[1, t[2]] - p[1, t[0]])
                        - (p[0, t[2]] - p[0, t[0]]) * (p[1, t[1]] - p[1, t[0]]))
    # The grading spans ~50x per axis on purpose; slivers were ~1e-16 of the largest.
    assert area.min() / area.max() > 1e-6


def test_fem_equals_the_lumped_model_when_the_interface_dominates():
    """Large Wagner number: current is uniform, the series model is exact."""
    model = IDEFieldModel(IDEGeometry(), resolution=24)
    d = model.diagnostics(100.0, 0.0)
    assert d["wagner_number"] > 100
    assert abs(d["lumped_error_percent"]) < 0.05


def test_fem_equals_the_bulk_when_the_interface_vanishes():
    """A near-zero interface leaves the conformal-map bulk resistance."""
    model = IDEFieldModel(IDEGeometry(), resolution=32, cpe_q=1e3)  # tiny interface
    f = np.array([1e3])
    z_fem = model.impedance(f[0], 0.0)
    z_bulk = ide_bulk_impedance(f, model.geometry, conductivity=model.conductivity,
                                permittivity_rel=model.permittivity_rel)[0]
    assert abs(z_fem) == pytest.approx(abs(z_bulk), rel=5e-3)


def test_current_crowding_makes_the_lumped_model_low_in_between():
    """Where the Wagner number is ~1, current crowds at the finger edges.

    The FEM then sees less effective interface than the lumped model assumes,
    so the lumped model under-predicts |Z| by a few percent --- and only there.
    """
    model = IDEFieldModel(IDEGeometry(), resolution=24)
    mid = model.diagnostics(1e5, 0.0)
    assert 0.3 < mid["wagner_number"] < 3.0
    assert -8.0 < mid["lumped_error_percent"] < -2.0


def test_fem_is_mesh_converged_at_the_default_resolution():
    z = [abs(IDEFieldModel(IDEGeometry(), resolution=n).impedance(1e5, 0.0))
         for n in (24, 48)]
    assert z[0] == pytest.approx(z[1], rel=5e-3)


def test_two_combs_put_two_interfaces_in_series():
    """At low frequency the IDE is interface-dominated: 2 z / A_comb, not z / A."""
    g = IDEGeometry()
    f = np.array([10.0])
    z = ide_well_impedance(f, coverage=0.0, geometry=g)[0]
    two_interfaces = 2.0 * electrode_specific_impedance(f)[0] / g.active_comb_area_cm2
    assert abs(z) == pytest.approx(abs(two_interfaces), rel=1e-3)
