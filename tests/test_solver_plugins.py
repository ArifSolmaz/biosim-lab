"""Stage 4 back-ends: the whole suite must pass without them installed.

This is the isolation contract from ARCHITECTURE.md section 1: a missing heavy
solver is a *capability gap* that the platform reports, never an import error or
a crash.
"""

import numpy as np
import pytest

from biosim_lab.core.solver import UnavailableSolver, get_solver
from biosim_lab.solvers.solver_elmer import ElmerSolver, surface_displacement_to_bc
from biosim_lab.solvers.solver_openfoam import (
    OpenFOAMSolver,
    rayleigh_streaming_velocity,
    streaming_unavailable_message,
)

OPENFOAM_PRESENT = OpenFOAMSolver.is_available()[0]
ELMER_PRESENT = ElmerSolver.is_available()[0]


def test_importing_the_heavy_backends_never_raises():
    """Merely importing must work everywhere — that is what keeps the core light."""
    assert OpenFOAMSolver.name == "openfoam"
    assert ElmerSolver.name == "elmer"
    assert OpenFOAMSolver.kind == "streaming"
    assert ElmerSolver.kind == "piezo"


def test_availability_reports_what_is_missing():
    ok, reason = OpenFOAMSolver.is_available()
    if not ok:
        assert "simpleFoam" in reason or "fluidfoam" in reason
    ok, reason = ElmerSolver.is_available()
    if not ok:
        assert "ElmerSolver" in reason or "ElmerGrid" in reason


@pytest.mark.skipif(OPENFOAM_PRESENT, reason="OpenFOAM is installed here")
def test_absent_openfoam_yields_an_explaining_placeholder():
    solver = get_solver("openfoam")
    assert isinstance(solver, UnavailableSolver)
    with pytest.raises(RuntimeError, match="not available"):
        solver.run()


def test_stage4_entry_points_raise_a_pointed_not_implemented():
    """The skeleton must say what is missing and where the templates are."""
    with pytest.raises(NotImplementedError, match="Stage 4"):
        OpenFOAMSolver().setup(None, [])
    with pytest.raises(NotImplementedError, match="Stage 4"):
        ElmerSolver().setup(None, [])


def test_templates_are_packaged():
    of = OpenFOAMSolver.templates()
    assert {"acousticRadiationForce.C", "acousticRadiationForce.H",
            "controlDict.template"} <= set(of)
    assert all(p.exists() and p.stat().st_size > 0 for p in of.values())
    elmer = ElmerSolver.templates()
    assert "saw_idt.sif.template" in elmer
    assert "LiNbO3" in elmer["saw_idt.sif.template"].read_text()


def test_the_fallback_message_is_stated_in_both_languages():
    msg = streaming_unavailable_message()
    assert "streaming devre dışı" in msg
    assert "streaming disabled" in msg


def test_rayleigh_streaming_scales_with_the_square_of_the_velocity():
    a = rayleigh_streaming_velocity(0.1, 1497.0)
    b = rayleigh_streaming_velocity(0.2, 1497.0)
    assert b / a == pytest.approx(4.0)
    assert a == pytest.approx(3 / 16 * 0.1**2 / 1497.0)


def test_elmer_bridge_converts_displacement_into_a_velocity_bc():
    """The Stage 4 -> Stage 1 hand-off must be a drop-in for the assumed drive."""
    import xarray as xr

    xs = np.linspace(0.0, 300e-6, 51)
    u = 1e-9 * np.sin(2 * np.pi * xs / 600e-6)
    surface = xr.Dataset(
        {"u_y_real": ("x", u), "u_y_imag": ("x", np.zeros_like(u))},
        coords={"x": xs},
    )
    bc = surface_displacement_to_bc(surface, frequency=6.632e6)
    value = bc(np.array([[150e-6]]))
    omega = 2 * np.pi * 6.632e6
    assert value == pytest.approx(-1j * omega * 1e-9, rel=1e-3)


@pytest.mark.optional_backend
@pytest.mark.skipif(not OPENFOAM_PRESENT, reason="OpenFOAM back-end not installed")
def test_streaming_matches_the_analytic_rayleigh_solution():  # pragma: no cover
    """Validation gate for the OpenFOAM bridge, run only when it is installed.

    Compares the simulated bulk streaming speed against
    ``(3/16) * u1^2 / c`` for a 2-D standing wave.
    """
    pytest.skip(
        "OpenFOAM is installed but the Stage 4 bridge is not implemented; this test "
        "is the acceptance gate for that work."
    )
