"""Plugin discovery and the Instrument/Solver contracts."""

import pytest

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import (
    Instrument,
    InstrumentResult,
    discover_instruments,
    get_instrument,
    optional_import,
)
from biosim_lab.core.solver import (
    Solver,
    UnavailableSolver,
    discover_solvers,
    get_solver,
    solver_report,
)

EXPECTED = {"saw_sorter", "impedance_rtca", "cell_counter", "cell_tracker"}


def test_all_shipped_instruments_are_discovered():
    found = discover_instruments()
    assert set(found) >= EXPECTED
    for name, cls in found.items():
        assert issubclass(cls, Instrument)
        assert cls.name == name, f"{name} advertises itself as {cls.name!r}"


def test_every_instrument_declares_its_metadata_and_example():
    for name, cls in discover_instruments().items():
        assert cls.display_name and cls.description, name
        assert cls.ConfigModel is not None, name
        example = cls.example_config()
        cfg = ExperimentConfig.model_validate(example)
        assert cfg.instrument == name
        # The example must validate against the instrument's own schema.
        cls.ConfigModel.model_validate(cfg.params)


def test_unknown_instrument_lists_the_available_ones():
    with pytest.raises(KeyError, match="saw_sorter"):
        get_instrument("does_not_exist")


def test_optional_import_returns_none_instead_of_raising():
    assert optional_import("definitely_not_a_real_module_xyz") is None
    assert optional_import("json") is not None


def test_builtin_solvers_are_available_and_heavy_ones_are_not_required():
    solvers = discover_solvers()
    assert {"builtin_helmholtz", "builtin_stokes",
            "builtin_electroquasistatic"} <= set(solvers)
    for name in ("builtin_helmholtz", "builtin_stokes", "builtin_electroquasistatic"):
        assert solvers[name].is_available()[0], name


def test_missing_backends_degrade_to_an_explaining_placeholder():
    """The core contract: an absent heavy solver is a capability gap, not a crash."""
    solver = get_solver("openfoam")
    if isinstance(solver, UnavailableSolver):
        with pytest.raises(RuntimeError, match="not available"):
            solver.setup(None, [])
        assert "openfoam" in repr(solver) or solver.name == "openfoam"
    else:  # pragma: no cover - only on a machine with OpenFOAM installed
        assert isinstance(solver, Solver)


def test_get_solver_of_an_unregistered_name_is_a_placeholder_not_an_error():
    solver = get_solver("no_such_backend")
    assert isinstance(solver, UnavailableSolver)
    with pytest.raises(RuntimeError, match="no entry point"):
        solver.run()


def test_solver_report_covers_every_registered_backend():
    rows = solver_report()
    names = {row["name"] for row in rows}
    assert {"builtin_helmholtz", "openfoam", "elmer"} <= names
    for row in rows:
        assert isinstance(row["available"], bool)
        assert row["detail"]


def test_results_before_run_is_an_explicit_error():
    cls = get_instrument("saw_sorter")
    inst = cls(ExperimentConfig.model_validate(cls.example_config()))
    with pytest.raises(RuntimeError, match="run"):
        inst.results()


def test_instrument_result_dataset_roundtrip():
    import numpy as np
    import xarray as xr

    result = InstrumentResult(
        fields=xr.Dataset({"a": ("x", np.arange(3.0))}),
        metrics={"x": 1.0},
        meta={"instrument": "t"},
    )
    back = InstrumentResult.from_dataset(result.to_dataset())
    assert back.metrics == {"x": 1.0}
    assert back.meta == {"instrument": "t"}
