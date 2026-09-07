"""Shared fixtures.

Tests must pass with **no optional back-end installed** — that is the contract
in ARCHITECTURE.md section 9.  Anything that needs Gmsh, OpenFOAM, Elmer,
Napari or a GPU segmentation model is skipped rather than failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.materials import WATER
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams


@pytest.fixture(scope="session")
def water():
    return WATER


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260907)


@pytest.fixture
def sorter_params() -> SAWSorterParams:
    """A small, fast, single-node MCF-7 vs RBC separation."""
    return SAWSorterParams(
        frequency=6.632e6,
        voltage_pp=15.0,
        channel_width=300e-6,
        channel_height=50e-6,
        channel_length=2e-3,
        populations=[
            {"cell_type": "mcf7", "count": 60, "target": True},
            {"cell_type": "rbc", "count": 60, "target": False},
        ],
        n_time_samples=41,
        seed=1234,
    )


@pytest.fixture
def sorter_config(sorter_params) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_sorter", instrument="saw_sorter",
        params=sorter_params.model_dump(mode="json"),
    )


def gmsh_ok() -> bool:
    """Whether the optional Gmsh mesher can actually run here."""
    from biosim_lab.core.geometry import gmsh_available

    return gmsh_available()[0]
