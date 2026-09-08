"""Cached wrappers around the simulation core.

Every function here takes primitives and returns plain data, because
``st.cache_data`` keys on the arguments and pickles the result. That is what
makes dragging a slider back to a previous value instant.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import streamlit as st

from biosim_lab.app.shared import UL_MIN
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.materials import get_cell


@st.cache_data(show_spinner="Solving the acoustic field and tracking cells…", max_entries=48)
def run_sorter(
    frequency_mhz: float,
    voltage_pp: float,
    flow_ul_min: float,
    width_um: float,
    height_um: float,
    length_mm: float,
    n_cells: int,
    collection_fraction: float,
    inlet: str,
    mode: str,
    target: str,
    background: str,
    fem_resolution: int,
    temperature_c: float,
    inlet_viability: float,
    rf_power: float,
    seed: int,
    inlet_side: str = "left",
    outlet_layout: str = "centre_band",
    split_position: float = 0.5,
    collect_side: str = "right",
    tilt_angle_deg: float = 0.0,
) -> dict[str, Any]:
    """Run one sorter experiment. Arguments are primitives so caching works."""
    from biosim_lab.instruments.saw_sorter.simulate import (
        SAWSorterParams,
        SAWSorterSimulation,
    )

    params = SAWSorterParams(
        frequency=frequency_mhz * 1e6,
        voltage_pp=voltage_pp,
        temperature=temperature_c + 273.15,
        inlet_viability=inlet_viability,
        rf_power=rf_power or None,
        channel_width=width_um * 1e-6,
        channel_height=height_um * 1e-6,
        channel_length=length_mm * 1e-3,
        flow_rate=flow_ul_min * UL_MIN,
        inlet=inlet,
        inlet_side=inlet_side,
        outlet_layout=outlet_layout,
        tilt_angle_deg=tilt_angle_deg,
        collection_fraction=collection_fraction,
        split_position=split_position,
        collect_side=collect_side,
        mode=mode,
        fem_resolution=fem_resolution,
        fem_grid=(201, 33),
        populations=[
            {"cell_type": target, "count": n_cells, "target": True},
            {"cell_type": background, "count": n_cells, "target": False},
        ],
        seed=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = SAWSorterSimulation(params)
        outcome = sim.run()

    return {
        "metrics": outcome.metrics,
        "cells": outcome.cells,
        "trajectories": outcome.tracks.trajectories,
        "diagnostics": outcome.diagnostics,
        "alive": outcome.cells["alive"].to_numpy(),
        "wavelength": sim.wavelength,
        "node_offset": sim.node_offset,
        "collection_bounds": sim.collection_bounds,
        "kappa_f": sim.fluid.kappa,
        "phi": {
            label: sim.phi_for(get_cell(pop.cell_type))
            for label, pop in zip(
                [p.resolved_label() for p in params.populations], params.populations
            )
        },
        "params": params.model_dump(mode="json"),
        "warnings": [
            {"category": w.category.__name__, "message": str(w.message)} for w in caught
        ],
    }


@st.cache_data(show_spinner="Simulating the impedance plate…", max_entries=16)
def run_rtca(
    frequency_khz: float,
    duration_h: float,
    treatment_h: float,
    true_ic50: float,
    hill: float,
    replicates: int,
    noise_cv: float,
    doubling_h: float,
    seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.impedance_rtca import ImpedanceRTCA

    cfg = ExperimentConfig(
        name="rtca", instrument="impedance_rtca",
        params={
            "frequency": frequency_khz * 1e3,
            "duration": duration_h * 3600.0,
            "treatment_time": treatment_h * 3600.0,
            "true_ic50": true_ic50,
            "hill_slope": hill,
            "replicates": replicates,
            "noise_cv": noise_cv,
            "doubling_time": doubling_h * 3600.0,
            "seed": seed,
        },
    )
    result = ImpedanceRTCA(cfg).run()
    return {
        "metrics": result.metrics,
        "table": result.table,
        "time": result.fields["time"].values,
        "cell_index": result.fields["cell_index"].values,
        "wells": [str(w) for w in result.fields["well"].values],
        "frequency": result.fields["frequency"].values,
        "z_real": result.fields["impedance_real"].values,
        "z_imag": result.fields["impedance_imag"].values,
    }


@st.cache_data(show_spinner="Segmenting the field of view…", max_entries=16)
def run_counter(
    n_cells: int, dead_fraction: float, image_size: int, pixel_size_um: float,
    min_radius_px: float, dilution: float, seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.cell_counter import CellCounter

    cfg = ExperimentConfig(
        name="counter", instrument="cell_counter",
        params={
            "n_cells": n_cells, "dead_fraction": dead_fraction,
            "image_size": image_size, "pixel_size": pixel_size_um * 1e-6,
            "min_radius_px": min_radius_px, "dilution_factor": dilution, "seed": seed,
        },
    )
    inst = CellCounter(cfg)
    result = inst.run()
    return {
        "metrics": result.metrics,
        "table": result.table,
        "image": np.asarray(inst.image),
        "labels": np.asarray(inst.segmentation.labels),
    }


@st.cache_data(show_spinner="Tracking cells frame by frame…", max_entries=12)
def run_tracker(
    n_frames: int, n_cells: int, image_size: int, speed_px: float,
    persistence: float, search_range: float, frame_interval_min: float, seed: int,
) -> dict[str, Any]:
    from biosim_lab.instruments.cell_tracker import CellTracker

    cfg = ExperimentConfig(
        name="tracker", instrument="cell_tracker",
        params={
            "n_frames": n_frames, "n_cells": n_cells, "image_size": image_size,
            "speed_px_per_frame": speed_px, "persistence": persistence,
            "search_range_px": search_range,
            "frame_interval": frame_interval_min * 60.0, "seed": seed,
        },
    )
    inst = CellTracker(cfg)
    result = inst.run()
    return {
        "metrics": result.metrics,
        "per_track": result.table,
        "tracks": inst.tracks,
        "first_frame": np.asarray(inst.movie[0]),
        "msd_lag": result.fields["lag"].values if "msd" in result.fields else None,
        "msd": result.fields["msd"].values if "msd" in result.fields else None,
    }



@st.cache_data(show_spinner="Re-running on fresh cell samples…", max_entries=16)
def run_replicates(n_replicates: int, **kwargs: Any) -> Any:
    """Replicate summary for one operating point, as a plain DataFrame."""
    from biosim_lab.instruments.saw_sorter.simulate import (
        SAWSorterParams,
        replicate_sorting,
    )

    params = SAWSorterParams(
        frequency=kwargs["frequency_mhz"] * 1e6,
        voltage_pp=kwargs["voltage_pp"],
        temperature=kwargs["temperature_c"] + 273.15,
        inlet_viability=kwargs["inlet_viability"],
        rf_power=kwargs["rf_power"] or None,
        channel_width=kwargs["width_um"] * 1e-6,
        channel_height=kwargs["height_um"] * 1e-6,
        channel_length=kwargs["length_mm"] * 1e-3,
        flow_rate=kwargs["flow_ul_min"] * UL_MIN,
        inlet=kwargs["inlet"],
        collection_fraction=kwargs["collection_fraction"],
        mode=kwargs["mode"],
        fem_resolution=kwargs["fem_resolution"],
        fem_grid=(201, 33),
        populations=[
            {"cell_type": kwargs["target"], "count": kwargs["n_cells"], "target": True},
            {"cell_type": kwargs["background"], "count": kwargs["n_cells"],
             "target": False},
        ],
        seed=kwargs["seed"],
    )
    return replicate_sorting(params, n_replicates=n_replicates)["table"]


