"""Simulation driver, sorting metrics and parameter sweeps for the SAW sorter.

Operating modes
---------------
One interface, three devices, chosen with ``mode``:

* ``"ssaw"`` (default) --- standing SAW, node planes parallel to the flow.
* ``"tassaw"`` --- tilted-angle SSAW: node planes at ``tilt_angle_deg`` to the
  flow, so a cell crosses many node--antinode regions and its deflection
  accumulates (Li et al. 2015, doi:10.1073/pnas.1504484112). Reproduced in
  ``benchmarks/benchmark_01_tassaw``.
* ``"alternating_baw"`` --- one piezoceramic switched in time between two bulk
  resonances of a hard-walled channel, e.g. 1 MHz (node at W/2) and 3 MHz
  (nodes at W/6, W/2, 5W/6), each phase with its own amplitude and duration,
  at least two switching cycles per cell (Zhang et al. 2023,
  doi:10.3390/ijms24043338). Integrated with fixed-step RK4 landing on every
  switch; each cell enters at its own point of the cycle. Reproduced in
  ``benchmarks/benchmark_02_alternating_baw``.

``field_model`` (``analytic``/``fem``) is independent of the mode; the FEM
cross-section is available for ``ssaw`` only. Configs written before 0.2, when
``mode`` meant the field model, still load (with a DeprecationWarning).

Geometry and frame
------------------
``x`` is the acoustic axis across the channel **width**, ``y`` the channel
**height**, ``z`` the flow direction.  Cells are advected along ``z`` by the
analytic Poiseuille profile while the acoustic radiation force displaces them
along ``x``; the simulation ends when a cell reaches ``z = channel_length``.

Straight and tilted IDTs
------------------------
``tilt_angle_deg`` selects between two genuinely different mechanisms, not two
geometries:

* **0 (default)** --- conventional SSAW. The wave runs across the channel, node
  planes lie parallel to the flow, and a cell migrates sideways to a node and
  stops. Displacement is capped by the node spacing.
* **non-zero** --- tilted-angle SSAW (doi:10.1073/pnas.1413325111). The node
  planes cross the flow, so a held cell is dragged across the channel as it
  travels: ``dx/dz = -tan(theta)``. Displacement grows with channel length, and
  the separation turns on whether a node can *hold* a cell at all. That limit
  scales with ``a^2``, so the small cells slip first --- see
  :func:`~biosim_lab.instruments.saw_sorter.physics.acoustics.max_trappable_tilt`
  and :func:`~...acoustics.cutoff_radius`, both reported in
  ``diagnostics["tilt"]``.

A tilted pattern varies along the flow, which the FEM cross-section cannot
represent, so ``field_model="fem"`` with a non-zero tilt is refused rather than
silently answering with a straight-IDT field.

Outlet layouts
--------------
Two chip topologies are supported, chosen with ``outlet_layout``:

* ``centre_band`` (default) — three outlets. A band centred on the pressure node
  creams off the cells that reached it; the sample usually enters at both walls
  (``inlet="sheath_sides"``).
* ``lateral_split`` — two outlets divided by a single line across the channel.
  The sample enters along one wall (``inlet="side"``), large cells cross toward
  the node and small ones do not, so one side of the divider holds the large
  cells and the other holds everything else.

For a lateral split the divider does **not** belong on the node: cells approach a
node asymptotically and settle a few microns short of it, so a divider placed on
it collects nothing (the model warns rather than reporting a bare zero). Put it
between the two populations' landing positions --- see
``examples/09_two_outlet_split.py``, which sweeps it.

Sorting metrics
---------------
With a target population T and a background population B, and a collection
outlet C:

* **Efficiency (recovery)** ``= N_T(C) / N_T`` — the fraction of target cells
  that end up in the collection outlet.
* **Purity** ``= N_T(C) / N(C)`` — the fraction of the collected cells that are
  targets.
* **Enrichment** ``= purity / (N_T / N)`` — fold improvement over the input
  ratio; the number a CTC assay actually cares about.

These are the definitions used in the CTC-separation literature, e.g.
Li et al. (2015), *Acoustic separation of circulating tumor cells*,
PNAS 112:4970, doi:10.1073/pnas.1504484112. The papers' own names ---
capture efficiency, contamination rate, recovery rate, removal rate,
separation distance --- are computed alongside, with their exact definitions,
by :mod:`biosim_lab.instruments.saw_sorter.metrics`.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator
from scipy.interpolate import RegularGridInterpolator

from biosim_lab.core.config import (
    BaseConfigModel,
    Compressibility,
    Density,
    EnergyDensity,
    Frequency,
    Length,
    Pressure,
    Temperature,
    Time,
    Voltage,
    VolumeFlow,
)
from biosim_lab.core.environment import KELVIN, fluid_at, thermal_budget
from biosim_lab.core.materials import (
    CellType,
    Provenance,
    Value,
    get_cell,
    get_fluid,
    get_substrate,
)
from biosim_lab.core.particles import (
    ForceRegistry,
    LagrangianTracker,
    ParticleState,
    TrackResult,
    make_state,
)
from biosim_lab.core.plugin import ConfigurationError, RegimeWarning
from biosim_lab.core.samples import compose, dilution_report
from biosim_lab.core.statistics import replicate, summary_table, wilson_interval
from biosim_lab.instruments.saw_sorter import metrics as paper
from biosim_lab.instruments.saw_sorter import viability as viability_model
from biosim_lab.instruments.saw_sorter.fem_model import SAWFieldModel, pressure_from_voltage
from biosim_lab.instruments.saw_sorter.flow import RectangularPoiseuille
from biosim_lab.instruments.saw_sorter.physics import baw
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    bruus_phi,
    check_gorkov_validity,
    contrast_factor,
    cutoff_radius,
    dbm_to_watt,
    effective_contrast_factor,
    max_trappable_tilt,
    node_positions,
    pressure_from_rf_power,
    primary_radiation_force_1d,
    saw_wavelength,
)
from biosim_lab.instruments.saw_sorter.physics.baw import (
    energy_density_from_voltage,
    pressure_from_energy_density,
)
from biosim_lab.instruments.saw_sorter.physics.drag import stokes_regime_report
from biosim_lab.instruments.saw_sorter.physics.secondary import (
    gravity_buoyancy,
    secondary_bjerknes_force,
    wall_repulsion,
)
from biosim_lab.instruments.saw_sorter.switching import Schedule

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


class Population(BaseConfigModel):
    """One cell population in the inlet suspension.

    Property overrides
    ------------------
    A benchmark often needs a cell *as a particular paper measured it* rather
    than the library's default for that line --- Li 2015's leukocytes are
    ~12 um, the library's mixed-leukocyte default is 8.5 um. ``diameter``,
    ``diameter_cv``, ``density`` and ``compressibility`` replace the library
    value for this population only, and ``override_source`` is then required:
    the provenance rule (every number sourced or flagged) applies to overrides
    exactly as it does to the library. Give ``override_doi`` when the source is
    a paper; without it the override is recorded as an ASSUMPTION.
    """

    cell_type: str = Field(..., description="key from core.materials.CELL_TYPES")
    count: int = Field(200, ge=1, description="number of simulated cells")
    abundance_per_ml: float | None = Field(
        None, ge=0.0,
        description="how many of these are in one mL of the REAL sample. Separate "
        "from `count`, which is only how many to simulate: a rare population needs "
        "many simulated cells for a usable probability and has almost none in the "
        "tube. Give it for every population to get metrics at the real ratio",
    )
    target: bool = Field(False, description="True for the population to be collected")
    label: str | None = Field(None, description="display name; defaults to cell_type")

    diameter: Length | None = Field(None, description="override the library mean diameter")
    diameter_cv: float | None = Field(
        None, ge=0.0, lt=1.0, description="override the size CV (0 = monodisperse)"
    )
    density: Density | None = Field(None, description="override the library density")
    compressibility: Compressibility | None = Field(
        None, description="override the library isentropic compressibility"
    )
    override_source: str | None = Field(
        None, description="where the overriding numbers come from; required with any override"
    )
    override_doi: str | None = Field(None, description="DOI backing override_source, if any")

    @model_validator(mode="after")
    def _overrides_need_a_source(self) -> Population:
        if self.has_overrides and not (self.override_source or "").strip():
            raise ValueError(
                f"population {self.resolved_label()!r} overrides library properties "
                "without saying where the numbers come from; set override_source "
                "(and override_doi if it is a paper)"
            )
        return self

    @property
    def has_overrides(self) -> bool:
        return any(v is not None for v in (
            self.diameter, self.diameter_cv, self.density, self.compressibility
        ))

    def resolved_label(self) -> str:
        return self.label or self.cell_type

    def resolved_cell(self) -> CellType:
        """The library cell with this population's overrides applied."""
        cell = get_cell(self.cell_type)
        if not self.has_overrides:
            return cell
        prov = (
            Provenance(doi=self.override_doi, citation=str(self.override_source))
            if self.override_doi else Provenance(assumption=str(self.override_source))
        )
        changes: dict[str, Any] = {}
        if self.diameter is not None:
            changes["radius_mean"] = Value(0.5 * float(self.diameter), "m", prov)
        if self.diameter_cv is not None:
            changes["radius_cv"] = Value(float(self.diameter_cv), "dimensionless", prov)
        if self.density is not None:
            changes["density"] = Value(float(self.density), "kg/m**3", prov)
        if self.compressibility is not None:
            changes["compressibility"] = Value(float(self.compressibility), "1/Pa", prov)
        return replace(cell, **changes)


class PowerDrive(BaseConfigModel):
    """RF drive given as power, converted to a pressure through one reference point.

    ``p0 = p_ref * sqrt((P / P_ref) * (L_ref / L))`` --- see
    :func:`~biosim_lab.instruments.saw_sorter.physics.acoustics.pressure_from_rf_power`
    for which half of that is physics and which half is an assumption.
    """

    power_dbm: float = Field(..., description="RF power delivered to the IDTs [dBm]")
    reference_pressure: Pressure = Field(
        ..., description="pressure amplitude measured (or calibrated) at the reference point"
    )
    reference_power_dbm: float = Field(..., description="power at the reference point [dBm]")
    reference_idt_length: Length = Field(
        ..., description="IDT length at the reference point; channel_length is the actual one"
    )
    reference_source: str = Field(
        ..., min_length=1, description="how reference_pressure was obtained (measurement, "
        "calibration run, ...); it is never a published number for a dBm-driven device",
    )


class FrequencyPhase(BaseConfigModel):
    """One phase of a switched drive: a frequency held for a duration."""

    frequency: Frequency
    duration: Time
    voltage_pp: Voltage = Field(..., description="drive amplitude during this phase")
    reference_energy_density: EnergyDensity = Field(
        ..., description="E_ac at reference_voltage_pp for this frequency (device-specific)"
    )
    reference_voltage_pp: Voltage = Field(
        ..., description="voltage the reference E_ac holds at"
    )
    energy_source: str = Field(
        ..., min_length=1, description="where reference_energy_density comes from"
    )
    label: str | None = None

    def energy_density(self) -> float:
        """``E_ac`` at this phase's voltage, ``E_ref (U/U_ref)^2`` (doi:10.1039/b920376a)."""
        return energy_density_from_voltage(
            self.voltage_pp,
            reference_voltage_pp=self.reference_voltage_pp,
            reference_energy_density=self.reference_energy_density,
        )

    def resolved_label(self) -> str:
        return self.label or f"{self.frequency / 1e6:g} MHz"


class SwitchingProtocol(BaseConfigModel):
    """The drive of ``mode="alternating_baw"``: phases played in a cycle.

    ``min_cycles`` enforces the requirement of Zhang et al. (2023,
    doi:10.3390/ijms24043338, Sec. 4.2) that "cells must undergo at least two
    switching cycles in the acoustic field": a configuration whose fastest cell
    sees fewer is refused, with the three ways to fix it.
    """

    phases: list[FrequencyPhase] = Field(..., min_length=2)
    min_cycles: int = Field(2, ge=2, description="switching cycles every cell must see")
    entry: Literal["random", "fixed"] = Field(
        "random",
        description="random: each cell enters at a uniformly random point of the cycle "
        "(a real device). fixed: every cell enters at entry_time_in_cycle",
    )
    entry_time_in_cycle: Time = Field(
        0.0, ge=0.0, description="cycle time at entry when entry='fixed' (0 = start of phase 1)"
    )
    integrator: Literal["rk4", "solve_ivp"] = Field(
        "rk4", description="rk4: fixed-step, lands on every switch (the paper's method)"
    )
    time_step: Time = Field(2e-3, gt=0.0, description="RK4 step")

    @property
    def period(self) -> float:
        return float(sum(p.duration for p in self.phases))


class SAWSorterParams(BaseConfigModel):
    """Configuration block for the ``saw_sorter`` instrument."""

    # -- acoustics
    frequency: Frequency = Field(6.632e6, description="IDT drive frequency")
    voltage_pp: Voltage | None = Field(
        15.0, description="IDT drive voltage; ignored when pressure_amplitude is given"
    )
    pressure_amplitude: Pressure | None = Field(
        None, description="acoustic pressure amplitude p0; overrides voltage_pp"
    )
    substrate: str = "linbo3_128yx"
    node_offset: Length | None = Field(
        None, description="pressure-node position along x; defaults to the channel centre"
    )

    # -- environment
    temperature: Temperature = Field(
        298.15,
        description="fluid temperature; accepts '37 degC'. Changes viscosity and "
        "therefore migration speed by tens of percent",
    )
    rf_power: float | None = Field(
        None,
        description="applied RF power [W], used only for the flagged transducer-"
        "heating estimate; None omits it",
    )

    # -- viability
    inlet_viability: float = Field(
        0.95, ge=0.0, le=1.0,
        description="fraction of the sample already alive before the device",
    )
    track_viability: bool = Field(
        True, description="compute thermal, shear and cavitation damage per cell"
    )

    # -- geometry and flow
    channel_width: Length = 300e-6
    channel_height: Length = 50e-6
    channel_length: Length = 2e-3  # acoustically active length = IDT aperture
    flow_rate: VolumeFlow = 8.333333333333333e-11  # 5 uL/min
    fluid: str = "water"

    # -- cells
    populations: list[Population] = Field(
        default_factory=lambda: [
            Population(cell_type="mcf7", count=200, target=True),
            Population(cell_type="rbc", count=200, target=False),
        ]
    )
    tilt_angle_deg: float = Field(
        0.0, gt=-90.0, lt=90.0,
        description="angle between the SAW propagation direction and the channel "
        "width axis. 0 = conventional SSAW, nodes parallel to the flow. Non-zero "
        "= tilted-angle SSAW (taSSAW), a different separation mechanism",
    )
    inlet: Literal["sheath_sides", "uniform", "centre", "side"] = "sheath_sides"
    inlet_side: Literal["left", "right"] = Field(
        "left", description="which wall the sample hugs when inlet='side'"
    )
    inlet_band: float = Field(
        0.15, gt=0, le=0.5, description="width of the inlet band as a fraction of the channel"
    )
    sheath_ratio: float | None = Field(
        None, gt=0.0,
        description="sheath:sample flow-rate ratio. When given, the inlet band is DERIVED "
        "from it through the Poiseuille profile (the sample occupies the fraction "
        "1/(1+ratio) of the flux) and inlet_band is ignored",
    )
    inlet_x: list[float] | None = Field(
        None,
        description="explicit inlet positions across the channel, as fractions of the "
        "width; each population takes them in order (cycled if it has more cells). For "
        "tracing chosen starting points, e.g. a trajectory figure; overrides the band",
    )
    inlet_weighting: Literal["uniform", "flux"] = Field(
        "uniform",
        description="uniform: inlet positions uniform over the band. flux: weighted by the "
        "local velocity, i.e. cells per unit time rather than per unit area --- the "
        "right weighting for rates counted at an outlet",
    )

    # -- outlets
    outlet_layout: Literal["centre_band", "lateral_split"] = Field(
        "centre_band",
        description="centre_band: three outlets, the middle one collects at the node. "
        "lateral_split: two outlets divided by one line across the channel",
    )
    collection_fraction: float = Field(
        1.0 / 3.0, gt=0, lt=1,
        description="width of the central collection outlet / channel; "
        "centre_band layout only",
    )
    split_position: float = Field(
        0.5, gt=0, lt=1,
        description="where the divider sits, as a fraction of the channel width; "
        "lateral_split layout only",
    )
    collect_side: Literal["left", "right"] = Field(
        "right",
        description="which side of the divider is the collection outlet; "
        "lateral_split layout only",
    )

    # -- operating mode
    mode: Literal["ssaw", "tassaw", "alternating_baw"] = Field(
        "ssaw",
        description="ssaw: standing SAW, nodes parallel to the flow. tassaw: tilted-angle "
        "SSAW, node planes at tilt_angle_deg to the flow (doi:10.1073/pnas.1504484112). "
        "alternating_baw: one piezoceramic switched between channel resonances "
        "(doi:10.3390/ijms24043338); needs `switching`",
    )
    power_drive: PowerDrive | None = Field(
        None, description="give the drive as RF power instead of voltage/pressure"
    )
    switching: SwitchingProtocol | None = Field(
        None, description="frequency phases for mode='alternating_baw'"
    )

    # -- model options
    field_model: Literal["analytic", "fem"] = Field(
        "analytic", description="analytic standing wave, or a Gmsh + scikit-fem Helmholtz solve"
    )
    integration: Literal["overdamped", "inertial"] = "overdamped"
    enable_vertical_arf: bool = Field(
        False,
        description=(
            "apply the vertical (y) component of the FEM Gor'kov force. Off by "
            "default: see SAWSorterSimulation._fem_arf for why"
        ),
    )
    enable_gravity: bool = False
    enable_wall_repulsion: bool = False
    enable_secondary_bjerknes: bool = False
    fem_resolution: int = Field(40, ge=8, le=400)
    fem_grid: tuple[int, int] = (241, 41)
    n_time_samples: int = Field(101, ge=11)
    record_forces: bool = Field(
        False, description="keep every force (and the Stokes drag) along each trajectory"
    )
    seed: int | None = 12345

    @model_validator(mode="before")
    @classmethod
    def _resolve_legacy_mode(cls, data: Any) -> Any:
        """Accept configs written before ``mode`` meant the operating mode.

        Until 0.2, ``mode`` chose the *field model* (``analytic``/``fem``) and a
        non-zero tilt silently meant taSSAW. Those configs keep working: the old
        values move to ``field_model``, and an unstated mode is inferred from
        the tilt. An explicit mode that contradicts the tilt is still an error.
        """
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if d.get("mode") in ("analytic", "fem"):
            legacy = d.pop("mode")
            if d.get("field_model", legacy) != legacy:
                raise ValueError(
                    f"mode={legacy!r} (legacy spelling of field_model) contradicts "
                    f"field_model={d['field_model']!r}"
                )
            d["field_model"] = legacy
            warnings.warn(
                f"mode={legacy!r} is the pre-0.2 spelling; use field_model={legacy!r}. "
                "`mode` now selects the operating mode: ssaw, tassaw or alternating_baw.",
                DeprecationWarning,
                stacklevel=2,
            )
        if d.get("mode") is None:
            tilt = d.get("tilt_angle_deg", 0.0)
            try:
                tilted = float(tilt or 0.0) != 0.0
            except (TypeError, ValueError):
                tilted = False
            d["mode"] = "tassaw" if tilted else "ssaw"
        return d

    @model_validator(mode="after")
    def _check(self) -> SAWSorterParams:
        if (self.pressure_amplitude is None and self.voltage_pp is None
                and self.power_drive is None and self.mode != "alternating_baw"):
            raise ValueError("give pressure_amplitude or voltage_pp (or a power_drive)")
        if not any(p.target for p in self.populations):
            raise ValueError("at least one population must be marked target=True")
        if self.mode == "ssaw" and self.tilt_angle_deg != 0.0:
            raise ValueError(
                f"mode='ssaw' with tilt_angle_deg={self.tilt_angle_deg:g}: a tilted IDT "
                "is the taSSAW mechanism, set mode='tassaw' (or omit mode)"
            )
        if self.mode == "tassaw" and self.tilt_angle_deg == 0.0:
            raise ValueError(
                "mode='tassaw' needs a non-zero tilt_angle_deg; at 0 the node planes are "
                "parallel to the flow and the device is a conventional SSAW sorter"
            )
        if self.mode == "tassaw" and self.field_model == "fem":
            # The FEM field is solved on the (x, y) cross-section and is
            # invariant along the flow. A tilted pattern varies with z by
            # construction, so that mesh cannot represent it — and the failure
            # would be silent, returning a straight-IDT field under a tilted
            # label, which is exactly the kind of wrong answer that looks right.
            raise ValueError(
                "field_model='fem' cannot represent a tilted pattern: the Helmholtz "
                "field is solved on the channel cross-section and does not vary "
                "along the flow, whereas tilting makes it vary along the flow by "
                "definition. Use field_model='analytic' for mode='tassaw'."
            )
        if self.mode == "alternating_baw":
            if self.switching is None:
                raise ValueError(
                    "mode='alternating_baw' needs a `switching` block: at least two "
                    "frequency phases, each with a duration, voltage and E_ac calibration"
                )
            if self.field_model == "fem":
                raise ValueError(
                    "field_model='fem' is the leaky-SAW Helmholtz model; the BAW "
                    "resonance of mode='alternating_baw' is analytic only"
                )
            if self.tilt_angle_deg != 0.0:
                raise ValueError("tilt_angle_deg applies to SAW modes, not alternating_baw")
        elif self.switching is not None:
            raise ValueError("`switching` is only used by mode='alternating_baw'")
        return self

    @property
    def mode_label(self) -> str:
        return {"ssaw": "standing SAW", "tassaw": "tilted-angle SSAW",
                "alternating_baw": "alternating-frequency BAW"}[self.mode]

    @property
    def p0(self) -> float:
        """Pressure amplitude [Pa], from the explicit value, the RF drive or the voltage.

        For ``alternating_baw`` this is the largest phase amplitude, which is
        what the heating and cavitation estimates need.
        """
        if self.pressure_amplitude is not None:
            return float(self.pressure_amplitude)
        if self.power_drive is not None:
            d = self.power_drive
            return pressure_from_rf_power(
                d.power_dbm,
                idt_length=self.channel_length,
                reference_pressure=d.reference_pressure,
                reference_power_dbm=d.reference_power_dbm,
                reference_idt_length=d.reference_idt_length,
            )
        if self.mode == "alternating_baw" and self.switching is not None:
            fluid = get_fluid(self.fluid)
            return max(
                pressure_from_energy_density(ph.energy_density(), fluid.rho, fluid.c)
                for ph in self.switching.phases
            )
        return pressure_from_voltage(float(self.voltage_pp))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------


@dataclass
class SortingOutcome:
    """Everything one sorter run produces."""

    tracks: TrackResult
    cells: pd.DataFrame
    metrics: dict[str, Any]
    field: xr.Dataset | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    viability: viability_model.ViabilityReport | None = None
    forces: xr.Dataset | None = None


class SAWSorterSimulation:
    """Build and run one SSAW sorting experiment."""

    def __init__(self, params: SAWSorterParams) -> None:
        self.params = params
        self.tabulated_fluid = get_fluid(params.fluid)
        self.temperature_c = params.temperature - KELVIN
        # Every downstream calculation uses the fluid AT THE RUN TEMPERATURE.
        # Viscosity alone changes by 22 % between the bench and an incubator, and
        # migration speed is inversely proportional to it.
        self.fluid = fluid_at(self.tabulated_fluid, self.temperature_c)
        self.substrate = get_substrate(params.substrate)
        self.rng = np.random.default_rng(params.seed)
        self.cells_by_label: dict[str, CellType] = {
            pop.resolved_label(): pop.resolved_cell() for pop in params.populations
        }
        self.node_offset = (
            0.5 * params.channel_width if params.node_offset is None else params.node_offset
        )
        self.flow = RectangularPoiseuille(
            params.channel_width, params.channel_height, self.fluid.mu, params.flow_rate
        )
        self._field_model: SAWFieldModel | None = None
        self._force_interp: dict[str, Any] = {}
        self.schedule: Schedule | None = None

        if params.mode == "alternating_baw":
            # A bulk resonance: the fluid, not a substrate wave, sets the node
            # pattern, so there is no SAW velocity to look up. The "wavelength"
            # quoted in diagnostics is the first phase's 2W/n.
            self.schedule = self._build_schedule()
            self.saw_velocity = float("nan")
            self.wavelength = 2.0 * params.channel_width / int(self.schedule.mode_numbers[0])
            self.node_offset = 0.5 * params.channel_width
        else:
            if self.substrate.saw_velocity is None:
                raise ValueError(f"substrate {params.substrate!r} has no SAW velocity")
            self.saw_velocity = float(self.substrate.saw_velocity)
            self.wavelength = saw_wavelength(params.frequency, self.saw_velocity)

    def _build_schedule(self) -> Schedule:
        """Resolve each switching phase to a channel resonance and an ``E_ac``."""
        sw = self.params.switching
        assert sw is not None
        w = self.params.channel_width
        with warnings.catch_warnings():
            warnings.simplefilter("always", RegimeWarning)
            modes = [baw.mode_number(ph.frequency, w, self.fluid.c) for ph in sw.phases]
        return Schedule(
            durations=np.array([ph.duration for ph in sw.phases], dtype=float),
            mode_numbers=np.array(modes, dtype=int),
            energy_densities=np.array([ph.energy_density() for ph in sw.phases], dtype=float),
            frequencies=np.array([ph.frequency for ph in sw.phases], dtype=float),
            labels=tuple(ph.resolved_label() for ph in sw.phases),
        )

    def cell_for(self, pop: Population) -> CellType:
        """The cell properties this run uses for *pop*, overrides applied."""
        return self.cells_by_label[pop.resolved_label()]

    @property
    def tilt_angle(self) -> float:
        """IDT tilt relative to the channel width axis [rad]."""
        return float(np.deg2rad(self.params.tilt_angle_deg))

    @property
    def lateral_node_spacing(self) -> float:
        """Distance between node planes measured ACROSS the channel [m].

        Tilting stretches the pattern as seen along ``x``: the wave still has
        wavelength ``lambda`` along its own direction, but the intersection of
        those planes with the channel width is spaced ``lambda / (2 cos theta)``.
        """
        return 0.5 * self.wavelength / float(np.cos(self.tilt_angle))

    # -- outlets ----------------------------------------------------------
    @property
    def collection_bounds(self) -> tuple[float, float]:
        """``(lo, hi)`` in metres of the collection outlet, whatever the layout.

        One definition, used by the outlet assignment and by every figure that
        shades the collected region. It was previously recomputed at seven call
        sites, which is how a second layout would have ended up drawn correctly
        in some plots and wrongly in others.

        ``centre_band`` is the three-outlet chip: a band centred on the pressure
        node creams off the cells that reached it. ``lateral_split`` is the
        two-outlet chip: one divider across the channel, everything on the
        chosen side is collected --- the layout you get when the sample enters
        along one wall and the large cells cross the channel to the node while
        the small ones do not.
        """
        p = self.params
        w = p.channel_width
        if p.outlet_layout == "lateral_split":
            divider = p.split_position * w
            return (divider, w) if p.collect_side == "right" else (0.0, divider)
        half = 0.5 * p.collection_fraction * w
        return (self.node_offset - half, self.node_offset + half)

    # -- wave numbers -----------------------------------------------------
    @property
    def k_transverse(self) -> float:
        """Standing-pattern wave number along ``x`` [1/m]."""
        return 2.0 * np.pi / self.wavelength

    @property
    def k_fluid(self) -> float:
        """``omega / c_f`` [1/m]."""
        return 2.0 * np.pi * self.params.frequency / self.fluid.c

    def phi_for(self, cell) -> float:
        """Contrast factor (classical normalisation) for *cell* in this device's field.

        An SSAW field is not a plane wave --- its lateral periodicity is the
        substrate's, not the fluid's --- so it gets the corrected factor. A BAW
        resonance *is* a plane standing wave in the fluid, where the classical
        ``Phi`` holds exactly (doi:10.1039/c2lc21068a, eq. 24).
        """
        if self.params.mode == "alternating_baw":
            return float(
                contrast_factor(cell.rho, self.fluid.rho, cell.kappa, self.fluid.kappa)
            )
        return float(
            effective_contrast_factor(
                cell.rho,
                self.fluid.rho,
                cell.kappa,
                self.fluid.kappa,
                k_transverse=self.k_transverse,
                k_fluid=self.k_fluid,
            )
        )

    # -- ensemble ---------------------------------------------------------
    def build_ensemble(self) -> ParticleState:
        """Sample cell radii and inlet positions for every configured population."""
        p = self.params
        xs, ys, radii, rhos, kappas, labels = [], [], [], [], [], []
        for pop in p.populations:
            cell = self.cell_for(pop)
            r = np.asarray(cell.sample_radii(pop.count, self.rng), dtype=float)

            # A cell taller than the channel cannot enter it: a real device holds
            # it at the inlet or clogs. Simulating one would put a sphere inside
            # a wall AND make the wall-clearance sampling below ill-posed
            # (low > high), so oversized cells are excluded and the loss is
            # reported rather than absorbed silently. Cell diameters are
            # log-normal, so at a tight channel height this is not a rare tail:
            # a 20 um channel rejects ~16 % of an MCF-7 population.
            smallest_um = 2.0 * float(r.min()) * 1e6
            fits = 2.0 * r <= p.channel_height
            n_excluded = int((~fits).sum())
            if n_excluded:
                warnings.warn(
                    f"{n_excluded} of {pop.count} {pop.resolved_label()} cells are "
                    f"taller than the {p.channel_height * 1e6:.0f} um channel and "
                    "were excluded from the run; they could not enter the device. "
                    "Metrics below describe only the cells that fit.",
                    RegimeWarning,
                    stacklevel=2,
                )
                r = r[fits]
            if r.size == 0:
                raise ConfigurationError(
                    f"no {pop.resolved_label()} cell fits in a "
                    f"{p.channel_height * 1e6:.0f} um channel (smallest sampled "
                    f"diameter {smallest_um:.1f} um); raise the channel height "
                    "or choose a smaller cell type"
                )

            n = int(r.size)
            if p.inlet_x:
                frac = np.resize(np.asarray(p.inlet_x, dtype=float), n)
                x = np.clip(frac * p.channel_width, r, p.channel_width - r)
                y = np.full(n, 0.5 * p.channel_height)
            elif p.inlet_weighting == "flux":
                x, y = self._sample_inlet_flux(r)
            else:
                x = self._sample_inlet_x(n, r)
                # Keep cell centres at least one radius from the top/bottom walls.
                y = self.rng.uniform(r, p.channel_height - r)
            xs.append(x)
            ys.append(y)
            radii.append(r)
            rhos.append(np.full(n, cell.rho))
            kappas.append(np.full(n, cell.kappa))
            labels.extend([pop.resolved_label()] * n)

        positions = np.column_stack(
            [np.concatenate(xs), np.concatenate(ys), np.zeros(sum(len(a) for a in xs))]
        )
        radius = np.concatenate(radii)
        density = np.concatenate(rhos)
        kappa = np.concatenate(kappas)
        extra: dict[str, np.ndarray] = {"fluid_density": np.array([self.fluid.rho])}
        if p.mode == "alternating_baw":
            extra["phi_bruus"] = np.asarray(
                bruus_phi(density, self.fluid.rho, kappa, self.fluid.kappa), dtype=float
            )
            extra["entry_offset"] = self._entry_offsets(radius.size)
        return make_state(
            positions, radius, density, kappa, np.asarray(labels, dtype=object), extra=extra,
        )

    def _entry_offsets(self, n: int) -> np.ndarray:
        """Cycle time at which each cell enters the acoustic region [s].

        Random entry is drawn on the RK4 grid, so that every cell's switching
        instants fall on step boundaries and no step straddles a switch.
        """
        sw = self.params.switching
        assert sw is not None and self.schedule is not None
        if sw.entry == "fixed":
            return np.full(n, float(sw.entry_time_in_cycle) % self.schedule.period)
        slots = max(1, int(round(self.schedule.period / sw.time_step)))
        return self.rng.integers(0, slots, size=n) * (self.schedule.period / slots)

    # -- inlet ------------------------------------------------------------
    def inlet_band_bounds(self) -> list[tuple[float, float]]:
        """The sample stream(s) at the inlet, ``[(lo, hi), ...]`` in metres.

        With ``sheath_ratio`` set, the band is where the sample's share of the
        *flux* goes: the sample occupies ``1/(1 + ratio)`` of the flow rate, and
        because the velocity falls to zero at the side walls a wall-hugging
        stream is wider than that fraction of the width. With a 1:2 sample to
        sheath ratio in the 737 um channel of Zhang et al. (2023,
        doi:10.3390/ijms24043338) this gives the ``y0 < W/3`` they state.
        """
        p = self.params
        w = p.channel_width
        if p.sheath_ratio is None:
            band = p.inlet_band * w
            if p.inlet == "side":
                return [(0.0, band)] if p.inlet_side == "left" else [(w - band, w)]
            if p.inlet == "centre":
                return [(0.5 * (w - band), 0.5 * (w + band))]
            if p.inlet == "uniform":
                return [(0.0, w)]
            return [(0.0, band), (w - band, w)]

        share = 1.0 / (1.0 + p.sheath_ratio)
        xs, ys, u = self.flow.profile(nx=801, ny=41)
        q = np.trapezoid(u, ys, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(xs))])
        cum /= cum[-1]

        def x_at(fraction: float) -> float:
            return float(np.interp(fraction, cum, xs))

        if p.inlet == "side":
            edge = x_at(share) if p.inlet_side == "left" else x_at(1.0 - share)
            return [(0.0, edge)] if p.inlet_side == "left" else [(edge, w)]
        if p.inlet == "centre":
            return [(x_at(0.5 - 0.5 * share), x_at(0.5 + 0.5 * share))]
        if p.inlet == "uniform":
            return [(0.0, w)]
        return [(0.0, x_at(0.5 * share)), (x_at(1.0 - 0.5 * share), w)]

    def _sample_inlet_flux(self, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flux-weighted inlet positions ``(x, y)`` inside the sample band(s).

        Cells enter the device at a rate proportional to the local velocity, so
        a slow near-wall streamline delivers fewer of them per second than the
        fast middle. Rejection sampling against ``u(x, y) / u_max`` draws
        exactly that density; each cell stays at least one radius from every
        wall.
        """
        p = self.params
        bands = self.inlet_band_bounds()
        widths = np.array([hi - lo for lo, hi in bands])
        n = radii.size
        x = np.empty(n)
        y = np.empty(n)
        todo = np.arange(n)
        u_max = self.flow.max_velocity
        for _ in range(200):
            if todo.size == 0:
                break
            r = radii[todo]
            which = self.rng.choice(len(bands), size=todo.size, p=widths / widths.sum())
            lo = np.array([bands[i][0] for i in which])
            hi = np.array([bands[i][1] for i in which])
            lo = np.maximum(lo, r)
            hi = np.minimum(hi, p.channel_width - r)
            hi = np.maximum(hi, lo)
            cx = self.rng.uniform(lo, hi)
            cy = self.rng.uniform(r, p.channel_height - r)
            accept = self.rng.uniform(0.0, 1.0, size=todo.size) * u_max \
                <= self.flow.velocity(cx, cy)
            x[todo[accept]] = cx[accept]
            y[todo[accept]] = cy[accept]
            todo = todo[~accept]
        if todo.size:  # pragma: no cover - needs a pathological band
            raise ConfigurationError("could not place cells in the inlet band")
        return x, y

    def _sample_inlet_x(self, n: int, radii: np.ndarray) -> np.ndarray:
        """Inlet x-positions [m] according to the configured focusing scheme."""
        p = self.params
        w = p.channel_width
        if p.sheath_ratio is not None:
            bands = self.inlet_band_bounds()
            widths = np.array([hi - lo for lo, hi in bands])
            which = self.rng.choice(len(bands), size=n, p=widths / widths.sum())
            lo = np.maximum([bands[i][0] for i in which], radii)
            hi = np.maximum(np.minimum([bands[i][1] for i in which], w - radii), lo)
            return self.rng.uniform(lo, hi)
        band = p.inlet_band * w
        if p.inlet == "uniform":
            return self.rng.uniform(radii, w - radii)
        if p.inlet == "side":
            # The whole sample enters hugging one wall, which is what a
            # two-outlet chip does: it gives every cell the full channel width
            # to migrate across, so the distance travelled --- and therefore the
            # size selectivity --- is as large as the geometry allows.
            if p.inlet_side == "left":
                return self.rng.uniform(radii, band)
            return self.rng.uniform(w - band, w - radii)
        if p.inlet == "centre":
            # size=n is required: both bounds are scalars here, so without it
            # Generator.uniform returns a single float rather than one position
            # per cell, and the caller concatenates zero-dimensional arrays.
            return self.rng.uniform(
                0.5 * w - 0.5 * band, 0.5 * w + 0.5 * band, size=n
            )
        # sheath_sides: two hydrodynamically focused streams hugging the side walls,
        # the standard SSAW sorter inlet (doi:10.1073/pnas.1504484112).
        side = self.rng.integers(0, 2, size=n)
        low = self.rng.uniform(radii, band)
        high = self.rng.uniform(w - band, w - radii)
        return np.where(side == 0, low, high)

    # -- forces -----------------------------------------------------------
    def build_forces(self) -> ForceRegistry:
        """Assemble the force registry for the configured mode."""
        p = self.params
        registry = ForceRegistry()

        if p.mode == "alternating_baw":
            registry.register(
                "acoustic_radiation",
                self._baw_arf,
                description="Gor'kov primary force, time-switched BAW resonances",
            )
        elif p.field_model == "analytic":
            registry.register(
                "acoustic_radiation",
                self._analytic_arf,
                description="Gor'kov primary force, analytic SSAW standing wave",
            )
        else:
            registry.register(
                "acoustic_radiation",
                self._fem_arf,
                description="Gor'kov primary force from the FEM Helmholtz field",
            )

        if p.enable_gravity:
            registry.register(
                "gravity_buoyancy",
                gravity_buoyancy(self.fluid.rho, axis=1),
                description="sedimentation (net weight minus buoyancy)",
            )
        if p.enable_wall_repulsion:
            registry.register(
                "wall_repulsion",
                wall_repulsion(
                    ((0.0, p.channel_width), (0.0, p.channel_height), None), stiffness=1e-6
                ),
                description="soft wall repulsion (numerical regulariser, not physics)",
            )
        if p.enable_secondary_bjerknes:
            registry.register(
                "secondary_bjerknes",
                secondary_bjerknes_force(
                    p0=p.p0,
                    wavelength=self.wavelength,
                    rho_f=self.fluid.rho,
                    kappa_f=self.fluid.kappa,
                    node_offset=self.node_offset,
                ),
                description="particle-particle acoustic interaction, O(N^2)",
            )
        return registry

    def _analytic_arf(self, state: ParticleState) -> np.ndarray:
        """Primary radiation force from the closed-form SSAW field."""
        phi = effective_contrast_factor(
            state.density,
            self.fluid.rho,
            state.compressibility,
            self.fluid.kappa,
            k_transverse=self.k_transverse,
            k_fluid=self.k_fluid,
        )
        theta = self.tilt_angle
        if theta == 0.0:
            # Conventional SSAW: the wave runs across the channel, so the node
            # planes lie parallel to the flow and the force is purely lateral.
            fx = primary_radiation_force_1d(
                state.x[:, 0],
                p0=self.params.p0,
                volume=state.volume,
                kappa_f=self.fluid.kappa,
                wavelength=self.wavelength,
                phi=phi,
                node_offset=self.node_offset,
            )
            out = np.zeros_like(state.x)
            out[:, 0] = fx
            return out

        # Tilted-angle SSAW (doi:10.1073/pnas.1413325111). Rotating the IDTs by
        # theta rotates the whole standing-wave pattern with them: the field
        # still varies sinusoidally with wavelength lambda, but along the
        # rotated propagation direction n = (cos theta, 0, sin theta) rather
        # than along x. So the same one-dimensional force law applies to the
        # PROJECTED coordinate, and the force points along n.
        #
        # The consequence is the mechanism itself. Node planes are no longer
        # parallel to the flow, so a particle held in one is dragged across the
        # channel as it travels downstream (dx/dz = -tan theta) instead of
        # parking at a fixed node. Displacement then grows with channel length
        # rather than saturating at the node, and the separation turns on
        # whether a particle can be held at all.
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xi = (state.x[:, 0] - self.node_offset) * cos_t + state.x[:, 2] * sin_t
        f_xi = primary_radiation_force_1d(
            xi,
            p0=self.params.p0,
            volume=state.volume,
            kappa_f=self.fluid.kappa,
            wavelength=self.wavelength,
            phi=phi,
            node_offset=0.0,
        )
        out = np.zeros_like(state.x)
        out[:, 0] = f_xi * cos_t
        out[:, 2] = f_xi * sin_t
        return out

    def _baw_arf(self, state: ParticleState) -> np.ndarray:
        """Radiation force of whichever resonance each cell currently sees.

        A cell's clock runs from its own entry, so the phase is looked up at
        ``t + entry_offset`` per cell (:class:`~.switching.Schedule`), and the
        mode number and ``E_ac`` vary along the ensemble. Force law and its
        source: :func:`~.physics.baw.radiation_force`.
        """
        assert self.schedule is not None
        idx = self.schedule.phase_index(state.t + state.extra["entry_offset"])
        out = np.zeros_like(state.x)
        out[:, 0] = baw.radiation_force(
            state.x[:, 0],
            n=self.schedule.mode_numbers[idx],
            width=self.params.channel_width,
            energy_density=self.schedule.energy_densities[idx],
            radius=state.radius,
            phi_bruus=state.extra["phi_bruus"],
        )
        return out

    def _fem_arf(self, state: ParticleState) -> np.ndarray:
        """Primary radiation force interpolated from the FEM Gor'kov field.

        The force is linear in particle volume at fixed material properties, so
        one field is solved per *population* (distinct density/compressibility)
        at a reference radius and rescaled by ``(r / r_ref)^3`` per cell.

        Why the vertical component is off by default
        --------------------------------------------
        The FEM pressure amplitude decays with height above the substrate (the
        leaky wave loses energy into the PDMS ceiling), so the Gor'kov potential
        has a real vertical gradient and positive-contrast cells are pushed
        towards the low-pressure region at the top of the channel.

        Nothing in this 2-D cross-section model balances that. A real device has
        inertial lift, wall lubrication and sedimentation holding cells at an
        equilibrium height; without them the cells pile up against a wall, where
        the Poiseuille axial velocity is zero, and **never reach the outlet** —
        which silently inflates the residence time and therefore the lateral
        displacement.

        The lateral force is the device physics and is always applied. Set
        ``enable_vertical_arf=True`` to include the vertical component, and
        check ``all_cells_exited`` in the metrics when you do.
        """
        out = np.zeros_like(state.x)
        p = self.params
        for key, (interp_x, interp_y, r_ref) in self._force_interp.items():
            sel = state.label == key
            if not np.any(sel):
                continue
            # The FEM force is only defined inside the channel; clamp rather than
            # extrapolate, which would diverge near the walls.
            pts = np.column_stack([
                np.clip(state.x[sel, 0], 0.0, p.channel_width),
                np.clip(state.x[sel, 1], 0.0, p.channel_height),
            ])
            scale = (state.radius[sel] / r_ref) ** 3
            out[sel, 0] = interp_x(pts) * scale
            if self.params.enable_vertical_arf:
                out[sel, 1] = interp_y(pts) * scale
        return out

    def _prepare_fem(self) -> xr.Dataset:
        """Solve the Helmholtz field once and build per-population force interpolators."""
        p = self.params
        model = SAWFieldModel(
            frequency=p.frequency,
            channel_width=p.channel_width,
            channel_height=p.channel_height,
            fluid=self.fluid,
            substrate=self.substrate,
            pressure_amplitude=p.p0,
            node_offset=self.node_offset,
            resolution=p.fem_resolution,
        )
        nx, ny = p.fem_grid
        model.solve(nx=nx, ny=ny)
        self._field_model = model

        merged: xr.Dataset | None = None
        for pop in p.populations:
            cell = self.cell_for(pop)
            ff = model.force_field(
                radius=cell.r, density=cell.rho, compressibility=cell.kappa
            )
            grid = (ff["x"].values, ff["y"].values)
            self._force_interp[pop.resolved_label()] = (
                RegularGridInterpolator(grid, ff["F_x"].values, bounds_error=False,
                                        fill_value=None),
                RegularGridInterpolator(grid, ff["F_y"].values, bounds_error=False,
                                        fill_value=None),
                cell.r,
            )
            tagged = ff.rename(
                {v: f"{v}_{pop.resolved_label()}" for v in ("F_x", "F_y", "gorkov_potential")}
            )
            merged = (tagged if merged is None
                      else xr.merge([merged, tagged], compat="override"))
        assert merged is not None
        merged.attrs.update(model.results().diagnostics)
        return merged

    # -- run --------------------------------------------------------------
    def run(self) -> SortingOutcome:
        """Execute the simulation and compute sorting metrics."""
        p = self.params
        state = self.build_ensemble()

        field_ds: xr.Dataset | None = None
        if p.field_model == "fem":
            field_ds = self._prepare_fem()

        registry = self.build_forces()
        # The Gor'kov expansion parameter is k*a in the *fluid* — it is a scattering
        # criterion — not k_SAW*a, so the fluid wavelength is the right yardstick.
        check_gorkov_validity(state.radius, self.fluid.c / self.drive_frequency)
        regime = stokes_regime_report(
            mean_velocity=self.flow.mean_velocity,
            width=p.channel_width,
            height=p.channel_height,
            radius=state.radius,
            rho_f=self.fluid.rho,
            viscosity=self.fluid.mu,
        )
        self._warn_if_multinode()
        self._warn_if_tilt_wastes_the_channel()

        def fluid_velocity(t: float, x: np.ndarray) -> np.ndarray:
            """Advection: zero in the cross-section, Poiseuille along z."""
            u = np.zeros_like(x)
            u[:, 2] = self.flow.velocity(x[:, 0], x[:, 1])
            return u

        tracker = LagrangianTracker(
            registry,
            fluid_velocity,
            self.fluid.mu,
            mode=p.integration,
            bounds=((0.0, p.channel_width), (0.0, p.channel_height), None),
            wall_clearance=True,
        )
        # The slowest cells hug a wall, where the Poiseuille velocity tends to zero,
        # so a multiple of the *mean* transit time is not enough. Size the integration
        # window from the slowest cell actually present, with a safety margin for the
        # acoustic migration that will speed it up.
        u_min = float(np.min(self.flow.velocity(state.x[:, 0], state.x[:, 1])))
        t_max = min(
            1.5 * p.channel_length / max(u_min, 1e-12),
            40.0 * self.flow.transit_time(p.channel_length),
        )
        # Sample densely while the cells are still inside the device and coarsely
        # afterwards: the window has to be long enough for the slowest wall-hugging
        # cell to exit, but uniform sampling would then leave the interesting part
        # of every fast trajectory with only a handful of points.
        t_device = min(4.0 * self.flow.transit_time(p.channel_length), t_max)
        t_eval = np.unique(
            np.concatenate([
                np.linspace(0.0, t_device, p.n_time_samples),
                np.linspace(t_device, t_max, max(p.n_time_samples // 4, 5)),
            ])
        )
        run_kwargs: dict[str, Any] = {}
        if p.mode == "alternating_baw":
            sw = p.switching
            assert sw is not None and self.schedule is not None
            self._require_min_cycles(state)
            dt = float(sw.time_step)
            # Snap the whole time grid to the RK4 step: with entry offsets also on
            # that grid, every switch of every cell is a step boundary.
            t_max = dt * np.ceil(t_max / dt)
            # Sample at least ten times per cycle --- the outlet position is
            # interpolated between samples, and cells move fastest right after
            # a switch --- on a 10 ms lattice, so the samples do not depend on
            # the step size and runs at different dt are directly comparable.
            lattice = 1e-2 if dt <= 1e-2 and abs(1e-2 / dt - round(1e-2 / dt)) < 1e-9 else dt
            spacing = min(t_device / p.n_time_samples, self.schedule.period / 10.0)
            spacing = max(lattice, lattice * np.floor(spacing / lattice))
            t_eval = np.unique(np.concatenate([
                np.arange(0.0, t_max + 0.5 * dt, spacing), [t_max],
            ]))
            offsets = state.extra["entry_offset"]
            breakpoints = self.schedule.switch_times(0.0, t_max, offsets)
            if sw.integrator == "rk4":
                length = p.channel_length
                run_kwargs = {
                    "integrator": "rk4", "dt": dt, "breakpoints": breakpoints,
                    # Past the outlet plane nothing can change the outcome.
                    "finished": lambda x: x[:, 2] >= length,
                }
            elif np.unique(offsets).size == 1:
                run_kwargs = {"breakpoints": breakpoints}
            # else: adaptive stepping through staggered switches, no restarts.
        tracks = tracker.run(state, (0.0, t_max), t_eval=t_eval, **run_kwargs)
        forces = tracker.force_history(tracks, state) if p.record_forces else None

        cells = self._cells_at_outlet(tracks, state)

        viability = None
        if p.track_viability:
            viability = self._assess_viability(cells, tracks)
            cells["alive_at_inlet"] = viability.alive_at_inlet
            cells["alive"] = viability.alive_at_outlet
            cells["survival_probability"] = viability.survival_probability
            cells["thermal_dose_cem43"] = viability.thermal_dose_cem43
            cells["shear_stress_Pa"] = viability.shear_stress_pa
        else:
            cells["alive_at_inlet"] = True
            cells["alive"] = True

        metrics = self.compute_metrics(cells)
        metrics.update(regime)
        if viability is not None:
            metrics.update(
                viability_in_percent=viability.viability_in_percent,
                viability_out_percent=viability.viability_out_percent,
                killed_by_device_percent=viability.killed_by_device_percent,
                **{k: v for k, v in viability.indicators.items()
                   if not isinstance(v, str)},
            )
        rf_power = p.rf_power
        if rf_power is None and p.power_drive is not None:
            rf_power = dbm_to_watt(p.power_drive.power_dbm)
        diagnostics = {
            "temperature_C": self.temperature_c,
            "viscosity_Pa_s": self.fluid.mu,
            "sound_speed_m_s": self.fluid.c,
            "density_kg_m3": self.fluid.rho,
            "thermal_budget": thermal_budget(
                pressure_amplitude=p.p0,
                frequency=self.drive_frequency,
                fluid=self.fluid,
                channel_width=p.channel_width,
                channel_height=p.channel_height,
                channel_length=p.channel_length,
                flow_rate=p.flow_rate,
                rf_power=rf_power,
            ),
            "saw_wavelength_m": self.wavelength,
            "node_spacing_m": 0.5 * self.wavelength,
            "node_positions_m": self.node_positions().tolist(),
            "pressure_amplitude_Pa": p.p0,
            "mean_velocity_m_s": self.flow.mean_velocity,
            "max_velocity_m_s": self.flow.max_velocity,
            "transit_time_s": self.flow.transit_time(p.channel_length),
            "pressure_gradient_Pa_m": self.flow.pressure_gradient,
            "inlet_bands_m": self.inlet_band_bounds(),
            "contrast_factors": {
                label: {
                    "phi_classical": float(
                        contrast_factor(cell.rho, self.fluid.rho, cell.kappa, self.fluid.kappa)
                    ),
                    "phi_bruus": float(
                        bruus_phi(cell.rho, self.fluid.rho, cell.kappa, self.fluid.kappa)
                    ),
                    "phi_effective": self.phi_for(cell),
                    # pre-0.2 name, kept so saved results and scripts still read
                    **({"phi_effective_ssaw": self.phi_for(cell)}
                       if p.mode != "alternating_baw" else {}),
                }
                for label, cell in self.cells_by_label.items()
            },
            "cell_properties": {
                label: {
                    "diameter_um": 2e6 * cell.r,
                    "diameter_cv": float(cell.radius_cv),
                    "density_kg_m3": cell.rho,
                    "compressibility_1_Pa": cell.kappa,
                    "sources": {
                        "diameter": str(cell.radius_mean.prov),
                        "compressibility": str(cell.compressibility.prov),
                    },
                }
                for label, cell in self.cells_by_label.items()
            },
            "active_forces": registry.active,
            "mode": p.mode,
            "field_model": p.field_model,
            "integrator": tracks.solver_info.get("integrator", "solve_ivp"),
            "vertical_arf_enabled": p.enable_vertical_arf,
            "tilt_angle_deg": p.tilt_angle_deg,
        }
        if p.mode == "alternating_baw":
            diagnostics["switching"] = self.switching_report(state)
        else:
            diagnostics["tilt"] = self.tilt_report()
        physiological = self.physiological_report(cells)
        if physiological is not None:
            diagnostics["physiological"] = physiological
        if self._field_model is not None:
            diagnostics["fem"] = self._field_model.results().diagnostics

        return SortingOutcome(
            tracks=tracks, cells=cells, metrics=metrics, field=field_ds,
            diagnostics=diagnostics, viability=viability, forces=forces,
        )

    # -- mode-aware helpers ----------------------------------------------
    @property
    def drive_frequency(self) -> float:
        """Frequency used for heating, cavitation and Gor'kov-validity checks [Hz].

        For a switched drive, the highest frequency: it has the shortest
        wavelength (the strictest ``k a``) and the strongest absorption.
        """
        if self.schedule is not None:
            return float(np.max(self.schedule.frequencies))
        return float(self.params.frequency)

    def node_positions(self) -> np.ndarray:
        """Pressure nodes across the channel [m] (the first phase's, for a switched drive)."""
        if self.schedule is not None:
            n_first = int(self.schedule.mode_numbers[0])
            return baw.node_positions(self.params.channel_width, n_first)
        return node_positions(
            self.params.channel_width, self.wavelength, node_offset=self.node_offset
        )

    def _require_min_cycles(self, state: ParticleState) -> None:
        """Refuse a switched run in which some cell sees too few cycles.

        "Cells must undergo at least two switching cycles in the acoustic field"
        (Zhang et al. 2023, doi:10.3390/ijms24043338, Sec. 4.2). The binding
        case is the fastest cell, so the check uses the largest axial velocity
        any simulated cell starts on.
        """
        sw = self.params.switching
        assert sw is not None and self.schedule is not None
        u_fast = float(np.max(self.flow.velocity(state.x[:, 0], state.x[:, 1])))
        t_fast = self.params.channel_length / max(u_fast, 1e-12)
        cycles = self.schedule.cycles_in(t_fast)
        if cycles + 1e-9 < sw.min_cycles:
            need_period = t_fast / sw.min_cycles
            raise ConfigurationError(
                f"the fastest cell crosses the {self.params.channel_length * 1e3:g} mm "
                f"acoustic region in {t_fast:.3g} s, which is {cycles:.2f} switching "
                f"cycles of {self.schedule.period:.3g} s; at least {sw.min_cycles} are "
                "required (doi:10.3390/ijms24043338, Sec. 4.2). Shorten the phases so "
                f"one cycle takes <= {need_period:.3g} s, lower the flow rate, or "
                "lengthen the acoustic region."
            )

    def switching_report(self, state: ParticleState | None = None) -> dict[str, Any]:
        """Phases, resonances, and the separation rule of a switched drive.

        The rule (Zhang et al. 2023, doi:10.3390/ijms24043338, Sec. 4.2):
        during the low-mode phase the target must move more than the distance
        from the high mode's first node to the edge of its basin --- ``W/6``
        for modes 1 and 3 --- and the background must move less, "thus
        ensuring that they are pulled to different nodes after switching". It is
        evaluated from the closed-form trajectory for each population's
        mean-sized cell starting on that node.
        """
        assert self.schedule is not None
        p = self.params
        w = p.channel_width
        sch = self.schedule
        phases = []
        for i, ph in enumerate(p.switching.phases):  # type: ignore[union-attr]
            n = int(sch.mode_numbers[i])
            phases.append({
                "label": sch.labels[i],
                "frequency_Hz": float(sch.frequencies[i]),
                "resonance_Hz": baw.resonance_frequency(n, w, self.fluid.c),
                "mode_number": n,
                "node_positions_um": (1e6 * baw.node_positions(w, n)).tolist(),
                "duration_s": float(sch.durations[i]),
                "voltage_pp": float(ph.voltage_pp),
                "energy_density_J_m3": float(sch.energy_densities[i]),
                "pressure_amplitude_Pa": pressure_from_energy_density(
                    float(sch.energy_densities[i]), self.fluid.rho, self.fluid.c
                ),
                "energy_source": ph.energy_source,
            })

        low = int(np.argmin(sch.mode_numbers))
        high = int(np.argmax(sch.mode_numbers))
        n_low, n_high = int(sch.mode_numbers[low]), int(sch.mode_numbers[high])
        start = w / (2.0 * n_high)          # first node of the sorting mode
        threshold = w / (2.0 * n_high)      # node to basin edge: W/6 for n = 3
        rule: dict[str, Any] = {
            "migration_phase": sch.labels[low], "sorting_phase": sch.labels[high],
            "start_um": 1e6 * start, "threshold_um": 1e6 * threshold,
            "populations": {},
        }
        for pop in p.populations:
            cell = self.cell_for(pop)
            phi_b = float(bruus_phi(cell.rho, self.fluid.rho, cell.kappa, self.fluid.kappa))
            disp = float(baw.displacement_in(
                float(sch.durations[low]), start, n=n_low, width=w,
                energy_density=float(sch.energy_densities[low]), radius=cell.r,
                phi_bruus=phi_b, viscosity=self.fluid.mu,
            ))
            ok = disp > threshold if pop.target else disp < threshold
            rule["populations"][pop.resolved_label()] = {
                "target": pop.target,
                "displacement_um": 1e6 * disp,
                "satisfied": bool(ok),
            }
        rule["satisfied"] = all(v["satisfied"] for v in rule["populations"].values())

        report: dict[str, Any] = {
            "period_s": sch.period,
            "phases": phases,
            "separation_rule": rule,
            "entry": p.switching.entry,  # type: ignore[union-attr]
        }
        if state is not None:
            u = self.flow.velocity(state.x[:, 0], state.x[:, 1])
            t_res = p.channel_length / np.maximum(u, 1e-12)
            report["cycles_seen"] = {
                "fastest_cell": float(sch.cycles_in(float(t_res.min()))),
                "median_cell": float(sch.cycles_in(float(np.median(t_res)))),
            }
        return report

    def physiological_report(self, cells: pd.DataFrame) -> dict[str, Any] | None:
        """Re-express the result at the sample's real composition.

        Returns ``None`` unless every population declares ``abundance_per_ml``.

        Simulated counts are chosen for statistics, not realism: a run with 300
        tumour cells and 300 red cells says nothing about a tube holding one
        tumour cell per billion. What transfers between the two is each
        population's *probability* of reaching the collection outlet, which is
        what the simulation actually measures. This applies those probabilities
        to the real abundances.

        It also checks the sample is dilute enough for the model to apply at all.
        Whole blood is 45 % cells by volume --- about two radii between
        neighbours --- where particle-particle scattering and streaming dominate
        and none of that is modelled here.
        """
        p = self.params
        abundances = {
            pop.resolved_label(): pop.abundance_per_ml
            for pop in p.populations
            if pop.abundance_per_ml is not None
        }
        if len(abundances) != len(p.populations):
            return None

        by_cell_type = {
            pop.cell_type: float(pop.abundance_per_ml or 0.0) for pop in p.populations
        }
        probability = {
            str(label): float((group["outlet"] == "collect").mean())
            for label, group in cells.groupby("label")
        }
        targets = {pop.resolved_label() for pop in p.populations if pop.target}

        simulated_counts = {
            str(label): int(len(group)) for label, group in cells.groupby("label")
        }
        report = compose(
            probability, abundances, targets=targets,
            simulated_counts=simulated_counts,
        )
        report["simulated_counts"] = simulated_counts
        report["collection_probability"] = probability
        dilution = dilution_report(by_cell_type)
        report["dilution"] = {
            "volume_fraction_percent": 100.0 * dilution.volume_fraction,
            "mean_separation_radii": dilution.mean_separation_radii,
            "required_dilution": dilution.required_dilution,
            "is_dilute": dilution.is_dilute,
            "summary": str(dilution),
        }
        if not dilution.is_dilute:
            warnings.warn(
                f"the declared sample is not a dilute suspension: "
                f"{100 * dilution.volume_fraction:.3g} % cells by volume, only "
                f"{dilution.mean_separation_radii:.1f} particle radii between "
                f"neighbours. This model treats cells as independent, which they "
                f"are not at that spacing --- particle-particle scattering and "
                f"acoustic streaming are not included. Dilute about "
                f"{dilution.required_dilution:.0f}x, or treat the numbers as an "
                "upper bound on what the device can do.",
                RegimeWarning,
                stacklevel=2,
            )
        return report

    def tilt_report(self) -> dict[str, Any]:
        """The design window for a tilted device: which cells a node can hold.

        In a tilted device the separation is not "how fast does it migrate" but
        "can a node keep hold of it at all". Each population has a maximum tilt,
        set by its radius, and any angle between the largest and the smallest
        separates them. This reports that window, and the cutoff diameter at the
        configured angle, so the choice does not have to be made by sweeping.
        """
        p = self.params
        flow_speed = p.flow_rate / (p.channel_width * p.channel_height)
        common = {
            "p0": p.p0,
            "kappa_f": self.fluid.kappa,
            "wavelength": self.wavelength,
            "viscosity": self.fluid.mu,
            "flow_speed": flow_speed,
        }
        per_population: dict[str, Any] = {}
        for pop in p.populations:
            cell = self.cell_for(pop)
            phi = self.phi_for(cell)
            per_population[pop.resolved_label()] = {
                "radius_um": float(cell.r) * 1e6,
                "max_trappable_tilt_deg": float(
                    np.degrees(max_trappable_tilt(float(cell.r), phi=phi, **common))
                ),
            }
        first_phi = self.phi_for(self.cell_for(p.populations[0]))
        report: dict[str, Any] = {
            "mean_flow_speed_m_s": flow_speed,
            "per_population": per_population,
        }
        if p.tilt_angle_deg != 0.0:
            radius = cutoff_radius(self.tilt_angle, phi=first_phi, **common)
            report["cutoff_radius_um"] = radius * 1e6
            report["cutoff_diameter_um"] = 2.0 * radius * 1e6
            report["lateral_node_spacing_um"] = self.lateral_node_spacing * 1e6
            report["geometric_drift_um"] = (
                p.channel_length * abs(np.tan(self.tilt_angle)) * 1e6
            )
        return report

    def _warn_if_tilt_wastes_the_channel(self) -> None:
        """Say so when the tilt deflects cells into the wall they entered on.

        A trapped particle drifts at ``dx/dz = -tan(theta)``, so a POSITIVE tilt
        moves cells toward -x and a negative one toward +x. Get the sign wrong
        with a one-wall inlet and every cell is pressed into the wall it started
        against: the run completes, the numbers look plausible, and the device
        has simply done nothing. Flipping the sign of the angle fixes it.
        """
        p = self.params
        if p.tilt_angle_deg == 0.0 or p.inlet != "side":
            return
        drifts_left = p.tilt_angle_deg > 0.0
        if (p.inlet_side == "left" and drifts_left) or (
            p.inlet_side == "right" and not drifts_left
        ):
            toward = "left" if drifts_left else "right"
            warnings.warn(
                f"tilt_angle_deg={p.tilt_angle_deg:+g} deflects cells toward the "
                f"{toward} wall, which is the wall the sample enters on "
                f"(inlet_side={p.inlet_side!r}). Cells are pressed into it and the "
                f"channel width is unused; flip the sign of the angle to deflect "
                "them across the channel instead.",
                RegimeWarning,
                stacklevel=2,
            )

    def _warn_if_multinode(self) -> None:
        # A tilted device carries cells ACROSS many nodes by design, and a BAW
        # device's node count is set by the resonance it is driven at.
        if self.params.mode != "ssaw":
            return
        nodes = node_positions(
            self.params.channel_width, self.wavelength, node_offset=self.node_offset
        )
        if nodes.size > 1:
            suggested = self.saw_velocity / (2.0 * self.params.channel_width)
            warnings.warn(
                f"{nodes.size} pressure nodes across the channel "
                f"(SAW wavelength {self.wavelength * 1e6:.1f} um). Two-outlet sorting assumes "
                f"one central node; use f = {suggested / 1e6:.2f} MHz for this width.",
                RegimeWarning,
                stacklevel=3,
            )

    def _cells_at_outlet(self, tracks: TrackResult, state: ParticleState) -> pd.DataFrame:
        """Interpolate each trajectory to the plane ``z = channel_length``."""
        p = self.params
        pos = tracks.trajectories["position"].values  # (n, t, 3)
        t = tracks.trajectories["time"].values
        z = pos[:, :, 2]
        exited = z[:, -1] >= p.channel_length

        x_out = np.empty(state.n)
        y_out = np.empty(state.n)
        t_out = np.full(state.n, np.nan)
        for i in range(state.n):
            if exited[i]:
                # z is monotonically increasing, so a plain interpolation is safe.
                t_cross = float(np.interp(p.channel_length, z[i], t))
                x_out[i] = float(np.interp(t_cross, t, pos[i, :, 0]))
                y_out[i] = float(np.interp(t_cross, t, pos[i, :, 1]))
                t_out[i] = t_cross
            else:
                x_out[i] = pos[i, -1, 0]
                y_out[i] = pos[i, -1, 1]

        lo, hi = self.collection_bounds
        outlet = np.where((x_out >= lo) & (x_out <= hi), "collect", "waste")

        if p.outlet_layout == "lateral_split" and not (outlet == "collect").any():
            # A divider placed at or beyond the node collects nothing, and the
            # reason is not obvious: cells approach the node asymptotically and
            # stop a few microns short of it, so "at the node" is already too
            # far. The bare 0 % / nan this produces otherwise looks like a
            # broken run rather than a divider in the wrong place.
            reached = float(np.min(np.abs(x_out - self.node_offset)))
            warnings.warn(
                f"the collection outlet caught no cells: the divider sits at "
                f"{p.split_position * p.channel_width * 1e6:.0f} um and nothing "
                f"crossed it. Cells stop short of the node rather than on it "
                f"(the closest stopped {reached * 1e6:.1f} um short), so place the "
                "divider between the two populations' final positions --- read "
                "them off the outlet histogram.",
                RegimeWarning,
                stacklevel=2,
            )

        df = tracks.final.copy()
        df["x_outlet_m"] = x_out
        df["y_outlet_m"] = y_out
        df["residence_time_s"] = t_out
        df["exited"] = exited
        df["outlet"] = outlet
        df["displacement_m"] = x_out - df["x_initial_m"]
        df["distance_to_node_m"] = np.abs(x_out - self.node_offset)
        df["is_target"] = df["label"].isin(
            [pop.resolved_label() for pop in p.populations if pop.target]
        )
        return df

    def _assess_viability(
        self, cells: pd.DataFrame, tracks: TrackResult
    ) -> viability_model.ViabilityReport:
        """Decide which cells arrive alive, from the exposure each one actually had.

        Shear is sampled along the whole trajectory rather than at the outlet,
        because a cell that spent the first millimetre against a wall took its
        damage there even if it finished in the middle.
        """
        pos = tracks.trajectories["position"].values  # (n, t, 3)
        shear_along_path = np.array([
            viability_model.shear_at(self.flow, pos[i, :, 0], pos[i, :, 1]).max()
            for i in range(pos.shape[0])
        ])
        residence = cells["residence_time_s"].to_numpy(dtype=float)
        return viability_model.assess(
            temperature_c=self.temperature_c,
            residence_time_s=residence,
            shear_stress_pa=shear_along_path,
            pressure_amplitude=self.params.p0,
            frequency=self.params.frequency,
            fluid=self.fluid,
            inlet_viability=self.params.inlet_viability,
            rng=self.rng,
        )

    # -- metrics ----------------------------------------------------------
    def compute_metrics(self, cells: pd.DataFrame) -> dict[str, Any]:
        """Efficiency, purity, enrichment and per-population outlet statistics."""
        n_total = len(cells)
        target = cells["is_target"]
        collected = cells["outlet"] == "collect"

        alive = cells["alive"] if "alive" in cells else pd.Series(True, index=cells.index)

        n_target = int(target.sum())
        n_collected = int(collected.sum())
        n_target_collected = int((target & collected).sum())
        n_live_target_collected = int((target & collected & alive).sum())
        n_live_collected = int((collected & alive).sum())

        efficiency = n_target_collected / n_target if n_target else float("nan")
        purity = n_target_collected / n_collected if n_collected else float("nan")
        input_ratio = n_target / n_total if n_total else float("nan")
        enrichment = purity / input_ratio if input_ratio else float("nan")

        per_pop = {}
        for label, group in cells.groupby("label"):
            per_pop[str(label)] = {
                "n": int(len(group)),
                "collected": int((group["outlet"] == "collect").sum()),
                "collected_fraction": float((group["outlet"] == "collect").mean()),
                "mean_displacement_um": float(group["displacement_m"].mean() * 1e6),
                "mean_abs_displacement_um": float(group["displacement_m"].abs().mean() * 1e6),
                "median_distance_to_node_um": float(
                    group["distance_to_node_m"].median() * 1e6
                ),
                "mean_outlet_x_um": float(group["x_outlet_m"].mean() * 1e6),
                "std_outlet_x_um": float(group["x_outlet_m"].std() * 1e6),
                "mean_radius_um": float(group["radius_m"].mean() * 1e6),
                "exited_fraction": float(group["exited"].mean()),
                "viable_fraction": float(group["alive"].mean())
                if "alive" in group else 1.0,
            }

        # A collected cell that is dead is of no use to the assay downstream, so
        # the live-cell figures are the ones a real workflow is judged on.
        n_live_target = int((target & alive).sum())
        live_efficiency = (
            n_live_target_collected / n_live_target if n_live_target else float("nan")
        )
        live_purity = (
            n_live_target_collected / n_live_collected if n_live_collected else float("nan")
        )

        # Even a perfectly deterministic device sorts a FINITE number of cells,
        # so every proportion here carries binomial counting error. Wilson rather
        # than the normal approximation, which gives a width of exactly zero at
        # 100 % — the case a working sorter hits constantly.
        efficiency_ci = wilson_interval(n_target_collected, n_target)
        purity_ci = wilson_interval(n_target_collected, n_collected)

        return {
            **paper.paper_metrics(cells),
            "n_cells": n_total,
            "n_target": n_target,
            "n_collected": n_collected,
            "n_alive_collected": n_live_collected,
            "n_live_target_collected": n_live_target_collected,
            "efficiency_percent": 100.0 * efficiency,
            "efficiency_ci_low": efficiency_ci.low_percent,
            "efficiency_ci_high": efficiency_ci.high_percent,
            "purity_percent": 100.0 * purity,
            "purity_ci_low": purity_ci.low_percent,
            "purity_ci_high": purity_ci.high_percent,
            "live_efficiency_percent": 100.0 * live_efficiency,
            "live_purity_percent": 100.0 * live_purity,
            "enrichment_fold": enrichment,
            "input_target_fraction_percent": 100.0 * input_ratio,
            "all_cells_exited": bool(cells["exited"].all()),
            "per_population": per_pop,
        }

    # -- histogram --------------------------------------------------------
    def outlet_histogram(self, cells: pd.DataFrame, bins: int = 40) -> xr.Dataset:
        """Outlet-position histogram per population, ready for plotting."""
        edges = np.linspace(0.0, self.params.channel_width, bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        labels = sorted(cells["label"].unique())
        counts = np.zeros((len(labels), bins), dtype=int)
        for i, label in enumerate(labels):
            sel = cells["label"] == label
            counts[i], _ = np.histogram(cells.loc[sel, "x_outlet_m"], bins=edges)
        ds = xr.Dataset(
            {"counts": (("population", "x"), counts)},
            coords={"population": labels, "x": centres},
        )
        ds["x"].attrs["units"] = "m"
        return ds


# ---------------------------------------------------------------------------
# parameter sweep
# ---------------------------------------------------------------------------


def parameter_sweep(
    base: SAWSorterParams,
    grid: dict[str, Sequence[Any]],
    *,
    progress: bool = False,
) -> tuple[pd.DataFrame, xr.Dataset]:
    """Run the sorter over a Cartesian grid of parameter values.

    Parameters
    ----------
    base:
        Configuration to start from; each sweep point overrides *grid* keys.
    grid:
        ``{"frequency": [...], "voltage_pp": [...], "flow_rate": [...]}``.
        Values may be unit strings — they pass through the same pydantic
        validation as the base config.
    progress:
        Print one line per sweep point.

    Returns
    -------
    (DataFrame, Dataset)
        A long table with one row per sweep point, and the same metrics folded
        into an N-dimensional :class:`xarray.Dataset` suitable for a heatmap.
    """
    if not grid:
        raise ValueError("sweep grid is empty")
    keys = list(grid)
    combos = list(itertools.product(*(list(grid[k]) for k in keys)))
    rows: list[dict[str, Any]] = []

    for i, combo in enumerate(combos, start=1):
        overrides = dict(zip(keys, combo))
        merged = {**base.model_dump(), **overrides}
        if "tilt_angle_deg" in overrides and "mode" not in overrides \
                and base.mode in ("ssaw", "tassaw"):
            # Sweeping the tilt through zero crosses between the two SAW
            # mechanisms; let the tilt decide rather than the base's label.
            merged.pop("mode")
        params = SAWSorterParams.model_validate(merged)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            sim = SAWSorterSimulation(params)
            outcome = sim.run()
        row = {k: getattr(params, k) for k in keys}
        row.update(
            capture_efficiency_percent=outcome.metrics["capture_efficiency_percent"],
            contamination_rate_percent=outcome.metrics["contamination_rate_percent"],
            recovery_rate_percent=outcome.metrics["recovery_rate_percent"],
            background_removal_percent=outcome.metrics["background_removal_percent"],
            separation_distance_um=outcome.metrics["separation_distance_um"],
            efficiency_percent=outcome.metrics["efficiency_percent"],
            purity_percent=outcome.metrics["purity_percent"],
            enrichment_fold=outcome.metrics["enrichment_fold"],
            n_collected=outcome.metrics["n_collected"],
            all_cells_exited=outcome.metrics["all_cells_exited"],
        )
        for label, stats in outcome.metrics["per_population"].items():
            row[f"collected_fraction_{label}"] = stats["collected_fraction"]
            row[f"mean_outlet_x_um_{label}"] = stats["mean_outlet_x_um"]
        rows.append(row)
        if progress:
            print(
                f"[{i}/{len(combos)}] "
                + ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                            for k, v in row.items() if k in keys)
                + f" -> eff {row['efficiency_percent']:.1f}% "
                f"purity {row['purity_percent']:.1f}%"
            )

    df = pd.DataFrame(rows)
    ds = df.set_index(keys).to_xarray()
    for k in keys:
        # NetCDF has no boolean attribute type; use an explicit flag string.
        ds[k].attrs["swept"] = "true"
    return df, ds


def replicate_sorting(
    params: SAWSorterParams,
    *,
    n_replicates: int = 5,
    confidence: float = 0.95,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the same device *n_replicates* times with different random samples.

    Two different uncertainties sit behind a single reported number, and this
    covers the second one:

    * **Counting error**, already in every run's metrics as ``*_ci_low`` /
      ``*_ci_high``: a proportion measured on a finite number of cells is not
      the population proportion, and that is true even with the seed fixed.
    * **Sample-to-sample error**, measured here: each replicate draws fresh
      radii from the log-normal size distribution, fresh inlet positions and
      fresh viability outcomes. The spread across replicates says how much of
      the result is the device and how much is the particular batch of cells.

    Quoting either alone is misleading, in opposite directions.

    Returns
    -------
    dict
        The structure from :func:`biosim_lab.core.statistics.replicate`, plus a
        ``table`` DataFrame ready to print or download.
    """
    default_metrics = (
        "efficiency_percent",
        "purity_percent",
        "live_purity_percent",
        "enrichment_fold",
        "viability_out_percent",
        "n_collected",
    )

    def run_once(seed: int) -> dict[str, Any]:
        replica = params.model_copy(update={"seed": seed})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            return SAWSorterSimulation(replica).run().metrics

    result = replicate(
        run_once,
        n_replicates=n_replicates,
        base_seed=int(params.seed or 0),
        metrics=list(metrics) if metrics is not None else list(default_metrics),
        confidence=confidence,
    )
    result["table"] = summary_table(result)
    return result


__all__ = [
    "Population",
    "SAWSorterParams",
    "SAWSorterSimulation",
    "SortingOutcome",
    "parameter_sweep",
    "replicate_sorting",
]
