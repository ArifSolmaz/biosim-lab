"""Simulation driver, sorting metrics and parameter sweeps for the SAW sorter.

Geometry and frame
------------------
``x`` is the acoustic axis across the channel **width**, ``y`` the channel
**height**, ``z`` the flow direction.  Cells are advected along ``z`` by the
analytic Poiseuille profile while the acoustic radiation force displaces them
along ``x``; the simulation ends when a cell reaches ``z = channel_length``.

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
PNAS 112:4970, doi:10.1073/pnas.1504484112.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator
from scipy.interpolate import RegularGridInterpolator

from biosim_lab.core.config import (
    BaseConfigModel,
    Frequency,
    Length,
    Pressure,
    Temperature,
    Voltage,
    VolumeFlow,
)
from biosim_lab.core.environment import KELVIN, fluid_at, thermal_budget
from biosim_lab.core.materials import get_cell, get_fluid, get_substrate
from biosim_lab.core.particles import (
    ForceRegistry,
    LagrangianTracker,
    ParticleState,
    TrackResult,
    make_state,
)
from biosim_lab.core.plugin import ConfigurationError, RegimeWarning
from biosim_lab.core.statistics import replicate, summary_table, wilson_interval
from biosim_lab.instruments.saw_sorter import viability as viability_model
from biosim_lab.instruments.saw_sorter.fem_model import SAWFieldModel, pressure_from_voltage
from biosim_lab.instruments.saw_sorter.flow import RectangularPoiseuille
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    check_gorkov_validity,
    contrast_factor,
    effective_contrast_factor,
    node_positions,
    primary_radiation_force_1d,
    saw_wavelength,
)
from biosim_lab.instruments.saw_sorter.physics.drag import stokes_regime_report
from biosim_lab.instruments.saw_sorter.physics.secondary import (
    gravity_buoyancy,
    secondary_bjerknes_force,
    wall_repulsion,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


class Population(BaseConfigModel):
    """One cell population in the inlet suspension."""

    cell_type: str = Field(..., description="key from core.materials.CELL_TYPES")
    count: int = Field(200, ge=1, description="number of simulated cells")
    target: bool = Field(False, description="True for the population to be collected")
    label: str | None = Field(None, description="display name; defaults to cell_type")

    def resolved_label(self) -> str:
        return self.label or self.cell_type


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
    inlet: Literal["sheath_sides", "uniform", "centre", "side"] = "sheath_sides"
    inlet_side: Literal["left", "right"] = Field(
        "left", description="which wall the sample hugs when inlet='side'"
    )
    inlet_band: float = Field(
        0.15, gt=0, le=0.5, description="width of the inlet band as a fraction of the channel"
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

    # -- model options
    mode: Literal["analytic", "fem"] = "analytic"
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
    seed: int | None = 12345

    @model_validator(mode="after")
    def _check(self) -> SAWSorterParams:
        if self.pressure_amplitude is None and self.voltage_pp is None:
            raise ValueError("give either pressure_amplitude or voltage_pp")
        if not any(p.target for p in self.populations):
            raise ValueError("at least one population must be marked target=True")
        return self

    @property
    def p0(self) -> float:
        """Pressure amplitude [Pa], from the explicit value or the voltage calibration."""
        if self.pressure_amplitude is not None:
            return float(self.pressure_amplitude)
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

        if self.substrate.saw_velocity is None:
            raise ValueError(f"substrate {params.substrate!r} has no SAW velocity")
        self.saw_velocity = float(self.substrate.saw_velocity)
        self.wavelength = saw_wavelength(params.frequency, self.saw_velocity)
        self.node_offset = (
            0.5 * params.channel_width if params.node_offset is None else params.node_offset
        )
        self.flow = RectangularPoiseuille(
            params.channel_width, params.channel_height, self.fluid.mu, params.flow_rate
        )
        self._field_model: SAWFieldModel | None = None
        self._force_interp: dict[str, Any] = {}

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
        """Effective contrast factor for *cell* in this SSAW field."""
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
            cell = get_cell(pop.cell_type)
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
        return make_state(
            positions,
            np.concatenate(radii),
            np.concatenate(rhos),
            np.concatenate(kappas),
            np.asarray(labels, dtype=object),
            extra={"fluid_density": np.array([self.fluid.rho])},
        )

    def _sample_inlet_x(self, n: int, radii: np.ndarray) -> np.ndarray:
        """Inlet x-positions [m] according to the configured focusing scheme."""
        p = self.params
        w = p.channel_width
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

        if p.mode == "analytic":
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
            cell = get_cell(pop.cell_type)
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
        if p.mode == "fem":
            field_ds = self._prepare_fem()

        registry = self.build_forces()
        # The Gor'kov expansion parameter is k*a in the *fluid* — it is a scattering
        # criterion — not k_SAW*a, so the fluid wavelength is the right yardstick.
        check_gorkov_validity(state.radius, self.fluid.c / p.frequency)
        regime = stokes_regime_report(
            mean_velocity=self.flow.mean_velocity,
            width=p.channel_width,
            height=p.channel_height,
            radius=state.radius,
            rho_f=self.fluid.rho,
            viscosity=self.fluid.mu,
        )
        self._warn_if_multinode()

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
        tracks = tracker.run(state, (0.0, t_max), t_eval=t_eval)

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
        diagnostics = {
            "temperature_C": self.temperature_c,
            "viscosity_Pa_s": self.fluid.mu,
            "sound_speed_m_s": self.fluid.c,
            "density_kg_m3": self.fluid.rho,
            "thermal_budget": thermal_budget(
                pressure_amplitude=p.p0,
                frequency=p.frequency,
                fluid=self.fluid,
                channel_width=p.channel_width,
                channel_height=p.channel_height,
                channel_length=p.channel_length,
                flow_rate=p.flow_rate,
                rf_power=p.rf_power,
            ),
            "saw_wavelength_m": self.wavelength,
            "node_spacing_m": 0.5 * self.wavelength,
            "node_positions_m": node_positions(
                p.channel_width, self.wavelength, node_offset=self.node_offset
            ).tolist(),
            "pressure_amplitude_Pa": p.p0,
            "mean_velocity_m_s": self.flow.mean_velocity,
            "max_velocity_m_s": self.flow.max_velocity,
            "transit_time_s": self.flow.transit_time(p.channel_length),
            "pressure_gradient_Pa_m": self.flow.pressure_gradient,
            "contrast_factors": {
                pop.resolved_label(): {
                    "phi_classical": float(
                        contrast_factor(
                            get_cell(pop.cell_type).rho,
                            self.fluid.rho,
                            get_cell(pop.cell_type).kappa,
                            self.fluid.kappa,
                        )
                    ),
                    "phi_effective_ssaw": self.phi_for(get_cell(pop.cell_type)),
                }
                for pop in p.populations
            },
            "active_forces": registry.active,
            "mode": p.mode,
            "vertical_arf_enabled": p.enable_vertical_arf,
        }
        if self._field_model is not None:
            diagnostics["fem"] = self._field_model.results().diagnostics

        return SortingOutcome(
            tracks=tracks, cells=cells, metrics=metrics, field=field_ds,
            diagnostics=diagnostics, viability=viability,
        )

    def _warn_if_multinode(self) -> None:
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
        params = base.model_copy(update={}, deep=True)
        params = SAWSorterParams.model_validate({**params.model_dump(), **overrides})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            sim = SAWSorterSimulation(params)
            outcome = sim.run()
        row = {k: getattr(params, k) for k in keys}
        row.update(
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
