"""``impedance_rtca`` — real-time cell analysis by electrode impedance.

Stage 2 status: a **minimal but working** instrument.  It simulates a 96-well
impedance plate: cells attach and proliferate (logistic coverage), the
Giaever-Keese model turns coverage into impedance, and the result is the Cell
Index time course a commercial RTCA instrument reports.  A drug added at a
chosen time reduces the carrying capacity in a dose-dependent way, from which
an IC50 is fitted.

It also reads **real** RTCA exports through
:func:`biosim_lab.core.io.read_rtca_csv`, so the same Cell Index / IC50 analysis
runs on measured plates.

Electrodes
----------
``electrode: interdigitated`` (default) is what an RTCA plate has: two equal
gold combs, so both interfaces are in series and the bulk resistance follows
the finger pattern (Olthuis et al. 1995, doi:10.1016/0925-4005(95)85053-8).
``electrode: disc`` is the classic ECIS layout (small working disc, large
counter electrode) kept for comparison.

``field_model: fem`` solves the electro-quasistatic Poisson problem on the IDE
(:mod:`.fem_model`) instead of the lumped series model; the Cell Index is then
read through a coverage -> impedance table computed by FEM at the readout
frequency. At 10 kHz the two agree to well under 1 %, because the interface
dominates there; above ~50 kHz current crowding at the finger edges makes the
lumped model a few percent low, and the spectra show it.

Not modelled (Stage 2 scope): cell micromotion noise, temperature drift, and
cells over the insulating gaps narrowing the electrolyte.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator

from biosim_lab.core.config import BaseConfigModel, ExperimentConfig, Frequency, Length, Time
from biosim_lab.core.io import normalise_well_label, read_rtca_csv
from biosim_lab.core.plugin import Instrument, InstrumentResult
from biosim_lab.instruments.impedance_rtca.physics import (
    RTCA_REFERENCE_IMPEDANCE_OHM,
    DoseResponseFit,
    ElectrodeGeometry,
    IDEGeometry,
    _parallel_coverage,
    alpha_parameter,
    cell_index,
    electrode_specific_impedance,
    fit_dose_response,
    giaever_keese_impedance,
    ide_bulk_impedance,
    ide_cell_constant,
    solution_resistance,
    wagner_number,
)


class RTCAParams(BaseConfigModel):
    """Configuration block for the ``impedance_rtca`` instrument."""

    # -- measurement
    frequency: Frequency = Field(1e4, description="single-frequency Cell Index readout")
    spectrum_frequencies: int = Field(60, ge=5, description="points in the |Z|(f) sweep")
    spectrum_range: tuple[float, float] = (1e2, 1e7)
    reference_impedance: float = RTCA_REFERENCE_IMPEDANCE_OHM

    # -- plate
    n_wells: Literal[96, 384] = 96
    duration: Time = 172800.0  # 48 h
    n_timepoints: int = Field(97, ge=5)

    # -- biology
    doubling_time: Time = 86400.0 / 1.2  # ~20 h, ASSUMPTION for a generic adherent line
    lag_time: Time = 7200.0  # 2 h attachment lag
    max_coverage: float = Field(0.95, gt=0, le=1.0)
    seeding_coverage: float = Field(0.05, gt=0, lt=1.0)

    # -- treatment
    treatment_time: Time | None = 86400.0
    concentrations: list[float] = Field(
        default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    )
    replicates: int = Field(3, ge=1)
    true_ic50: float = Field(1.0, gt=0, description="IC50 used to synthesise the plate")
    hill_slope: float = 1.3
    noise_cv: float = Field(0.02, ge=0.0, description="multiplicative measurement noise")

    # -- electrical model
    electrode: Literal["interdigitated", "disc"] = Field(
        "interdigitated",
        description="interdigitated: two equal gold combs, as on an RTCA plate. disc: "
        "classic ECIS working disc against a large counter electrode",
    )
    field_model: Literal["analytic", "fem"] = Field(
        "analytic",
        description="analytic: lumped series model. fem: electro-quasistatic Poisson on "
        "the IDE unit cell (interdigitated only)",
    )
    conductivity: float = 1.4
    permittivity_rel: float = Field(78.0, description="medium relative permittivity")
    junction_resistance: float = 2.0
    membrane_capacitance: float = 1.0e-6
    electrode_area_cm2: float = Field(8.0e-3, description="disc electrode only")
    ide_finger_width: Length = Field(50e-6, description="interdigitated: metal width")
    ide_finger_spacing: Length = Field(50e-6, description="interdigitated: gap")
    ide_finger_length: Length = Field(3e-3, description="interdigitated: finger length")
    ide_n_fingers: int = Field(30, ge=2, description="interdigitated: fingers, both combs")
    fem_resolution: int = Field(24, ge=8, le=128, description="cells per half-finger")
    cell_radius: float = 8.0e-6
    gap_height: float = 100e-9

    # -- data source
    source_file: str | None = Field(
        None, description="path to a real RTCA CSV/XLSX export; simulates when None"
    )
    plate_layout: dict[str, float] | None = Field(
        None, description="measured plates: well -> concentration (0 = vehicle control). "
        "Without a layout a measured plate gets Cell Index but no IC50",
    )
    plate_layout_file: str | None = Field(
        None, description="CSV with columns `well` and `concentration`, as plate_layout",
    )
    seed: int | None = 7

    @model_validator(mode="after")
    def _check(self) -> RTCAParams:
        if self.spectrum_range[0] >= self.spectrum_range[1]:
            raise ValueError("spectrum_range must be increasing")
        if self.seeding_coverage >= self.max_coverage:
            raise ValueError("seeding_coverage must be below max_coverage")
        if self.field_model == "fem" and self.electrode != "interdigitated":
            raise ValueError(
                "field_model='fem' solves the interdigitated electrode; a disc against "
                "a distant counter electrode has the closed-form spreading resistance "
                "already (use field_model='analytic')"
            )
        return self

    @property
    def geometry(self) -> ElectrodeGeometry:
        return ElectrodeGeometry(
            area_cm2=self.electrode_area_cm2,
            cell_radius=self.cell_radius,
            gap_height=self.gap_height,
        )

    @property
    def ide(self) -> IDEGeometry:
        return IDEGeometry(
            finger_width=self.ide_finger_width,
            finger_spacing=self.ide_finger_spacing,
            finger_length=self.ide_finger_length,
            n_fingers=self.ide_n_fingers,
        )


def normalized_cell_index(
    time: np.ndarray, ci: np.ndarray, treatment_time: float | None
) -> np.ndarray | None:
    """Cell Index divided by its value at the last time point before treatment.

    The RTCA software's standard view: every well starts the treated phase at
    1, so differences in seeding density drop out and only the response to
    the compound remains. ``None`` without a treatment time, or when no time
    point precedes it; wells whose baseline is not positive come out NaN.
    """
    if treatment_time is None:
        return None
    before = np.flatnonzero(np.asarray(time) <= treatment_time)
    if before.size == 0:
        return None
    base = np.asarray(ci, dtype=float)[before[-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(base > 0, np.asarray(ci, dtype=float) / base, np.nan)


def logistic_coverage(
    t: np.ndarray, *, seeding: float, capacity: float, doubling_time: float, lag: float
) -> np.ndarray:
    """Logistic growth of electrode coverage.

    ``dN/dt = r N (1 - N/K)`` with ``r = ln(2) / doubling_time``; coverage is
    clamped to the seeding value during the attachment *lag*.
    """
    t = np.asarray(t, dtype=float)
    r = np.log(2.0) / doubling_time
    te = np.clip(t - lag, 0.0, None)
    ratio = seeding / capacity
    return capacity * ratio * np.exp(r * te) / (1.0 + ratio * (np.exp(r * te) - 1.0))


class ImpedanceRTCA(Instrument):
    """xCELLigence-style real-time impedance analyser."""

    name = "impedance_rtca"
    display_name = "Real-time cell impedance analyser (RTCA)"
    description = (
        "Electrode-impedance cell monitoring: Giaever-Keese shell model, Cell Index "
        "time courses, |Z|(f) spectra and IC50 dose-response."
    )
    ConfigModel = RTCAParams

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.params: RTCAParams | None = None
        self.rng: np.random.Generator | None = None
        self._layout: pd.DataFrame | None = None
        self._fem_model: Any = None
        self._potential_map: xr.Dataset | None = None
        self._readout_info: dict[str, Any] = {}

    # -- Instrument contract ---------------------------------------------
    def setup(self) -> None:
        """Validate parameters and lay out the plate (dose per well)."""
        if self._is_set_up:
            return
        self.params = RTCAParams.model_validate(self.config.params)
        self.rng = np.random.default_rng(self.params.seed)
        self._layout = self._plate_layout()
        self._is_set_up = True

    def _plate_layout(self) -> pd.DataFrame:
        """Assign a concentration (or vehicle control) to each used well."""
        assert self.params is not None
        p = self.params
        rows = 8 if p.n_wells == 96 else 16
        cols = 12 if p.n_wells == 96 else 24
        wells, doses, kinds = [], [], []
        conc = list(p.concentrations)
        # Column 1 is the vehicle control, remaining columns carry the dose series.
        for r in range(rows):
            for c in range(cols):
                idx = c - 1
                if c == 0:
                    dose, kind = 0.0, "control"
                elif idx < len(conc) and r < p.replicates:
                    dose, kind = conc[idx], "treated"
                else:
                    continue
                wells.append(f"{chr(ord('A') + r)}{c + 1}")
                doses.append(dose)
                kinds.append(kind)
        return pd.DataFrame({"well": wells, "concentration": doses, "kind": kinds})

    def run(self) -> InstrumentResult:
        """Simulate (or read) the plate, then compute Cell Index and IC50."""
        self._ensure_setup()
        assert self.params is not None and self._layout is not None
        p = self.params

        if p.source_file:
            ds = read_rtca_csv(p.source_file)
            time = ds["time"].values
            ci = ds["cell_index"].values
            wells = [str(w) for w in ds["well"].values]
            layout = self._measured_layout(wells)
            spectra = None
        else:
            time, ci, wells, layout, spectra = self._simulate_plate()

        endpoint = self._endpoint_response(time, ci, layout)
        fit = self._fit_ic50(endpoint)

        fields = xr.Dataset(
            {"cell_index": (("time", "well"), ci)},
            coords={"time": time, "well": wells},
        )
        fields["time"].attrs["units"] = "s"
        fields["cell_index"].attrs["units"] = "dimensionless"
        nci = normalized_cell_index(time, ci, p.treatment_time)
        if nci is not None:
            fields["normalized_cell_index"] = (("time", "well"), nci)
            fields["normalized_cell_index"].attrs.update(
                units="dimensionless",
                normalised_at_s=float(time[time <= p.treatment_time][-1]),  # type: ignore[operator]
            )
        if spectra is not None:
            fields = xr.merge([fields, spectra])

        metrics: dict[str, Any] = {
            "n_wells": len(wells),
            "duration_h": float(time[-1] / 3600.0),
            "max_cell_index": float(np.nanmax(ci)),
            "readout_frequency_Hz": p.frequency,
        }
        if not p.source_file:
            metrics.update(self._electrode_metrics())
            metrics.update(getattr(self, "_readout_info", {}))
        if fit is not None:
            metrics.update(
                ic50=float(fit.ic50),
                ic50_stderr=float(fit.ic50_stderr),
                fit_bottom=float(fit.bottom),
                fit_top=float(fit.top),
                hill_slope=float(fit.hill),
                fit_r_squared=float(fit.r_squared),
                ic50_recovery_ratio=float(fit.ic50 / p.true_ic50) if not p.source_file
                else float("nan"),
            )

        self._result = InstrumentResult(
            fields=fields,
            metrics=metrics,
            table=endpoint,
            meta={
                "instrument": self.name,
                "config_hash": self.config.hash,
                "source": p.source_file or "simulated",
                "params": p.model_dump(mode="json"),
                "model": "Giaever-Keese cell layer (doi:10.1073/pnas.88.17.7896); "
                + ("interdigitated electrode, Olthuis cell constant "
                   "(doi:10.1016/0925-4005(95)85053-8)"
                   if p.electrode == "interdigitated" else "disc electrode (ECIS)")
                + ("; electro-quasistatic FEM" if p.field_model == "fem" else ""),
            },
        )
        return self._result

    # -- electrical model ---------------------------------------------------
    def _fem(self):
        """The IDE field model, built once per instrument."""
        if getattr(self, "_fem_model", None) is None:
            from biosim_lab.instruments.impedance_rtca.fem_model import IDEFieldModel

            p = self.params
            assert p is not None
            self._fem_model = IDEFieldModel(
                p.ide, conductivity=p.conductivity, permittivity_rel=p.permittivity_rel,
                rb=p.junction_resistance, specific_capacitance=p.membrane_capacitance,
                cell_radius=p.cell_radius, gap_height=p.gap_height,
                resolution=p.fem_resolution,
            )
        return self._fem_model

    def _lumped(self, frequency: np.ndarray, coverage: np.ndarray) -> np.ndarray:
        """Lumped impedance [Ohm] for every (frequency, coverage) pair, broadcast."""
        assert self.params is not None
        p = self.params
        f = np.asarray(frequency, dtype=float)
        z_naked = electrode_specific_impedance(f)
        alpha = alpha_parameter(p.cell_radius, p.gap_height, 1.0 / p.conductivity)
        z_covered = giaever_keese_impedance(
            f, z_naked_specific=z_naked, rb=p.junction_resistance, alpha=alpha,
            specific_capacitance=p.membrane_capacitance, coverage=1.0,
        )
        z_spec = _parallel_coverage(z_covered, z_naked, np.asarray(coverage, dtype=float))
        if p.electrode == "interdigitated":
            return 2.0 * z_spec / p.ide.active_comb_area_cm2 + ide_bulk_impedance(
                f, p.ide, conductivity=p.conductivity, permittivity_rel=p.permittivity_rel
            )
        return z_spec / p.electrode_area_cm2 + solution_resistance(
            p.electrode_area_cm2, conductivity=p.conductivity
        )

    def _readout(self) -> tuple[Any, dict[str, Any]]:
        """``|Z|(coverage)`` at the readout frequency, and what it rests on.

        Lumped: exact and vectorised. FEM: 41 solves across coverage, then
        interpolation --- |Z| is smooth and monotonic in coverage, so this is
        far below the measurement noise.
        """
        assert self.params is not None
        p = self.params
        f = np.array([p.frequency])
        info: dict[str, Any] = {}
        if p.field_model == "fem":
            fem = self._fem()
            grid = np.linspace(0.0, 1.0, 41)
            table = np.abs(fem.coverage_table(p.frequency, grid))
            lumped = np.abs(self._lumped(f, grid))
            info.update(
                readout_lumped_vs_fem_max_percent=float(
                    100.0 * np.max(np.abs(lumped - table) / table)
                ),
                readout_wagner_number=fem.wagner(p.frequency, 0.0),
                fem_mesh_nodes=int(fem.mesh.mesh.p.shape[1]),
            )
            return (lambda c: np.interp(c, grid, table)), info
        return (lambda c: np.abs(self._lumped(f, c))), info

    def _simulate_plate(self):
        """Synthesise a whole plate of Cell Index traces plus |Z|(f) spectra."""
        assert self.params is not None and self.rng is not None and self._layout is not None
        p = self.params
        layout = self._layout
        time = np.linspace(0.0, p.duration, p.n_timepoints)
        z_of_coverage, info = self._readout()
        self._readout_info = info
        z_blank = float(z_of_coverage(np.array([0.0]))[0])

        base = logistic_coverage(
            time, seeding=p.seeding_coverage, capacity=p.max_coverage,
            doubling_time=p.doubling_time, lag=p.lag_time,
        )
        coverage = np.repeat(base[:, None], len(layout), axis=1)
        if p.treatment_time is not None:
            after = time >= p.treatment_time
            # After dosing the population relaxes towards the reduced capacity.
            relax = 1.0 - np.exp(-(time[after] - p.treatment_time) / max(p.doubling_time, 1.0))
            for j, (_, row) in enumerate(layout.iterrows()):
                if row["kind"] != "treated":
                    continue
                target = self._capacity_for(row["concentration"]) * p.max_coverage
                coverage[after, j] = coverage[after, j] + (target - coverage[after, j]) * relax

        z = z_of_coverage(coverage.ravel()).reshape(coverage.shape)
        noise = 1.0 + self.rng.normal(0.0, p.noise_cv, size=z.shape)
        ci = cell_index(z * noise, z_blank, reference_impedance=p.reference_impedance)

        freqs = np.logspace(
            np.log10(p.spectrum_range[0]), np.log10(p.spectrum_range[1]),
            p.spectrum_frequencies,
        )
        levels = [0.0, 0.5, p.max_coverage]
        lumped = np.column_stack([self._lumped(freqs, c) for c in levels])
        if p.field_model == "fem":
            fem = self._fem()
            chosen = np.column_stack([fem.spectrum(freqs, c) for c in levels])
            self._potential_map = fem.potential_map()
        else:
            chosen = lumped
        dims = ("frequency", "coverage")
        spectra = xr.Dataset(
            {"impedance_real": (dims, chosen.real), "impedance_imag": (dims, chosen.imag)},
            coords={"frequency": freqs, "coverage": levels},
        )
        if p.field_model == "fem":
            spectra["impedance_lumped_real"] = (dims, lumped.real)
            spectra["impedance_lumped_imag"] = (dims, lumped.imag)
            deviation = 100.0 * np.abs(np.abs(lumped) - np.abs(chosen)) / np.abs(chosen)
            self._readout_info["spectrum_lumped_vs_fem_max_percent"] = float(deviation.max())
            self._readout_info["spectrum_worst_frequency_Hz"] = float(
                freqs[np.unravel_index(deviation.argmax(), deviation.shape)[0]]
            )
        spectra["frequency"].attrs["units"] = "Hz"
        for name in spectra.data_vars:
            spectra[name].attrs["units"] = "ohm"

        return time, ci, list(layout["well"]), layout, spectra

    def _measured_layout(self, wells: list[str]) -> pd.DataFrame:
        """Doses for a measured plate, from ``plate_layout`` / ``plate_layout_file``."""
        assert self.params is not None
        p = self.params
        doses: dict[str, float] = {}
        if p.plate_layout_file:
            table = pd.read_csv(p.plate_layout_file)
            cols = {c.lower().strip(): c for c in table.columns}
            if "well" not in cols or "concentration" not in cols:
                raise ValueError(
                    f"{p.plate_layout_file}: needs columns 'well' and 'concentration'"
                )
            doses.update(zip(table[cols["well"]].astype(str),
                             table[cols["concentration"]].astype(float)))
        if p.plate_layout:
            doses.update(p.plate_layout)
        doses = {normalise_well_label(k): float(v) for k, v in doses.items()}
        conc = [doses.get(w, np.nan) for w in wells]
        kind = ["measured" if not np.isfinite(c) else ("control" if c == 0 else "treated")
                for c in conc]
        return pd.DataFrame({"well": wells, "concentration": conc, "kind": kind})

    def _electrode_metrics(self) -> dict[str, Any]:
        """Electrode constants of the simulated plate."""
        assert self.params is not None
        p = self.params
        f = np.array([p.frequency])
        out: dict[str, Any] = {"electrode": p.electrode, "field_model": p.field_model}
        if p.electrode == "interdigitated":
            out.update(
                cell_constant_1_per_m=ide_cell_constant(p.ide),
                solution_resistance_ohm=float(np.abs(ide_bulk_impedance(
                    f, p.ide, conductivity=p.conductivity,
                    permittivity_rel=p.permittivity_rel))[0]),
                electrode_footprint_mm2=100.0 * p.ide.footprint_cm2,
                wagner_number_blank=float(wagner_number(
                    electrode_specific_impedance(f)[0], conductivity=p.conductivity,
                    length=0.5 * p.ide_finger_width)),
            )
        else:
            out["solution_resistance_ohm"] = solution_resistance(
                p.electrode_area_cm2, conductivity=p.conductivity)
        out["blank_impedance_ohm"] = float(np.abs(self._lumped(f, np.array([0.0])))[0])
        return out

    def _capacity_for(self, concentration: float) -> float:
        """Surviving fraction of the carrying capacity at a given dose (Hill model)."""
        assert self.params is not None
        p = self.params
        if not np.isfinite(concentration) or concentration <= 0:
            return 1.0
        return float(1.0 / (1.0 + (concentration / p.true_ic50) ** p.hill_slope))

    def _endpoint_response(
        self, time: np.ndarray, ci: np.ndarray, layout: pd.DataFrame
    ) -> pd.DataFrame:
        """Normalised endpoint Cell Index per well, the input to the IC50 fit."""
        endpoint_ci = ci[-1]
        df = layout.copy().reset_index(drop=True)
        df["endpoint_cell_index"] = endpoint_ci[: len(df)]
        controls = df.loc[df["kind"] == "control", "endpoint_cell_index"]
        baseline = float(controls.mean()) if len(controls) else float(np.nanmax(endpoint_ci))
        df["normalised_response"] = df["endpoint_cell_index"] / baseline if baseline else np.nan
        return df

    def _fit_ic50(self, endpoint: pd.DataFrame) -> DoseResponseFit | None:
        """Fit a 4PL curve to the treated wells; ``None`` when there are too few."""
        treated = endpoint[(endpoint["kind"] == "treated") & (endpoint["concentration"] > 0)]
        if len(treated) < 4:
            return None
        grouped = treated.groupby("concentration")["normalised_response"].mean()
        try:
            return fit_dose_response(grouped.index.to_numpy(), grouped.to_numpy())
        except (RuntimeError, ValueError):
            return None

    # -- dashboard --------------------------------------------------------
    def dashboard(self) -> Any:
        """Cell Index time courses, Nyquist/Bode spectra, plate map and IC50."""
        from biosim_lab.instruments.impedance_rtca.dashboard import build_dashboard

        return build_dashboard(self)

    @classmethod
    def example_config(cls) -> dict[str, Any]:
        return {
            "name": "rtca_ic50",
            "instrument": cls.name,
            "seed": 7,
            "description": "48 h impedance growth curve with a drug dose-response at 24 h.",
            "params": {
                "frequency": "10 kHz",
                "duration": "48 h",
                "treatment_time": "24 h",
                "n_wells": 96,
                "true_ic50": 1.0,
                "concentrations": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "replicates": 3,
            },
        }


__all__ = ["ImpedanceRTCA", "RTCAParams", "logistic_coverage", "normalized_cell_index"]
