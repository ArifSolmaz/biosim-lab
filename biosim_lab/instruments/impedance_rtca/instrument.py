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

What is deliberately not modelled yet (Stage 2 scope): electrode geometry
effects beyond the lumped ``alpha`` parameter, cell micromotion noise, and
temperature drift.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator

from biosim_lab.core.config import BaseConfigModel, ExperimentConfig, Frequency, Time
from biosim_lab.core.io import read_rtca_csv
from biosim_lab.core.plugin import Instrument, InstrumentResult
from biosim_lab.instruments.impedance_rtca.physics import (
    RTCA_REFERENCE_IMPEDANCE_OHM,
    DoseResponseFit,
    ElectrodeGeometry,
    cell_index,
    fit_dose_response,
    well_impedance,
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
    conductivity: float = 1.4
    junction_resistance: float = 2.0
    membrane_capacitance: float = 1.0e-6
    electrode_area_cm2: float = 8.0e-3
    cell_radius: float = 8.0e-6
    gap_height: float = 100e-9

    # -- data source
    source_file: str | None = Field(
        None, description="path to a real RTCA CSV/XLSX export; simulates when None"
    )
    seed: int | None = 7

    @model_validator(mode="after")
    def _check(self) -> RTCAParams:
        if self.spectrum_range[0] >= self.spectrum_range[1]:
            raise ValueError("spectrum_range must be increasing")
        if self.seeding_coverage >= self.max_coverage:
            raise ValueError("seeding_coverage must be below max_coverage")
        return self

    @property
    def geometry(self) -> ElectrodeGeometry:
        return ElectrodeGeometry(
            area_cm2=self.electrode_area_cm2,
            cell_radius=self.cell_radius,
            gap_height=self.gap_height,
        )


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
            layout = pd.DataFrame({"well": wells,
                                   "concentration": np.nan, "kind": "measured"})
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
        if spectra is not None:
            fields = xr.merge([fields, spectra])

        metrics: dict[str, Any] = {
            "n_wells": len(wells),
            "duration_h": float(time[-1] / 3600.0),
            "max_cell_index": float(np.nanmax(ci)),
            "readout_frequency_Hz": p.frequency,
        }
        if fit is not None:
            metrics.update(
                ic50=float(fit.ic50),
                ic50_stderr=float(fit.ic50_stderr),
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
                "model": "Giaever-Keese, doi:10.1073/pnas.88.17.7896",
            },
        )
        return self._result

    def _simulate_plate(self):
        """Synthesise a whole plate of Cell Index traces plus one blank spectrum."""
        assert self.params is not None and self.rng is not None and self._layout is not None
        p = self.params
        layout = self._layout
        time = np.linspace(0.0, p.duration, p.n_timepoints)

        z_blank = np.abs(
            well_impedance(
                np.array([p.frequency]),
                coverage=0.0,
                geometry=p.geometry,
                conductivity=p.conductivity,
                rb=p.junction_resistance,
                specific_capacitance=p.membrane_capacitance,
            )
        )[0]

        ci = np.zeros((len(time), len(layout)))
        for j, (_, row) in enumerate(layout.iterrows()):
            capacity = self._capacity_for(row["concentration"])
            coverage = logistic_coverage(
                time,
                seeding=p.seeding_coverage,
                capacity=p.max_coverage,
                doubling_time=p.doubling_time,
                lag=p.lag_time,
            )
            if p.treatment_time is not None and row["kind"] == "treated":
                after = time >= p.treatment_time
                # After dosing the population relaxes towards the reduced capacity.
                relax = 1.0 - np.exp(-(time[after] - p.treatment_time)
                                     / max(p.doubling_time, 1.0))
                target = capacity * p.max_coverage
                coverage[after] = coverage[after] + (target - coverage[after]) * relax
            z = np.abs(
                well_impedance(
                    np.full(time.shape, p.frequency),
                    coverage=1.0,
                    geometry=p.geometry,
                    conductivity=p.conductivity,
                    rb=p.junction_resistance,
                    specific_capacitance=p.membrane_capacitance,
                )
            )
            # Coverage enters through the parallel-patch rule, evaluated per timepoint.
            z = np.array(
                [
                    np.abs(
                        well_impedance(
                            np.array([p.frequency]),
                            coverage=float(c),
                            geometry=p.geometry,
                            conductivity=p.conductivity,
                            rb=p.junction_resistance,
                            specific_capacitance=p.membrane_capacitance,
                        )
                    )[0]
                    for c in coverage
                ]
            )
            noise = 1.0 + self.rng.normal(0.0, p.noise_cv, size=z.shape)
            ci[:, j] = cell_index(
                z * noise, z_blank, reference_impedance=p.reference_impedance
            )

        freqs = np.logspace(
            np.log10(p.spectrum_range[0]), np.log10(p.spectrum_range[1]),
            p.spectrum_frequencies,
        )
        spectra = xr.Dataset(
            {
                "impedance_real": (
                    ("frequency", "coverage"),
                    np.column_stack([
                        well_impedance(freqs, coverage=c, geometry=p.geometry,
                                       conductivity=p.conductivity,
                                       rb=p.junction_resistance,
                                       specific_capacitance=p.membrane_capacitance).real
                        for c in (0.0, 0.5, p.max_coverage)
                    ]),
                ),
                "impedance_imag": (
                    ("frequency", "coverage"),
                    np.column_stack([
                        well_impedance(freqs, coverage=c, geometry=p.geometry,
                                       conductivity=p.conductivity,
                                       rb=p.junction_resistance,
                                       specific_capacitance=p.membrane_capacitance).imag
                        for c in (0.0, 0.5, p.max_coverage)
                    ]),
                ),
            },
            coords={"frequency": freqs, "coverage": [0.0, 0.5, p.max_coverage]},
        )
        spectra["frequency"].attrs["units"] = "Hz"
        spectra["impedance_real"].attrs["units"] = "ohm"
        spectra["impedance_imag"].attrs["units"] = "ohm"

        return time, ci, list(layout["well"]), layout, spectra

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


__all__ = ["ImpedanceRTCA", "RTCAParams", "logistic_coverage"]
