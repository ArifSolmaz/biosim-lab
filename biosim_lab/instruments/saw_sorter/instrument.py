"""The ``saw_sorter`` instrument plugin.

Simulates a standing-surface-acoustic-wave (SSAW) microfluidic cell sorter of
the kind used to enrich circulating tumour cells from blood: cells flow through
a rectangular channel above a piezoelectric substrate carrying a standing SAW,
and the acoustic radiation force pushes them towards the pressure node at a
rate that scales with ``radius^3 * contrast_factor`` — so large, stiff tumour
cells migrate far faster than erythrocytes and can be creamed off a central
outlet.

Commercial equivalents: the acoustic cell separators sold for CTC enrichment
and for washing cell-therapy products.  Literature reference for the device
concept: Li et al. (2015), PNAS 112:4970, doi:10.1073/pnas.1504484112.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import Instrument, InstrumentResult
from biosim_lab.instruments.saw_sorter.simulate import (
    SAWSorterParams,
    SAWSorterSimulation,
    SortingOutcome,
    parameter_sweep,
    replicate_sorting,
)


class SAWSorter(Instrument):
    """Standing-SAW acoustophoretic cell sorter."""

    name = "saw_sorter"
    display_name = "SAW acoustophoretic cell sorter"
    description = (
        "Standing surface-acoustic-wave separation of cancer cells from blood cells "
        "in a microfluidic channel (Gor'kov radiation force + Poiseuille flow)."
    )
    ConfigModel = SAWSorterParams

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.params: SAWSorterParams | None = None
        self.simulation: SAWSorterSimulation | None = None
        self.outcome: SortingOutcome | None = None

    # -- Instrument contract ---------------------------------------------
    def setup(self) -> None:
        """Validate the parameter block and build the simulation object."""
        if self._is_set_up:
            return
        self.params = SAWSorterParams.model_validate(self.config.params)
        self.simulation = SAWSorterSimulation(self.params)
        self._is_set_up = True

    def run(self) -> InstrumentResult:
        """Run the sorting simulation and package the result."""
        self._ensure_setup()
        assert self.simulation is not None and self.params is not None

        outcome = self.simulation.run()
        self.outcome = outcome

        fields = outcome.tracks.trajectories
        histogram = self.simulation.outlet_histogram(outcome.cells)
        fields = xr.merge([fields, histogram.rename({"counts": "outlet_histogram"})])
        if outcome.field is not None:
            fields = xr.merge([fields, outcome.field], combine_attrs="drop_conflicts")

        self._result = InstrumentResult(
            fields=fields,
            metrics=outcome.metrics,
            table=outcome.cells,
            meta={
                "instrument": self.name,
                "config_hash": self.config.hash,
                "experiment": self.config.name,
                "diagnostics": outcome.diagnostics,
                "params": self.params.model_dump(mode="json"),
            },
        )
        return self._result

    # -- extras -----------------------------------------------------------
    def sweep(
        self, grid: dict[str, Any], *, progress: bool = False
    ) -> tuple[Any, xr.Dataset]:
        """Run a parameter sweep around the configured operating point."""
        self._ensure_setup()
        assert self.params is not None
        return parameter_sweep(self.params, grid, progress=progress)

    def replicate(self, *, n_replicates: int = 5, confidence: float = 0.95) -> Any:
        """Re-run the same device on fresh random samples and summarise the spread.

        Complements the counting error already in ``metrics`` (``*_ci_low`` /
        ``*_ci_high``, which is binomial and present even with the seed fixed).
        This measures the other uncertainty: how much the answer moves when the
        cells themselves are redrawn.
        """
        self._ensure_setup()
        assert self.params is not None
        return replicate_sorting(
            self.params, n_replicates=n_replicates, confidence=confidence
        )

    def dashboard(self) -> Any:
        """Interactive Panel dashboard with live parameter sliders."""
        from biosim_lab.instruments.saw_sorter.dashboard import build_dashboard

        self._ensure_setup()
        assert self.params is not None
        return build_dashboard(self.params)

    @classmethod
    def example_config(cls) -> dict[str, Any]:
        """A working single-node MCF-7 vs RBC separation."""
        return {
            "name": "ctc_vs_rbc",
            "instrument": cls.name,
            "seed": 12345,
            "description": (
                "MCF-7 breast cancer cells separated from erythrocytes by a single-node "
                "standing SAW at 6.63 MHz."
            ),
            "params": {
                "frequency": "6.632 MHz",
                "voltage_pp": "15 V",
                "channel_width": "300 um",
                "channel_height": "50 um",
                "channel_length": "2 mm",
                "flow_rate": "5 uL/min",
                "fluid": "water",
                "substrate": "linbo3_128yx",
                "inlet": "sheath_sides",
                "collection_fraction": 0.3333333333333333,
                "mode": "analytic",
                "populations": [
                    {"cell_type": "mcf7", "count": 300, "target": True},
                    {"cell_type": "rbc", "count": 300, "target": False},
                ],
            },
        }


__all__ = ["SAWSorter"]
