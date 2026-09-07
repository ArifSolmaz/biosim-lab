"""``cell_counter`` — automated cell counter (Countess / Cellometer analogue).

Stage 3 status: **skeleton with a working synthetic demo**.  It segments a
field of view, gates it by size, classifies trypan-blue viability, and reports
concentration with its Poisson counting uncertainty.  With
``source_image`` pointing at a real TIFF the same pipeline runs on measured
data.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from pydantic import Field

from biosim_lab.core.config import BaseConfigModel, ExperimentConfig, Length
from biosim_lab.core.io import read_image_stack
from biosim_lab.core.plugin import Instrument, InstrumentResult
from biosim_lab.instruments.cell_counter.counting import (
    NEUBAUER_DEPTH_M,
    count_field,
    counting_uncertainty,
)
from biosim_lab.instruments.cell_counter.segmentation import segment
from biosim_lab.instruments.cell_counter.synthetic import SyntheticImageSpec, synthetic_field


class CellCounterParams(BaseConfigModel):
    """Configuration block for the ``cell_counter`` instrument."""

    source_image: str | None = Field(
        None, description="path to a TIFF/PNG field of view; synthesises when None"
    )
    pixel_size: Length = 0.65e-6
    chamber_depth: Length = NEUBAUER_DEPTH_M
    dilution_factor: float = Field(2.0, gt=0, description="2.0 for a 1:1 trypan-blue mix")

    backend: Literal["classical", "cellpose", "stardist"] = "classical"
    min_radius_px: float = 4.0
    min_diameter_um: float = 5.0
    max_diameter_um: float = 40.0
    viability_threshold: float | None = None

    # synthetic-demo controls
    n_cells: int = 150
    dead_fraction: float = 0.15
    image_size: int = 512
    seed: int | None = 0


class CellCounter(Instrument):
    """Automated brightfield cell counter with trypan-blue viability."""

    name = "cell_counter"
    display_name = "Automated cell counter"
    description = (
        "Segmentation-based counting: concentration in cells/mL from the chamber "
        "geometry, trypan-blue viability and the size distribution."
    )
    ConfigModel = CellCounterParams

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.params: CellCounterParams | None = None
        self.image: np.ndarray | None = None
        self.truth: np.ndarray | None = None
        self.segmentation: Any = None

    def setup(self) -> None:
        """Load or synthesise the field of view."""
        if self._is_set_up:
            return
        self.params = CellCounterParams.model_validate(self.config.params)
        p = self.params
        if p.source_image:
            stack = read_image_stack(p.source_image)
            image = np.asarray(stack)
            if image.ndim == 3:
                image = image[0] if image.shape[0] < image.shape[-1] else image.mean(axis=-1)
            self.image = image.astype(float)
            self.image = (self.image - self.image.min()) / max(
                float(np.ptp(self.image)), 1e-12
            )
            self.truth = None
        else:
            field = synthetic_field(
                SyntheticImageSpec(
                    shape=(p.image_size, p.image_size),
                    pixel_size=p.pixel_size,
                    n_cells=p.n_cells,
                    dead_fraction=p.dead_fraction,
                    seed=p.seed,
                )
            )
            self.image = field["image"]
            self.truth = field["truth"]
        self._is_set_up = True

    def run(self) -> InstrumentResult:
        """Segment, gate, count and report."""
        self._ensure_setup()
        assert self.params is not None and self.image is not None
        p = self.params

        kwargs: dict[str, Any] = {}
        if p.backend == "classical":
            kwargs["min_radius_px"] = p.min_radius_px
        self.segmentation = segment(self.image, backend=p.backend, **kwargs)

        counts = count_field(
            self.segmentation,
            shape_px=self.image.shape,  # type: ignore[arg-type]
            pixel_size=p.pixel_size,
            depth=p.chamber_depth,
            dilution_factor=p.dilution_factor,
            viability_threshold=p.viability_threshold,
            min_diameter_um=p.min_diameter_um,
            max_diameter_um=p.max_diameter_um,
        )

        metrics = counts.as_metrics()
        metrics["counting_relative_uncertainty"] = counting_uncertainty(counts.n_total)
        metrics["segmentation_backend"] = self.segmentation.backend
        if self.truth is not None:
            n_truth = int(len(self.truth))
            metrics["ground_truth_n"] = n_truth
            metrics["detection_recall"] = float(min(counts.n_total, n_truth) / n_truth)
            metrics["ground_truth_viability_percent"] = float(
                100.0 * self.truth["alive"].mean()
            )

        hist, edges = np.histogram(counts.diameters_um, bins=30)
        fields = xr.Dataset(
            {
                "image": (("row", "col"), np.asarray(self.image)),
                "labels": (("row", "col"), self.segmentation.labels.astype(np.int32)),
                "diameter_histogram": ("diameter_bin", hist),
            },
            coords={
                "row": np.arange(self.image.shape[0]),
                "col": np.arange(self.image.shape[1]),
                "diameter_bin": 0.5 * (edges[:-1] + edges[1:]),
            },
        )
        fields["diameter_bin"].attrs["units"] = "um"

        self._result = InstrumentResult(
            fields=fields,
            metrics=metrics,
            table=counts.table,
            meta={
                "instrument": self.name,
                "config_hash": self.config.hash,
                "source": p.source_image or "synthetic",
                "params": p.model_dump(mode="json"),
                "geometry_reference": "Neubauer improved chamber, depth 0.1 mm",
            },
        )
        return self._result

    def view_napari(self, show: bool = True) -> Any:
        """Open the field of view with its segmentation in Napari (optional)."""
        from biosim_lab.core.viz.napari_layers import view_segmentation

        self._ensure_setup()
        result = self.results()
        points = None
        if result.table is not None and "centroid-0" in result.table:
            points = result.table.rename(
                columns={"centroid-0": "y", "centroid-1": "x"}
            )[["y", "x"]]
        return view_segmentation(
            self.image, self.segmentation.labels, points=points, show=show
        )

    def dashboard(self) -> Any:
        """Panel dashboard: image, overlay, size distribution, object table."""
        from biosim_lab.instruments.cell_counter.dashboard import build_dashboard

        return build_dashboard(self)

    @classmethod
    def example_config(cls) -> dict[str, Any]:
        return {
            "name": "cell_count_demo",
            "instrument": cls.name,
            "seed": 0,
            "description": "Synthetic brightfield field of view with trypan-blue viability.",
            "params": {
                "pixel_size": "0.65 um",
                "chamber_depth": "100 um",
                "dilution_factor": 2.0,
                "n_cells": 150,
                "dead_fraction": 0.15,
                "backend": "classical",
            },
        }


__all__ = ["CellCounter", "CellCounterParams"]
