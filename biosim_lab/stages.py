"""Concrete pipeline stages wrapping the instruments.

Each class here is a thin adapter: it takes a :class:`~biosim_lab.pipeline.Sample`,
runs one instrument, and returns the sample as that instrument leaves it. The
physics lives in the instruments; what these add is the *bookkeeping* --- what
went in, what came out, and what each step cost in certainty.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.statistics import wilson_interval
from biosim_lab.pipeline import Sample, Stage, StageReport


class SortStage(Stage):
    """Acoustic sorting: keeps the cells that reach the collection outlet.

    The uncertainty contributed here is *binomial*. A finite number of cells
    either cross into the collection outlet or do not, so the recovered fraction
    carries counting error even from a perfectly deterministic device. A Wilson
    interval is used rather than the normal approximation, whose width collapses
    to exactly zero at 100 % recovery --- the case a working sorter hits
    constantly.
    """

    name = "sort"

    def __init__(self, params: Any) -> None:
        self.params = params
        self.outcome: Any = None

    def apply(self, sample: Sample) -> Sample:
        from biosim_lab.instruments.saw_sorter.simulate import SAWSorterSimulation

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            self.outcome = SAWSorterSimulation(self.params).run()

        cells = self.outcome.cells
        collected = cells[cells["outlet"] == "collect"].reset_index(drop=True)

        n_in, n_out = int(len(cells)), int(len(collected))
        interval = wilson_interval(n_out, n_in)
        # Half the interval width, as a fraction of the recovered proportion.
        relative = (
            interval.half_width_percent / interval.point_percent
            if interval.point_percent else float("nan")
        )

        report = StageReport(
            stage=self.name, n_in=n_in, n_out=n_out,
            relative_uncertainty=float(relative),
            metrics={
                "efficiency_percent": self.outcome.metrics.get("efficiency_percent"),
                "purity_percent": self.outcome.metrics.get("purity_percent"),
                "recovery_ci_low_percent": interval.low_percent,
                "recovery_ci_high_percent": interval.high_percent,
            },
        )
        return Sample(
            cells=collected,
            volume_ml=sample.volume_ml,
            history=[*sample.history, report],
        )


class CountStage(Stage):
    """Image the suspension and count it.

    The size distribution handed to the imaging model is the one the *incoming
    sample actually has*, not a default. That is the point of chaining: after a
    size-selective sort the population is enriched in large cells, and a counter
    calibrated on the loaded distribution would be gated wrongly.

    The uncertainty contributed is *Poisson*: a field of view holds a finite,
    randomly-placed number of cells, giving ``1/sqrt(N)``.
    """

    name = "count"

    def __init__(
        self,
        *,
        pixel_size: float = 0.65e-6,
        image_size: int = 512,
        n_cells: int | None = None,
        seed: int | None = 0,
        **counter_params: Any,
    ) -> None:
        self.pixel_size = float(pixel_size)
        self.image_size = int(image_size)
        self.n_cells = n_cells
        self.seed = seed
        self.counter_params = counter_params
        self.instrument: Any = None

    def apply(self, sample: Sample) -> Sample:
        from biosim_lab.instruments.cell_counter.instrument import CellCounter

        stats = sample.radius_stats()
        if not np.isfinite(stats["mean_m"]):
            raise ValueError("cannot image a sample with no usable cell radii")

        # Carry the sample's own size distribution into the field of view.
        radius_mean_px = stats["mean_m"] / self.pixel_size
        radius_cv = stats["cv"] if np.isfinite(stats["cv"]) else 0.0

        dead_fraction = 0.0
        if "alive" in sample.cells:
            dead_fraction = float(1.0 - sample.cells["alive"].mean())

        params = {
            "pixel_size": self.pixel_size,
            "image_size": self.image_size,
            "n_cells": int(self.n_cells if self.n_cells is not None else sample.n),
            "radius_mean_px": float(radius_mean_px),
            "radius_cv": float(radius_cv),
            "dead_fraction": dead_fraction,
            "seed": self.seed,
            **self.counter_params,
        }
        config = ExperimentConfig.model_validate(
            {"instrument": "cell_counter", "params": params}
        )
        self.instrument = CellCounter(config)
        result = self.instrument.run()

        n_counted = int(result.metrics.get("n_total", 0))
        relative = 1.0 / np.sqrt(n_counted) if n_counted > 0 else float("nan")

        report = StageReport(
            stage=self.name, n_in=sample.n, n_out=n_counted,
            relative_uncertainty=float(relative),
            metrics={
                "concentration_per_ml": result.metrics.get("concentration_per_ml"),
                "viability_percent": result.metrics.get("viability_percent"),
                "mean_diameter_um": result.metrics.get("mean_diameter_um"),
                "detection_recall": result.metrics.get("detection_recall"),
                "imaged_radius_mean_px": radius_mean_px,
                "imaged_radius_cv": radius_cv,
            },
        )
        # Counting measures the sample; it does not consume it.
        return Sample(
            cells=sample.cells,
            volume_ml=sample.volume_ml,
            history=[*sample.history, report],
        )


class TrackStage(Stage):
    """Seed the surviving cells and track their motion.

    Contributes *sample-to-sample* uncertainty: the standard error of the mean
    over the tracks that were long enough to measure. This is a different thing
    from the counting error above --- it describes how much cells differ from
    each other, not how many of them were seen.
    """

    name = "track"

    def __init__(self, *, seed: int | None = 0, **tracker_params: Any) -> None:
        self.seed = seed
        self.tracker_params = tracker_params
        self.instrument: Any = None

    def apply(self, sample: Sample) -> Sample:
        from biosim_lab.instruments.cell_tracker.instrument import CellTracker

        params = {"seed": self.seed, **self.tracker_params}
        params.setdefault("n_cells", min(sample.n, 60))
        config = ExperimentConfig.model_validate(
            {"instrument": "cell_tracker", "params": params}
        )
        self.instrument = CellTracker(config)
        result = self.instrument.run()

        n_tracks = int(result.metrics.get("n_tracks", 0))
        table = result.table
        relative = float("nan")
        if isinstance(table, pd.DataFrame) and "mean_speed_m_s" in table and n_tracks > 1:
            speeds = np.asarray(table["mean_speed_m_s"], dtype=float)
            speeds = speeds[np.isfinite(speeds)]
            if speeds.size > 1 and speeds.mean() != 0:
                sem = float(speeds.std(ddof=1) / np.sqrt(speeds.size))
                relative = sem / float(speeds.mean())

        report = StageReport(
            stage=self.name, n_in=sample.n, n_out=n_tracks,
            relative_uncertainty=relative,
            metrics={
                "mean_speed_um_per_min": result.metrics.get("mean_speed_um_per_min"),
                "msd_alpha": result.metrics.get("msd_alpha"),
                "n_tracks": n_tracks,
            },
        )
        return Sample(
            cells=sample.cells,
            volume_ml=sample.volume_ml,
            history=[*sample.history, report],
        )


__all__ = ["SortStage", "CountStage", "TrackStage"]
