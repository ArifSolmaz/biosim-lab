"""``cell_tracker`` — live-cell tracking (Incucyte / CellTracker analogue).

Stage 3 status: **skeleton with a working synthetic demo**.  A synthetic
time-lapse with known ground-truth tracks is segmented frame by frame, linked
with ``trackpy``, and reduced to speed / persistence / MSD metrics.  Pointing
``source_movie`` at a real TIFF stack runs the identical pipeline on measured
data; ``btrack`` lineage reconstruction is defined but not implemented.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from biosim_lab.core.config import BaseConfigModel, ExperimentConfig, Length, Time
from biosim_lab.core.io import read_image_stack
from biosim_lab.core.plugin import Instrument, InstrumentResult
from biosim_lab.instruments.cell_counter.segmentation import segment
from biosim_lab.instruments.cell_counter.synthetic import SyntheticImageSpec, synthetic_movie
from biosim_lab.instruments.cell_tracker.tracking import (
    detections_from_labels,
    link_detections,
    track_metrics,
)


class CellTrackerParams(BaseConfigModel):
    """Configuration block for the ``cell_tracker`` instrument."""

    source_movie: str | None = Field(
        None, description="path to a TIFF stack or image directory; synthesises when None"
    )
    pixel_size: Length = 0.65e-6
    frame_interval: Time = 600.0  # 10 min

    backend: Literal["classical", "cellpose", "stardist"] = "classical"
    min_radius_px: float = 4.0
    search_range_px: float = Field(
        12.0, gt=0, description="max displacement between frames; the key linking parameter"
    )
    memory_frames: int = 2
    min_track_length: int = 5

    # synthetic-demo controls
    n_frames: int = 24
    n_cells: int = 60
    image_size: int = 320
    speed_px_per_frame: float = 2.2
    persistence: float = 0.85
    seed: int | None = 2


class CellTracker(Instrument):
    """Live-cell tracker with motility metrics."""

    name = "cell_tracker"
    display_name = "Live-cell tracker"
    description = (
        "Frame-by-frame segmentation, trackpy linking, and speed / persistence / MSD "
        "motility metrics for time-lapse microscopy."
    )
    ConfigModel = CellTrackerParams

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self.params: CellTrackerParams | None = None
        self.movie: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.tracks: pd.DataFrame | None = None
        self.truth: np.ndarray | None = None

    def setup(self) -> None:
        """Load or synthesise the time-lapse."""
        if self._is_set_up:
            return
        self.params = CellTrackerParams.model_validate(self.config.params)
        p = self.params
        if p.source_movie:
            stack = np.asarray(read_image_stack(p.source_movie), dtype=float)
            if stack.ndim == 4:  # (t, y, x, c) -> grayscale
                stack = stack.mean(axis=-1)
            lo, span = stack.min(), max(float(np.ptp(stack)), 1e-12)
            self.movie = ((stack - lo) / span).astype(np.float32)
            self.truth = None
        else:
            data = synthetic_movie(
                SyntheticImageSpec(
                    shape=(p.image_size, p.image_size),
                    pixel_size=p.pixel_size,
                    n_cells=p.n_cells,
                    dead_fraction=0.0,
                    seed=p.seed,
                ),
                n_frames=p.n_frames,
                speed_px_per_frame=p.speed_px_per_frame,
                persistence=p.persistence,
            )
            self.movie = data["movie"]
            self.truth = data["truth"]
        self._is_set_up = True

    def run(self) -> InstrumentResult:
        """Segment every frame, link, and compute motility metrics."""
        self._ensure_setup()
        assert self.params is not None and self.movie is not None
        p = self.params

        kwargs: dict[str, Any] = {}
        if p.backend == "classical":
            kwargs["min_radius_px"] = p.min_radius_px
        label_stack = np.stack(
            [segment(frame, backend=p.backend, clear_border=False, **kwargs).labels
             if p.backend == "classical"
             else segment(frame, backend=p.backend, **kwargs).labels
             for frame in self.movie]
        )
        self.labels = label_stack

        detections = detections_from_labels(label_stack, self.movie)
        tracks = link_detections(
            detections,
            search_range=p.search_range_px,
            memory=p.memory_frames,
            min_track_length=p.min_track_length,
        )
        self.tracks = tracks

        metrics_obj = track_metrics(
            tracks, pixel_size=p.pixel_size, frame_interval=p.frame_interval
        )
        per_track = metrics_obj.per_track

        metrics: dict[str, Any] = {
            "n_frames": int(self.movie.shape[0]),
            "n_detections": int(len(detections)),
            "n_tracks": int(per_track["particle"].nunique()) if len(per_track) else 0,
            "mean_track_length_frames": float(per_track["n_frames"].mean())
            if len(per_track) else float("nan"),
            "mean_speed_um_per_min": float(per_track["mean_speed_m_s"].mean() * 6e7)
            if len(per_track) else float("nan"),
            "median_persistence": float(per_track["persistence"].median())
            if len(per_track) else float("nan"),
            "msd_alpha": metrics_obj.alpha,
            "diffusion_coefficient_m2_s": metrics_obj.diffusion_coefficient,
            "segmentation_backend": p.backend,
        }
        if self.truth is not None:
            n_true = int(np.unique(self.truth["particle"]).size)
            metrics["ground_truth_n_tracks"] = n_true
            metrics["track_recovery_ratio"] = (
                metrics["n_tracks"] / n_true if n_true else float("nan")
            )
            true_speed = p.speed_px_per_frame * p.pixel_size / p.frame_interval
            metrics["ground_truth_speed_um_per_min"] = float(true_speed * 6e7)

        fields = xr.Dataset(
            {
                "movie": (("time", "row", "col"), np.asarray(self.movie)),
                "labels": (("time", "row", "col"), label_stack.astype(np.int32)),
            },
            coords={
                "time": np.arange(self.movie.shape[0]) * p.frame_interval,
                "row": np.arange(self.movie.shape[1]),
                "col": np.arange(self.movie.shape[2]),
            },
        )
        fields["time"].attrs["units"] = "s"
        if len(metrics_obj.msd):
            msd = metrics_obj.msd
            fields = xr.merge([
                fields,
                xr.Dataset(
                    {"msd": ("lag", msd["msd_m2"].to_numpy())},
                    coords={"lag": msd["lag_s"].to_numpy()},
                ),
            ])

        self._result = InstrumentResult(
            fields=fields,
            metrics=metrics,
            table=per_track,
            meta={
                "instrument": self.name,
                "config_hash": self.config.hash,
                "source": p.source_movie or "synthetic",
                "params": p.model_dump(mode="json"),
                "linking": "trackpy (Crocker-Grier), doi:10.1006/jcis.1996.0217",
            },
        )
        return self._result

    def view_napari(self, show: bool = True) -> Any:
        """Open the movie with a Napari tracks layer (optional back-end)."""
        from biosim_lab.core.viz.napari_layers import view_tracks

        self._ensure_setup()
        if self.tracks is None:
            self.run()
        return view_tracks(self.movie, self.tracks, show=show)

    def dashboard(self) -> Any:
        """Panel dashboard: track overlay, speed/persistence histograms, MSD."""
        from biosim_lab.instruments.cell_tracker.dashboard import build_dashboard

        return build_dashboard(self)

    @classmethod
    def example_config(cls) -> dict[str, Any]:
        return {
            "name": "cell_track_demo",
            "instrument": cls.name,
            "seed": 2,
            "description": "Synthetic 24-frame time-lapse of persistently migrating cells.",
            "params": {
                "pixel_size": "0.65 um",
                "frame_interval": "10 min",
                "n_frames": 24,
                "n_cells": 60,
                "search_range_px": 12.0,
                "backend": "classical",
            },
        }


__all__ = ["CellTracker", "CellTrackerParams"]
