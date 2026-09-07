"""Linking and motility metrics for live-cell tracking.

Linking
-------
Detections are linked frame to frame with ``trackpy``, which implements the
Crocker-Grier nearest-neighbour algorithm with a maximum displacement and a
memory for temporary disappearances.

Reference: Crocker & Grier (1996), *Methods of digital video microscopy for
colloidal studies*, J. Colloid Interface Sci. 179:298,
doi:10.1006/jcis.1996.0217.  Python implementation: Allan et al., *trackpy*,
doi:10.5281/zenodo.1226458.

Motility metrics
----------------
* **Speed** — mean instantaneous speed along the track [m/s].
* **Net displacement** and **path length** [m].
* **Directional persistence** ``D/L`` (net displacement over path length): 1 for
  a straight path, near 0 for a random walk.
* **Mean-squared displacement** ``MSD(tau) = <|r(t+tau) - r(t)|^2>``, and the
  anomalous exponent ``alpha`` from a log-log fit: ``alpha = 1`` is diffusive,
  ``alpha ~ 2`` ballistic, ``alpha < 1`` subdiffusive.

Reference for the MSD analysis of migrating cells: Selmeczi et al. (2005),
*Cell motility as persistent random motion*, Biophys. J. 89:912,
doi:10.1529/biophysj.105.061150.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.plugin import optional_import


def link_detections(
    detections: pd.DataFrame,
    *,
    search_range: float,
    memory: int = 2,
    min_track_length: int = 5,
) -> pd.DataFrame:
    """Link per-frame detections into tracks with ``trackpy``.

    Parameters
    ----------
    detections:
        Long table with ``frame``, ``y``, ``x`` columns (pixel units).
    search_range:
        Maximum displacement between consecutive frames [px].  Too small drops
        fast cells; too large produces identity swaps — the classic trade-off,
        and the reason ``search_range`` is exposed rather than guessed.
    memory:
        How many frames a cell may vanish for and still be relinked.
    min_track_length:
        Tracks shorter than this are discarded.
    """
    tp = optional_import("trackpy")
    if tp is None:
        raise RuntimeError(
            "trackpy is required for linking; install it with `pip install biosim-lab`"
        )
    required = {"frame", "y", "x"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"detections table is missing columns: {sorted(missing)}")

    tp.quiet()
    linked = tp.link(detections, search_range=search_range, memory=memory)
    if min_track_length > 1:
        linked = tp.filter_stubs(linked, threshold=min_track_length)
    return linked.reset_index(drop=True)


@dataclass
class TrackMetrics:
    """Per-track motility summary plus the ensemble MSD."""

    per_track: pd.DataFrame
    msd: pd.DataFrame
    alpha: float
    diffusion_coefficient: float


def track_metrics(
    tracks: pd.DataFrame,
    *,
    pixel_size: float,
    frame_interval: float,
    max_lag_frames: int | None = None,
) -> TrackMetrics:
    """Compute speed, persistence and MSD from a linked track table.

    Parameters
    ----------
    tracks:
        Output of :func:`link_detections` (``particle``, ``frame``, ``y``, ``x``).
    pixel_size:
        Metres per pixel.
    frame_interval:
        Seconds between frames.
    """
    if not len(tracks):
        empty = pd.DataFrame()
        return TrackMetrics(empty, empty, float("nan"), float("nan"))

    rows = []
    for pid, group in tracks.sort_values(["particle", "frame"]).groupby("particle"):
        y = group["y"].to_numpy() * pixel_size
        x = group["x"].to_numpy() * pixel_size
        t = group["frame"].to_numpy() * frame_interval
        if len(y) < 2:
            continue
        steps = np.hypot(np.diff(y), np.diff(x))
        dt = np.diff(t)
        path_length = float(steps.sum())
        net = float(np.hypot(y[-1] - y[0], x[-1] - x[0]))
        speeds = steps / np.where(dt > 0, dt, np.nan)
        angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])
        rows.append(
            {
                "particle": int(pid),
                "n_frames": int(len(y)),
                "duration_s": float(t[-1] - t[0]),
                "path_length_m": path_length,
                "net_displacement_m": net,
                "mean_speed_m_s": float(np.nanmean(speeds)),
                "max_speed_m_s": float(np.nanmax(speeds)),
                "persistence": float(net / path_length) if path_length > 0 else np.nan,
                "direction_rad": float(angle),
            }
        )
    per_track = pd.DataFrame(rows)

    msd_df = ensemble_msd(
        tracks, pixel_size=pixel_size, frame_interval=frame_interval,
        max_lag_frames=max_lag_frames,
    )
    alpha, d_coeff = fit_msd(msd_df)
    return TrackMetrics(per_track=per_track, msd=msd_df, alpha=alpha,
                        diffusion_coefficient=d_coeff)


def ensemble_msd(
    tracks: pd.DataFrame,
    *,
    pixel_size: float,
    frame_interval: float,
    max_lag_frames: int | None = None,
) -> pd.DataFrame:
    """Time- and ensemble-averaged mean-squared displacement.

    Returns a table with ``lag_s``, ``msd_m2`` and ``n_samples``.  Lags beyond a
    quarter of the shortest track are dropped, because their statistics are
    dominated by a handful of pairs.
    """
    grouped = list(tracks.sort_values(["particle", "frame"]).groupby("particle"))
    if not grouped:
        return pd.DataFrame(columns=["lag_s", "msd_m2", "n_samples"])
    longest = max(len(g) for _, g in grouped)
    max_lag = max_lag_frames or max(1, longest // 4)

    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1, dtype=int)
    for _, group in grouped:
        y = group["y"].to_numpy() * pixel_size
        x = group["x"].to_numpy() * pixel_size
        for lag in range(1, min(max_lag, len(y) - 1) + 1):
            d2 = (y[lag:] - y[:-lag]) ** 2 + (x[lag:] - x[:-lag]) ** 2
            sums[lag] += float(d2.sum())
            counts[lag] += int(d2.size)

    lags = np.arange(1, max_lag + 1)
    valid = counts[1:] > 0
    return pd.DataFrame(
        {
            "lag_s": lags[valid] * frame_interval,
            "msd_m2": sums[1:][valid] / counts[1:][valid],
            "n_samples": counts[1:][valid],
        }
    )


def fit_msd(msd: pd.DataFrame) -> tuple[float, float]:
    """Fit ``MSD = 4*D*tau^alpha`` (2-D) and return ``(alpha, D)``.

    ``alpha = 1`` is normal diffusion, ``alpha -> 2`` ballistic/directed
    motion, ``alpha < 1`` subdiffusive (confinement, crowding).
    """
    if len(msd) < 3:
        return float("nan"), float("nan")
    log_t = np.log(msd["lag_s"].to_numpy())
    log_m = np.log(msd["msd_m2"].to_numpy())
    good = np.isfinite(log_t) & np.isfinite(log_m)
    if good.sum() < 3:
        return float("nan"), float("nan")
    alpha, intercept = np.polyfit(log_t[good], log_m[good], 1)
    return float(alpha), float(np.exp(intercept) / 4.0)


def detections_from_labels(
    label_stack: np.ndarray, intensity_stack: np.ndarray | None = None
) -> pd.DataFrame:
    """Build a trackpy-compatible detection table from a stack of label images."""
    from skimage.measure import regionprops_table

    rows = []
    for frame, labels in enumerate(label_stack):
        if labels.max() == 0:
            continue
        props = regionprops_table(
            labels,
            intensity_image=None if intensity_stack is None else intensity_stack[frame],
            properties=("label", "centroid", "area")
            + (("mean_intensity",) if intensity_stack is not None else ()),
        )
        df = pd.DataFrame(props)
        df = df.rename(columns={"centroid-0": "y", "centroid-1": "x", "area": "mass"})
        df["frame"] = frame
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["frame", "y", "x", "mass"])
    return pd.concat(rows, ignore_index=True)


def build_lineage(tracks: pd.DataFrame) -> Any:
    """Reconstruct a division lineage with ``btrack`` (optional back-end).

    Raises a clear message when ``btrack`` is absent; the rest of the tracker
    works without it.

    Reference: Ulicna et al. (2021), *Automated deep lineage tree analysis
    using a Bayesian single cell tracking approach*, Front. Comput. Sci. 3:734559,
    doi:10.3389/fcomp.2021.734559.
    """
    btrack = optional_import("btrack")
    if btrack is None:
        raise RuntimeError(
            "btrack is not installed, so lineage trees are unavailable. "
            "Install it with `pip install biosim-lab[imaging]`. Speed, persistence "
            "and MSD metrics work without it."
        )
    raise NotImplementedError(
        "btrack lineage reconstruction is a Stage 3+ feature: the configuration "
        "surface is defined but the bridge is not implemented yet."
    )


__all__ = [
    "link_detections",
    "TrackMetrics",
    "track_metrics",
    "ensemble_msd",
    "fit_msd",
    "detections_from_labels",
    "build_lineage",
]
