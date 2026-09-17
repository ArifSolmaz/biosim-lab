"""Count a sorter's outcome from a microscope video, the way the papers did.

Neither reference paper computed its sorting metrics from a model. Zhang et al.
(2023, doi:10.3390/ijms24043338, Sec. 4.1, 4.6) counted cells in a video of the
chip: "the captured cancer cells or PBMCs were counted as the ones that
converged to the midline of the microchannel as observed in the microscope,
while the cancer cells and PBMCs were distinguished by size". Li et al.
(2015, doi:10.1073/pnas.1504484112) counted calcein-stained cancer cells by
fluorescence. A simulated sorter that reports its capture efficiency directly
skips the step where the real number is actually made --- and every error that
step introduces.

This module puts it back. It renders a synthetic microscope video of the
region just upstream of the sorter's outlets from the simulated trajectories,
segments every frame (:mod:`biosim_lab.core.imaging`), links the detections
into tracks with trackpy (:mod:`biosim_lab.instruments.cell_tracker.tracking`),
classifies each track --- by size, as Zhang did, or by a fluorescence channel,
as Li did --- assigns it an outlet from where it leaves the field of view, and
computes capture efficiency and contamination from those counts alone. The
simulation's own numbers are then used only to *score* the video readout:
which cells were never counted, which were misclassified, which were given
the wrong outlet.

Two things make the video faithful rather than decorative:

* **Cells arrive as a stream.** The simulation launches every cell at once;
  a real channel carries them continuously. Each cell's trajectory is
  shifted to its own arrival time (a Poisson stream sized to a chosen number
  of cells in view), which is exact because the model's cells do not interact.
* **Arrival is locked to the drive.** In ``mode="alternating_baw"`` a cell's
  history depends on which point of the switching cycle it entered at, and
  the simulation drew that point per cell. The arrival time is therefore
  placed on that same phase of the relay's clock, so the video shows each
  cell doing exactly what it did in the simulation.

This module sits above the instruments (like :mod:`biosim_lab.stages`): it is
where the sorter and the tracker meet, so neither has to know the other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import ndimage

from biosim_lab.core.imaging.segmentation import segment
from biosim_lab.core.imaging.synthetic import SyntheticImageSpec
from biosim_lab.core.materials import IMAGING_MODEL
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.statistics import wilson_interval
from biosim_lab.instruments.cell_tracker.tracking import close_gaps, link_detections
from biosim_lab.instruments.saw_sorter.metrics import paper_metrics
from biosim_lab.instruments.saw_sorter.simulate import (
    SAWSorterParams,
    SAWSorterSimulation,
    SortingOutcome,
)

_O = IMAGING_MODEL


@dataclass
class CameraSpec:
    """The microscope looking at the channel just upstream of the outlets.

    Attributes
    ----------
    pixel_size:
        Metres per pixel in the object plane.
    frame_rate:
        Frames per second. ``None`` picks the rate at which the fastest cell
        moves one smallest-cell radius per frame: slower than that and linking
        becomes ambiguous, because two cells can then sit closer together than
        one cell moves between frames.
    window_length:
        Length of channel in view, ending at the outlet plane [m].
    cells_in_view:
        Mean number of cells in the field of view (sets the arrival rate).
        Crowd it and cells touch, merge in segmentation and swap identities
        --- a real limit of video counting, deliberately exposed.
    margin_px:
        Rows of chip rendered beyond each channel wall, so a cell against a
        wall is not cut by the image border.
    optics:
        Brightfield rendering and noise (shared with the counter's synthetic
        images).
    fluorescence_contrast, fluorescence_background, fluorescence_noise:
        The second channel, in which only target cells are stained (as with
        the calcein-AM stain of Li et al. 2015).
    """

    # A chip-overview objective: the channel width has to fit the frame. The
    # counter's synthetic images run at the higher magnification you would put
    # on a counting chamber. Both are registered in the material library.
    pixel_size: float = float(_O.chip_pixel_size)
    frame_rate: float | None = None
    window_length: float = 250e-6
    cells_in_view: float = 6.0
    margin_px: int = 20
    optics: SyntheticImageSpec = field(default_factory=lambda: SyntheticImageSpec(
        blur_sigma=1.0, illumination_amplitude=0.06, shot_noise_photons=900.0,
        read_noise=0.012,
    ))
    fluorescence_contrast: float = float(_O.fluorescence_contrast)
    fluorescence_background: float = float(_O.fluorescence_background)
    fluorescence_noise: float = float(_O.fluorescence_noise)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _visible_intervals(pos: np.ndarray, time: np.ndarray, z0: float, z1: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per particle, the simulation times at which it enters and leaves ``[z0, z1]``."""
    n = pos.shape[0]
    t_in = np.full(n, np.nan)
    t_out = np.full(n, np.nan)
    for i in range(n):
        z = np.maximum.accumulate(pos[i, :, 2])  # frozen tails keep it monotone
        if z[-1] < z0:
            continue
        t_in[i] = float(np.interp(z0, z, time))
        t_out[i] = float(np.interp(z1, z, time)) if z[-1] >= z1 else float(time[-1])
    return t_in, t_out


def _arrivals(t_in: np.ndarray, rate: float, rng: np.random.Generator,
              period: float | None, phase: np.ndarray | None) -> np.ndarray:
    """Inlet times such that cells reach the window as a Poisson stream.

    The stream is laid out at the *window*, not the inlet: transit times
    differ by tens of seconds between the channel's middle and its walls, so a
    stream launched at the inlet would reach the camera spread over all of
    them. With a switched drive, each inlet time is then moved to the nearest
    instant at which the drive is at the phase this cell entered at in the
    simulation, so the cell repeats its simulated history exactly.
    """
    n = t_in.size
    reach = np.cumsum(rng.exponential(1.0 / rate, size=n))[rng.permutation(n)]
    arrival = reach - t_in
    if period is not None and phase is not None:
        arrival = phase + period * np.round((arrival - phase) / period)
    return arrival


def _draw_disc(canvas: np.ndarray, row: float, col: float, radius: float,
               value: float) -> None:
    """Add a filled disc to *canvas* in place, touching only its bounding box."""
    h, w = canvas.shape
    r0, r1 = int(max(0, np.floor(row - radius - 1))), int(min(h, np.ceil(row + radius + 2)))
    c0, c1 = int(max(0, np.floor(col - radius - 1))), int(min(w, np.ceil(col + radius + 2)))
    if r0 >= r1 or c0 >= c1:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
    canvas[r0:r1, c0:c1][inside] = value


def _calibrate(frames: list[np.ndarray], *, smoothing_sigma: float = 1.0
               ) -> tuple[np.ndarray, float]:
    """Static background and one segmentation threshold, from frames across the video.

    The background is the temporal median (cells cover a small fraction of
    any pixel's history, so it is the empty chip). The threshold is Otsu's
    over the pooled, background-subtracted and smoothed pixels --- the image
    :func:`~biosim_lab.core.imaging.segmentation.segment_classical` thresholds
    --- but never below six robust noise sigmas, so a stretch of video with
    few cells cannot pull it into the noise.
    """
    from skimage.filters import gaussian, threshold_otsu

    background = np.median(np.asarray(frames), axis=0)
    values = np.concatenate([
        gaussian(img - background, sigma=smoothing_sigma, preserve_range=True).ravel()
        for img in frames])
    centre = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - centre)))
    return background, max(float(threshold_otsu(values)), centre + 6.0 * noise)


def otsu_split(values: np.ndarray) -> float:
    """Otsu's two-class threshold for a handful of values, computed exactly.

    Every cut between consecutive sorted values is tried and the one with the
    largest between-class variance kept (Otsu 1979,
    doi:10.1109/TSMC.1979.4310076); ties --- every cut inside an empty gap
    scores the same --- go to the widest gap, and the threshold sits in its
    middle. A 256-bin histogram version returns the first tied bin instead,
    i.e. the very edge of the lower class, which misfiles its top member.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size < 2 or v[0] == v[-1]:
        return float("nan")
    n = v.size
    k = np.arange(1, n)
    csum = np.cumsum(v)[:-1]
    m0 = csum / k
    m1 = (v.sum() - csum) / (n - k)
    between = k * (n - k) * (m0 - m1) ** 2
    gap = np.diff(v)
    best = np.flatnonzero(np.isclose(between, between.max(), rtol=1e-9, atol=0.0))
    i = int(best[np.argmax(gap[best])])
    return float(0.5 * (v[i] + v[i + 1]))


def _illumination(shape: tuple[int, int], amplitude: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    yy = (yy - h / 2) / (h / 2)
    xx = (xx - w / 2) / (w / 2)
    return 1.0 + amplitude * (0.6 * xx + 0.4 * yy - 0.5 * (xx**2 + yy**2))


# ---------------------------------------------------------------------------
# the workflow
# ---------------------------------------------------------------------------


@dataclass
class SorterFilm:
    """A rendered, segmented and linked video of a sorter's outlet region.

    Everything a camera and a tracker produce, plus --- kept apart, and used
    only for scoring --- where each simulated cell was actually drawn.
    """

    detections: pd.DataFrame
    links: pd.DataFrame
    truth: pd.DataFrame
    sample_frames: np.ndarray
    sample_fluorescence: np.ndarray
    geometry: dict[str, Any]
    outcome: SortingOutcome
    is_target: np.ndarray
    in_view: np.ndarray


@dataclass
class VideoReadout:
    """What the video says, what the simulation says, and why they differ."""

    video_metrics: dict[str, Any]
    truth_metrics: dict[str, Any]
    error_budget: dict[str, Any]
    tracks: pd.DataFrame
    film: SorterFilm

    def summary(self) -> str:
        v, t, e = self.video_metrics, self.truth_metrics, self.error_budget
        return "\n".join([
            f"capture efficiency: video {v['capture_efficiency_percent']:.1f} % "
            f"[{v['capture_ci_low']:.1f}, {v['capture_ci_high']:.1f}] vs simulation "
            f"{t['capture_efficiency_percent']:.1f} %",
            f"contamination:      video {v['contamination_rate_percent']:.1f} % vs "
            f"simulation {t['contamination_rate_percent']:.1f} %",
            f"cells counted {e['counted_fraction_percent']:.1f} %, classified correctly "
            f"{e['classification_accuracy_percent']:.1f} %, outlet agrees "
            f"{e['outlet_agreement_percent']:.1f} %",
        ])


def film_sorter(
    params: SAWSorterParams,
    *,
    camera: CameraSpec | None = None,
    backend: Literal["classical", "cellpose", "stardist"] = "classical",
    seed: int | None = 0,
    outcome: SortingOutcome | None = None,
    n_cells: int | None = None,
    keep_frames: int = 90,
    keep_stride: int = 1,
    max_gap_frames: int = 30,
    max_frames: int = 30000,
) -> SorterFilm:
    """Simulate a sort (or take a finished one), film the outlet region, track the cells.

    Parameters
    ----------
    params:
        The sorter. When it is simulated here, trajectories are sampled at
        least 1500 times, densely enough to render.
    camera:
        Field of view and optics.
    backend:
        Segmentation back-end (watershed by default).
    outcome:
        A finished run of the same *params*, to skip re-simulating.
    n_cells:
        Film a random subset of this many of the simulated cells (the rest
        are simply not in the video); ``None`` films all of them.
    keep_frames, keep_stride:
        Keep this many frames, every *keep_stride*-th, centred on the middle of
        the video (where the stream is steady), for figures and animations.
    max_gap_frames:
        Longest occlusion bridged by gap closing
        (:func:`~biosim_lab.instruments.cell_tracker.tracking.close_gaps`); 0
        turns gap closing off.
    """
    cam = camera or CameraSpec()
    rng = np.random.default_rng(seed)
    if outcome is None:
        params = params.model_copy(
            update={"n_time_samples": max(params.n_time_samples, 1500)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        sim = SAWSorterSimulation(params)
        if outcome is None:
            outcome = sim.run()
    lo, hi = sim.collection_bounds
    width, length = params.channel_width, params.channel_length
    z0 = length - cam.window_length
    if z0 < 0:
        raise ValueError("camera window is longer than the channel")

    tr = outcome.tracks.trajectories
    pos = tr["position"].values
    time = tr["time"].values
    radius = tr["radius"].values
    is_target = outcome.cells.set_index("particle").loc[
        tr["particle"].values, "is_target"].to_numpy(dtype=bool)

    t_in, t_out = _visible_intervals(pos, time, z0, length)
    seen = np.isfinite(t_in)
    if n_cells is not None and n_cells < seen.sum():
        keep = rng.choice(np.flatnonzero(seen), size=n_cells, replace=False)
        seen = np.isin(np.arange(seen.size), keep)
    if not seen.any():
        raise ValueError("no cell reaches the camera window")
    rate = cam.cells_in_view / float(np.nanmedian(t_out - t_in))
    period = sim.schedule.period if sim.schedule is not None else None
    phase = None
    if period is not None:
        if "entry_offset_s" not in outcome.cells:
            raise ValueError("the sorter outcome does not record per-cell entry phases")
        phase = outcome.cells["entry_offset_s"].to_numpy(dtype=float)
    arrival = _arrivals(np.nan_to_num(t_in), rate, rng, period, phase)
    start = arrival + t_in
    stop = arrival + t_out

    px = cam.pixel_size
    rows = int(np.ceil(width / px)) + 2 * cam.margin_px
    cols = int(np.ceil(cam.window_length / px))
    u_max = sim.flow.max_velocity
    r_min = float(np.min(radius[seen]))
    # Auto frame rate: the fastest cell moves one smallest-cell radius per frame,
    # so a search radius of 1.5 steps stays below the 2 r_min two cells can get.
    frame_rate = cam.frame_rate if cam.frame_rate is not None else u_max / r_min
    dt = 1.0 / frame_rate
    t_first, t_last = float(np.min(start[seen])), float(np.max(stop[seen]))
    n_frames = int(np.ceil((t_last - t_first) / dt)) + 1
    if n_frames > max_frames:
        raise ValueError(
            f"{n_frames} frames needed (> max_frames={max_frames}); raise cells_in_view, "
            "lower the frame rate, or simulate fewer cells"
        )

    opt = cam.optics
    illum = _illumination((rows, cols), opt.illumination_amplitude)
    order = np.argsort(start)
    keep_first = max(0, n_frames // 2 - (keep_frames * keep_stride) // 2)

    def render(k: int, noise: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, float, float]]]:
        """Brightfield and fluorescence frame *k*, and where each cell was drawn."""
        tau = t_first + k * dt
        live = order[(start[order] <= tau) & (stop[order] >= tau) & seen[order]]
        bf_signal = np.zeros((rows, cols))
        fl_signal = np.zeros((rows, cols))
        drawn = []
        for i in live:
            t_sim = tau - arrival[i]
            row = cam.margin_px + float(np.interp(t_sim, time, pos[i, :, 0])) / px
            col = (float(np.interp(t_sim, time, pos[i, :, 2])) - z0) / px
            r_px = radius[i] / px
            _draw_disc(bf_signal, row, col, r_px, opt.cell_contrast)
            if is_target[i]:
                _draw_disc(fl_signal, row, col, r_px, cam.fluorescence_contrast)
            drawn.append((k, int(i), row, col, r_px))
        bf = (opt.background_level + ndimage.gaussian_filter(bf_signal, opt.blur_sigma)) * illum
        # Shot noise in its Gaussian limit (hundreds of photons per pixel) plus read noise.
        var = opt.read_noise**2 + (np.clip(bf, 0, None) / opt.shot_noise_photons
                                   if opt.shot_noise_photons > 0 else 0.0)
        bf = np.clip(bf + np.sqrt(var) * noise.standard_normal(bf.shape), 0.0, 1.0)
        fl = cam.fluorescence_background + ndimage.gaussian_filter(fl_signal, opt.blur_sigma)
        fl = np.clip(fl + noise.normal(0.0, cam.fluorescence_noise, fl.shape), 0.0, 1.0)
        return bf, fl, drawn

    # One background and one threshold for the whole video, as an operator would
    # fix them: Otsu per frame splits pure noise into specks whenever no whole
    # cell is in view.
    min_r_px = max(1.5, 0.6 * r_min / px)
    seg_kw: dict[str, Any] = {}
    if backend == "classical":
        background, level = _calibrate(
            [render(k, np.random.default_rng((seed or 0, k)))[0]
             for k in np.linspace(0, n_frames - 1, min(n_frames, 41)).astype(int)])
        seg_kw = {"min_radius_px": min_r_px, "threshold": level, "background": background}

    det_rows: list[pd.DataFrame] = []
    truth_rows: list[tuple[int, int, float, float, float]] = []
    kept_bf: list[np.ndarray] = []
    kept_fl: list[np.ndarray] = []
    for k in range(n_frames):
        bf, fl, drawn = render(k, rng)
        truth_rows.extend(drawn)
        if k >= keep_first and (k - keep_first) % keep_stride == 0 \
                and len(kept_bf) < keep_frames:
            kept_bf.append(bf.astype(np.float32))
            kept_fl.append(fl.astype(np.float32))
        seg = segment(bf, backend=backend, **seg_kw)
        props = seg.properties
        if len(props) == 0:
            continue
        index = props["label"].to_numpy()
        det_rows.append(pd.DataFrame({
            "frame": k,
            "y": props["centroid-0"].to_numpy(dtype=float),
            "x": props["centroid-1"].to_numpy(dtype=float),
            "area_px": props["area"].to_numpy(dtype=float),
            "diameter_um": props["equivalent_diameter"].to_numpy(dtype=float) * px * 1e6,
            "fluorescence": np.asarray(ndimage.mean(fl, seg.labels, index), dtype=float),
        }))

    columns = ["frame", "y", "x", "area_px", "diameter_um", "fluorescence"]
    detections = pd.concat(det_rows, ignore_index=True) if det_rows \
        else pd.DataFrame(columns=columns)
    truth = pd.DataFrame(truth_rows, columns=["frame", "particle", "y", "x", "radius_px"])
    search = max(3.0, 1.5 * u_max * dt / px)
    if len(detections):
        # Link everything, rejoin pieces an occlusion split (a cell passing
        # behind another on the same node line), and only then drop stubs.
        links = link_detections(detections, search_range=search, memory=2,
                                min_track_length=1)
        if max_gap_frames > 0:
            links = close_gaps(links, max_gap=max_gap_frames, max_distance=search)
        links = links[links.groupby("particle")["frame"].transform("size") >= 3]
        links = links.reset_index(drop=True)
    else:
        links = detections.assign(particle=pd.Series(dtype=int))
    geometry = {
        "rows": rows, "cols": cols, "pixel_size_m": px, "frame_rate_hz": frame_rate,
        "frame_interval_s": dt, "n_frames": n_frames, "search_range_px": search,
        "max_gap_frames": max_gap_frames,
        "margin_px": cam.margin_px, "collection_um": (lo * 1e6, hi * 1e6),
        "channel_width_um": width * 1e6, "window_start_m": z0,
        "min_radius_px": min_r_px, "cells_in_view": cam.cells_in_view,
        "segmentation_threshold": seg_kw.get("threshold"), "keep_stride": keep_stride,
        "sample_first_frame": keep_first,
        "n_cells_filmed": int(seen.sum()),
    }
    return SorterFilm(
        detections=detections, links=links, truth=truth,
        sample_frames=np.asarray(kept_bf), sample_fluorescence=np.asarray(kept_fl),
        geometry=geometry, outcome=outcome, is_target=is_target, in_view=seen,
    )


def count_film(film: SorterFilm, classify_by: Literal["size", "fluorescence"] = "size",
               *, exit_fraction: float = 0.85) -> VideoReadout:
    """Count capture efficiency and contamination from the video alone.

    Parameters
    ----------
    classify_by:
        ``size`` --- each track's median equivalent diameter, split in two by
        Otsu's threshold on its logarithm (the populations of Zhang et al.
        2023 "were distinguished by size"); ``fluorescence`` --- its median
        signal in the stained channel (calcein-stained targets, Li et al.
        2015). Neither sees the simulation's labels.
    exit_fraction:
        A track is counted if it crosses this fraction of the window's length
        (i.e. it reached the outlets); its outlet is read from its lateral
        position when last seen.
    """
    g = film.geometry
    px_um = g["pixel_size_m"] * 1e6
    tracks = _summarise_tracks(film.links, px_um=px_um, margin=g["margin_px"])
    exit_col = exit_fraction * g["cols"]
    counted = tracks[(tracks["x_last"] >= exit_col) & (tracks["x_first"] < exit_col)].copy()
    feature = "diameter_um" if classify_by == "size" else "fluorescence"
    values = counted[feature].to_numpy(dtype=float)
    scale = np.log(values) if classify_by == "size" else values
    cut = otsu_split(scale)
    lo, hi = g["collection_um"]
    counted["is_target"] = scale > cut
    counted["outlet"] = np.where(
        (counted["lateral_um"] >= lo) & (counted["lateral_um"] <= hi), "collect", "waste")
    counted["x_outlet_m"] = counted["lateral_um"] * 1e-6
    counted["exited"] = True  # counted tracks are the ones that reached the outlets
    # Scoring only: the simulated cell each track sat on as it crossed the
    # counting line (which is what its count and outlet refer to), and the one
    # it sat on over its whole length --- they differ after an identity swap.
    counted["particle_truth"] = _match_truth(film.links, film.truth, counted.index,
                                             from_col=exit_col)
    counted["particle_truth_whole"] = _match_truth(film.links, film.truth, counted.index,
                                                   one_to_one=False)

    video: dict[str, Any] = dict(paper_metrics(counted))
    n_t = int(counted["is_target"].sum())
    ci = wilson_interval(int((counted["is_target"] & (counted["outlet"] == "collect")).sum()),
                         n_t) if n_t else None
    video.update({
        "classify_by": classify_by,
        "threshold": float(np.exp(cut)) if classify_by == "size" else cut,
        "threshold_units": "um diameter" if classify_by == "size" else "intensity",
        "n_tracks_counted": int(len(counted)),
        "n_target": n_t,
        "n_background": int(len(counted) - n_t),
        "capture_ci_low": ci.low_percent if ci else float("nan"),
        "capture_ci_high": ci.high_percent if ci else float("nan"),
    })
    # The simulation's own numbers for the same cells: those the camera saw.
    cells = film.outcome.cells.set_index("particle").loc[np.flatnonzero(film.in_view)]
    truth_metrics: dict[str, Any] = paper_metrics(cells)
    truth_metrics.update(n_target=int(cells["is_target"].sum()),
                         n_background=int((~cells["is_target"]).sum()))
    return VideoReadout(
        video_metrics=video, truth_metrics=truth_metrics,
        error_budget=_error_budget(counted, film),
        tracks=counted, film=film,
    )


def sorter_video_readout(
    params: SAWSorterParams,
    *,
    classify_by: Literal["size", "fluorescence"] = "size",
    **film_kwargs: Any,
) -> VideoReadout:
    """:func:`film_sorter` then :func:`count_film`, in one call."""
    return count_film(film_sorter(params, **film_kwargs), classify_by)


def _summarise_tracks(links: pd.DataFrame, *, px_um: float, margin: int) -> pd.DataFrame:
    """One row per track: extent along the flow, size, stain, and where it was last seen."""
    columns = ["particle", "n_frames", "x_first", "x_last", "lateral_um", "diameter_um",
               "fluorescence"]
    if not len(links):
        return pd.DataFrame(columns=columns)
    ordered = links.sort_values("frame")
    per = ordered.groupby("particle")
    out = pd.DataFrame({
        "particle": per.size().index.astype(int),
        "n_frames": per.size().to_numpy(),
        "x_first": per["x"].first().to_numpy(),
        "x_last": per["x"].last().to_numpy(),
        "lateral_um": (per["y"].last().to_numpy() - margin) * px_um,
        "diameter_um": per["diameter_um"].median().to_numpy(),
        "fluorescence": per["fluorescence"].median().to_numpy(),
    })
    return out.set_index("particle", drop=False)


def _match_truth(links: pd.DataFrame, truth: pd.DataFrame, which: pd.Index, *,
                 from_col: float = 0.0, one_to_one: bool = True) -> list[int]:
    """For each track in *which*, the simulated cell it sat on (``-1``: none).

    Only detections at ``x >= from_col`` vote, so passing the counting line
    decides the match. A detection sits on a cell if its centroid is within
    that cell's radius (at least 3 px) of where the cell was drawn in that
    frame. With *one_to_one*, tracks and cells are paired to maximise the
    total votes (Hungarian assignment), so when two cells cross merged and
    the blob is split in two, each half is credited to a different cell
    rather than both to the larger one; otherwise each track takes the cell
    it sat on most often.
    """
    from scipy.optimize import linear_sum_assignment

    det = links.loc[links["particle"].isin(which) & (links["x"] >= from_col),
                    ["frame", "particle", "y", "x"]]
    det = det.reset_index(drop=True).rename(columns={"particle": "track"})
    det["det"] = np.arange(len(det))
    pairs = det.merge(truth.rename(columns={"y": "ty", "x": "tx"}), on="frame")
    pairs["d"] = np.hypot(pairs["y"] - pairs["ty"], pairs["x"] - pairs["tx"])
    pairs = pairs[pairs["d"] <= np.maximum(3.0, pairs["radius_px"])]
    if not one_to_one:
        nearest = pairs.loc[pairs.groupby("det")["d"].idxmin(), ["track", "particle"]]
        votes = nearest.groupby("track")["particle"].agg(lambda v: int(v.mode().iloc[0]))
        return [int(votes.get(pid, -1)) for pid in which]
    tally = pairs.groupby(["track", "particle"]).size()
    if tally.empty:
        return [-1] * len(which)
    table = tally.unstack(fill_value=0)
    rows, cols = linear_sum_assignment(-table.to_numpy())
    match = {table.index[r]: int(table.columns[c]) for r, c in zip(rows, cols)
             if table.iat[r, c] > 0}
    return [match.get(pid, -1) for pid in which]


def _occluded(truth: pd.DataFrame, missed: np.ndarray, min_frames: int = 3) -> np.ndarray:
    """Which missed cells overlapped another cell, in projection, while in view.

    Two discs overlap when their centres are closer than the sum of their
    radii. On a node line cells at different heights overtake one another,
    so this is how a video loses them: merged into one blob at the counting
    line, or tangled in a cluster until the tracker gives up.
    """
    if missed.size == 0:
        return np.zeros(0, dtype=bool)
    mine = truth[truth["particle"].isin(missed)]
    pairs = mine.merge(truth, on="frame", suffixes=("", "_o"))
    pairs = pairs[pairs["particle"] != pairs["particle_o"]]
    touching = np.hypot(pairs["y"] - pairs["y_o"], pairs["x"] - pairs["x_o"]) \
        < pairs["radius_px"] + pairs["radius_px_o"]
    frames = pairs[touching].groupby("particle")["frame"].nunique()
    return frames.reindex(missed, fill_value=0).to_numpy() >= min_frames


def _error_budget(counted: pd.DataFrame, film: SorterFilm) -> dict[str, Any]:
    """Where the video readout departs from the simulation, cell by cell."""
    is_target, seen = film.is_target, film.in_view
    matched = counted[counted["particle_truth"] >= 0]
    ids = matched["particle_truth"].to_numpy(dtype=int)
    unique = np.unique(ids)
    n_seen = int(seen.sum())
    missed = np.setdiff1d(np.flatnonzero(seen), unique)
    occluded = _occluded(film.truth, missed)
    true_target = is_target[ids]
    classified = matched["is_target"].to_numpy(dtype=bool)
    true_outlet = film.outcome.cells.set_index("particle")["outlet"]
    sim_outlet = true_outlet.loc[ids].to_numpy()

    def missed_share(select: np.ndarray) -> float:
        pool = np.flatnonzero(select & seen)
        return 100.0 * float(np.isin(pool, missed).mean()) if pool.size else float("nan")

    collected = (true_outlet.reindex(np.arange(seen.size)) == "collect").to_numpy()
    agree = matched["outlet"].to_numpy() == sim_outlet
    return {
        "cells_through_window": n_seen,
        "counted_fraction_percent": 100.0 * unique.size / n_seen if n_seen else float("nan"),
        "counted_target_percent": 100.0 * float(
            np.isin(np.flatnonzero(is_target & seen), unique).mean()),
        "counted_background_percent": 100.0 * float(
            np.isin(np.flatnonzero(~is_target & seen), unique).mean()),
        "missed_cells": int(missed.size),
        "missed_occluded": int(occluded.sum()),
        "missed_other": int((~occluded).sum()),
        "missed_targets": int(is_target[missed].sum()),
        # A contaminant is missed more often when it shares its line with the
        # (larger) targets; if so, a video count under-reports contamination.
        "missed_background_collected_percent": missed_share(~is_target & collected),
        "missed_background_waste_percent": missed_share(~is_target & ~collected),
        "extra_tracks": int((counted["particle_truth"] < 0).sum()),
        "identity_swaps": int(
            (matched["particle_truth_whole"] != matched["particle_truth"]).sum()),
        "classification_accuracy_percent": 100.0 * float((true_target == classified).mean())
        if ids.size else float("nan"),
        "targets_called_background": int((true_target & ~classified).sum()),
        "background_called_targets": int((~true_target & classified).sum()),
        "outlet_agreement_percent": 100.0 * float(agree.mean()) if ids.size else float("nan"),
        "outlet_disagreements": int((~agree).sum()),
    }


__all__ = [
    "CameraSpec",
    "SorterFilm",
    "VideoReadout",
    "count_film",
    "film_sorter",
    "otsu_split",
    "sorter_video_readout",
]
