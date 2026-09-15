"""Synthetic microscopy image and video generator.

Image-analysis instruments need ground truth to be testable at all: with a
synthetic field of view the true cell count, the true radii and the true tracks
are known exactly, so segmentation and linking can be scored rather than
eyeballed.  The same generator produces the demo data for ``cell_counter`` and
``cell_tracker``.

The image model is deliberately simple and explicit:

* cells are discs with a soft (Gaussian-blurred) edge, on a slowly varying
  illumination background — the two artefacts that break naive thresholding;
* dead cells are darker, standing in for trypan-blue uptake;
* Poisson (shot) plus Gaussian (read) noise, the standard CCD/CMOS noise model
  (Janesick, *Photon Transfer*, doi:10.1117/3.725073);
* touching cells are allowed, because separating them is the whole point of the
  watershed step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticImageSpec:
    """Parameters of a synthetic field of view.

    Attributes
    ----------
    shape:
        ``(height, width)`` in pixels.
    pixel_size:
        Physical size of one pixel [m]; a 10x objective on a 6.5 um sensor
        pixel gives 0.65 um/px.
    n_cells:
        Number of cells to place.
    radius_mean_px, radius_cv:
        Log-normal radius distribution in pixels.
    dead_fraction:
        Fraction of cells rendered as dead (trypan-blue positive).
    background_level, cell_contrast:
        Base intensity and the cell's intensity increment, in the 0-1 range
        before noise.
    allow_touching:
        When ``False`` cells are rejected until they are non-overlapping.
    """

    shape: tuple[int, int] = (512, 512)
    pixel_size: float = 0.65e-6
    n_cells: int = 120
    radius_mean_px: float = 9.0
    radius_cv: float = 0.18
    dead_fraction: float = 0.15
    background_level: float = 0.25
    cell_contrast: float = 0.45
    dead_contrast_factor: float = -0.55
    blur_sigma: float = 1.4
    illumination_amplitude: float = 0.12
    shot_noise_photons: float = 900.0
    read_noise: float = 0.012
    allow_touching: bool = True
    margin_px: int = 6
    seed: int | None = 0
    extra: dict = field(default_factory=dict)


def _illumination(shape: tuple[int, int], amplitude: float) -> np.ndarray:
    """Smooth multiplicative illumination gradient (vignetting-like)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    yy = (yy - h / 2) / (h / 2)
    xx = (xx - w / 2) / (w / 2)
    return 1.0 + amplitude * (0.6 * xx + 0.4 * yy - 0.5 * (xx**2 + yy**2))


def _place_cells(spec: SyntheticImageSpec, rng: np.random.Generator) -> np.ndarray:
    """Sample cell centres and radii; returns ``(n, 3)`` of ``(y, x, r)``."""
    h, w = spec.shape
    sigma = np.sqrt(np.log1p(spec.radius_cv**2))
    mu = np.log(spec.radius_mean_px) - 0.5 * sigma**2
    out: list[tuple[float, float, float]] = []
    attempts = 0
    while len(out) < spec.n_cells and attempts < spec.n_cells * 200:
        attempts += 1
        r = float(rng.lognormal(mu, sigma))
        y = float(rng.uniform(spec.margin_px + r, h - spec.margin_px - r))
        x = float(rng.uniform(spec.margin_px + r, w - spec.margin_px - r))
        if not spec.allow_touching and any(
            (y - yy) ** 2 + (x - xx) ** 2 < (r + rr) ** 2 for yy, xx, rr in out
        ):
            continue
        out.append((y, x, r))
    return np.asarray(out, dtype=float).reshape(-1, 3)


def synthetic_field(spec: SyntheticImageSpec | None = None) -> dict:
    """Render one field of view with ground truth.

    Returns
    -------
    dict
        ``image`` (float32, 0-1), ``labels`` (int ground-truth mask),
        ``truth`` (record array of y, x, radius_px, alive) and ``spec``.
    """
    spec = spec or SyntheticImageSpec()
    rng = np.random.default_rng(spec.seed)
    h, w = spec.shape
    cells = _place_cells(spec, rng)
    n = len(cells)
    alive = rng.random(n) >= spec.dead_fraction

    signal = np.zeros(spec.shape, dtype=float)
    labels = np.zeros(spec.shape, dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    # Draw largest first so a small cell on top still gets its own label.
    order = np.argsort(-cells[:, 2])
    for new_id, i in enumerate(order, start=1):
        cy, cx, r = cells[i]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        contrast = spec.cell_contrast * (
            1.0 if alive[i] else 1.0 + spec.dead_contrast_factor
        )
        signal[disc] = contrast
        labels[disc] = new_id

    image = spec.background_level + gaussian_filter(signal, spec.blur_sigma)
    image = image * _illumination(spec.shape, spec.illumination_amplitude)

    # Photon shot noise, then read noise.
    if spec.shot_noise_photons > 0:
        image = rng.poisson(np.clip(image, 0, None) * spec.shot_noise_photons) / (
            spec.shot_noise_photons
        )
    image = image + rng.normal(0.0, spec.read_noise, size=spec.shape)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    truth = np.zeros(
        n,
        dtype=[("y", float), ("x", float), ("radius_px", float), ("alive", bool),
               ("label", np.int32)],
    )
    truth["y"] = cells[:, 0]
    truth["x"] = cells[:, 1]
    truth["radius_px"] = cells[:, 2]
    truth["alive"] = alive
    for new_id, i in enumerate(order, start=1):
        truth["label"][i] = new_id

    return {"image": image, "labels": labels, "truth": truth, "spec": spec}


def synthetic_movie(
    spec: SyntheticImageSpec | None = None,
    *,
    n_frames: int = 30,
    speed_px_per_frame: float = 2.2,
    persistence: float = 0.85,
    drift: tuple[float, float] = (0.0, 0.6),
) -> dict:
    """Render a time-lapse with known ground-truth tracks.

    Motion model: a persistent random walk — each step keeps a fraction
    *persistence* of the previous direction and adds a fresh random component,
    plus a constant *drift*.  This is the standard model for migrating cells and
    yields a mean-squared displacement that is ballistic at short lag and
    diffusive at long lag (Selmeczi et al. (2005), Biophys. J. 89:912,
    doi:10.1529/biophysj.105.061150).

    Returns
    -------
    dict
        ``movie`` ``(t, y, x)``, ``truth`` (long-format DataFrame-like record
        array with frame, particle, y, x) and ``spec``.
    """
    spec = spec or SyntheticImageSpec()
    rng = np.random.default_rng(spec.seed)
    h, w = spec.shape

    base = _place_cells(spec, rng)
    n = len(base)
    radii = base[:, 2]
    pos = base[:, :2].copy()
    direction = rng.normal(size=(n, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    frames = np.zeros((n_frames, h, w), dtype=np.float32)
    records = []
    yy, xx = np.mgrid[0:h, 0:w]

    for t in range(n_frames):
        signal = np.zeros(spec.shape, dtype=float)
        for i in range(n):
            cy, cx = pos[i]
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radii[i] ** 2
            signal[disc] = spec.cell_contrast
            records.append((t, i, cy, cx, radii[i]))

        image = spec.background_level + gaussian_filter(signal, spec.blur_sigma)
        image = image * _illumination(spec.shape, spec.illumination_amplitude)
        if spec.shot_noise_photons > 0:
            image = rng.poisson(np.clip(image, 0, None) * spec.shot_noise_photons) / (
                spec.shot_noise_photons
            )
        image = image + rng.normal(0.0, spec.read_noise, size=spec.shape)
        frames[t] = np.clip(image, 0.0, 1.0).astype(np.float32)

        step = rng.normal(size=(n, 2))
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        direction = persistence * direction + (1.0 - persistence) * step
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        pos = pos + speed_px_per_frame * direction + np.asarray(drift)
        # Reflect at the border so the population size stays constant.
        for axis, limit in enumerate((h, w)):
            lo = spec.margin_px + radii
            hi = limit - spec.margin_px - radii
            pos[:, axis] = np.clip(pos[:, axis], lo, hi)

    truth = np.array(
        records,
        dtype=[("frame", int), ("particle", int), ("y", float), ("x", float),
               ("radius_px", float)],
    )
    return {"movie": frames, "truth": truth, "spec": spec}


__all__ = ["SyntheticImageSpec", "synthetic_field", "synthetic_movie"]
