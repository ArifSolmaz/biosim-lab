"""Pluggable segmentation back-ends for the image instruments.

The **classical** back-end (Otsu threshold -> distance transform -> watershed)
is the default because it has no dependency beyond scikit-image and no model
weights to download.  Cellpose and StarDist are optional: they are selected by
name, and if the package is missing the caller gets a clear message naming the
extra to install rather than an ImportError at import time.

Watershed reference: Beucher & Meyer, *The morphological approach to
segmentation: the watershed transformation*, in Dougherty (ed.),
*Mathematical Morphology in Image Processing* (1993), ISBN 978-0824787240.
The distance-transform seeding used here is the standard recipe for splitting
touching convex objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from biosim_lab.core.plugin import optional_import

SegmentationBackend = Literal["classical", "cellpose", "stardist"]


@dataclass
class SegmentationResult:
    """Label image plus the per-object measurements downstream code needs."""

    labels: np.ndarray
    properties: Any  # pandas.DataFrame
    backend: str
    parameters: dict[str, Any]

    @property
    def n_objects(self) -> int:
        return int(self.labels.max())


def available_backends() -> dict[str, tuple[bool, str]]:
    """Report which segmentation back-ends can actually run here."""
    out: dict[str, tuple[bool, str]] = {}
    skimage = optional_import("skimage")
    out["classical"] = (
        skimage is not None,
        "scikit-image" if skimage else "scikit-image is not installed",
    )
    for name, module, extra in (
        ("cellpose", "cellpose", "biosim-lab[segmentation]"),
        ("stardist", "stardist", "biosim-lab[segmentation]"),
    ):
        mod = optional_import(module)
        out[name] = (
            mod is not None,
            f"{module} {getattr(mod, '__version__', '')}".strip()
            if mod
            else f"not installed (pip install {extra})",
        )
    return out


def segment_classical(
    image: np.ndarray,
    *,
    min_radius_px: float = 4.0,
    smoothing_sigma: float = 1.0,
    background_sigma: float = 30.0,
    min_distance: int | None = None,
    threshold: float | None = None,
    clear_border: bool = True,
) -> SegmentationResult:
    """Otsu + distance-transform watershed segmentation.

    Steps, each undoing one artefact the generator puts in:

    1. Gaussian background subtraction removes the illumination gradient.
    2. Gaussian smoothing suppresses shot noise.
    3. Otsu thresholding separates cells from background.
    4. Small objects and border-touching objects are removed (a cell cut by the
       frame edge has no valid area and would bias the size distribution).
    5. Peaks of the Euclidean distance transform seed a watershed, which splits
       touching cells.

    Parameters
    ----------
    min_radius_px:
        Objects smaller than a disc of this radius are discarded as debris.
    min_distance:
        Minimum separation of watershed seeds; defaults to ``min_radius_px``.
    threshold:
        Fixed threshold; ``None`` uses Otsu.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, threshold_otsu
    from skimage.segmentation import clear_border as sk_clear_border
    from skimage.segmentation import watershed

    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"segment_classical expects a 2-D image, got shape {img.shape}")

    background = gaussian(img, sigma=background_sigma, preserve_range=True)
    flat = img - background
    smooth = gaussian(flat, sigma=smoothing_sigma, preserve_range=True)

    level = threshold_otsu(smooth) if threshold is None else float(threshold)
    mask = smooth > level

    # Drop debris by connected-component area. Done directly rather than through
    # skimage.morphology.remove_small_objects, whose min_size/max_size semantics
    # changed across versions.
    min_area = max(int(np.pi * min_radius_px**2), 4)
    components, _ = ndi.label(mask)
    sizes = np.bincount(components.ravel())
    small = np.flatnonzero(sizes < min_area)
    if small.size:
        mask = mask & ~np.isin(components, small)
    mask = ndi.binary_fill_holes(mask)

    distance = ndi.distance_transform_edt(mask)
    seeds_coords = peak_local_max(
        distance,
        min_distance=int(min_distance or max(min_radius_px, 3)),
        labels=mask,
        exclude_border=False,
    )
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (r, c) in enumerate(seeds_coords, start=1):
        markers[r, c] = i
    labels = watershed(-distance, markers, mask=mask)

    if clear_border:
        labels = sk_clear_border(labels)
    labels = _relabel_sequential(labels)

    props = _measure(labels, img)
    return SegmentationResult(
        labels=labels,
        properties=props,
        backend="classical",
        parameters={
            "min_radius_px": min_radius_px,
            "smoothing_sigma": smoothing_sigma,
            "background_sigma": background_sigma,
            "threshold": level,
            "clear_border": clear_border,
        },
    )


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber labels 1..N with no gaps."""
    unique = np.unique(labels)
    unique = unique[unique != 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[unique] = np.arange(1, len(unique) + 1, dtype=np.int32)
    return lookup[labels]


def _measure(labels: np.ndarray, intensity: np.ndarray) -> Any:
    """Per-object measurements as a DataFrame."""
    import pandas as pd
    from skimage.measure import regionprops_table

    if labels.max() == 0:
        return pd.DataFrame(
            columns=["label", "centroid-0", "centroid-1", "area", "equivalent_diameter",
                     "mean_intensity", "eccentricity", "solidity"]
        )
    table = regionprops_table(
        labels,
        intensity_image=intensity,
        properties=(
            "label", "centroid", "area", "equivalent_diameter",
            "mean_intensity", "eccentricity", "solidity", "perimeter",
        ),
    )
    return pd.DataFrame(table)


def segment_cellpose(
    image: np.ndarray, *, diameter: float | None = None, model_type: str = "cyto3"
) -> SegmentationResult:
    """Cellpose segmentation (optional back-end).

    Reference: Stringer et al. (2021), *Cellpose: a generalist algorithm for
    cellular segmentation*, Nat. Methods 18:100,
    doi:10.1038/s41592-020-01018-x.
    """
    cellpose = optional_import("cellpose.models")
    if cellpose is None:
        raise RuntimeError(
            "cellpose is not installed. Install it with "
            "`pip install biosim-lab[segmentation]`, or use backend='classical'."
        )
    model = cellpose.Cellpose(model_type=model_type)
    masks, _flows, _styles, _diams = model.eval(
        np.asarray(image), diameter=diameter, channels=[0, 0]
    )
    labels = _relabel_sequential(np.asarray(masks, dtype=np.int32))
    return SegmentationResult(
        labels=labels,
        properties=_measure(labels, np.asarray(image, dtype=float)),
        backend="cellpose",
        parameters={"diameter": diameter, "model_type": model_type},
    )


def segment_stardist(image: np.ndarray, *, model_name: str = "2D_versatile_fluo"):
    """StarDist segmentation (optional back-end).

    Reference: Schmidt et al. (2018), *Cell detection with star-convex
    polygons*, MICCAI, doi:10.1007/978-3-030-00934-2_30.
    """
    stardist = optional_import("stardist.models")
    csbdeep = optional_import("csbdeep.utils")
    if stardist is None or csbdeep is None:
        raise RuntimeError(
            "stardist is not installed. Install it with "
            "`pip install biosim-lab[segmentation]`, or use backend='classical'."
        )
    model = stardist.StarDist2D.from_pretrained(model_name)
    labels, _ = model.predict_instances(csbdeep.normalize(np.asarray(image)))
    labels = _relabel_sequential(np.asarray(labels, dtype=np.int32))
    return SegmentationResult(
        labels=labels,
        properties=_measure(labels, np.asarray(image, dtype=float)),
        backend="stardist",
        parameters={"model_name": model_name},
    )


def segment(
    image: np.ndarray, backend: SegmentationBackend = "classical", **kwargs: Any
) -> SegmentationResult:
    """Dispatch to the requested segmentation back-end."""
    dispatch: dict[str, Callable[..., SegmentationResult]] = {
        "classical": segment_classical,
        "cellpose": segment_cellpose,
        "stardist": segment_stardist,
    }
    if backend not in dispatch:
        raise ValueError(f"unknown backend {backend!r}; available: {sorted(dispatch)}")
    return dispatch[backend](image, **kwargs)


__all__ = [
    "SegmentationBackend",
    "SegmentationResult",
    "available_backends",
    "segment",
    "segment_classical",
    "segment_cellpose",
    "segment_stardist",
]
