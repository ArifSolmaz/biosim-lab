"""Optional Napari layers for image-based instruments.

Napari pulls in Qt, which is heavy and needs a display, so it is an **optional**
dependency: nothing in the core imports it at module load, and the helpers here
degrade to a clear error if it is absent.  Use
``pip install biosim-lab[imaging]`` to enable them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.plugin import optional_import


def napari_available() -> tuple[bool, str]:
    """``(available, reason)`` for the Napari viewer."""
    napari = optional_import("napari")
    if napari is None:
        return False, "napari is not installed (pip install biosim-lab[imaging])"
    return True, f"napari {getattr(napari, '__version__', 'unknown')}"


def _require_napari() -> Any:
    napari = optional_import("napari")
    if napari is None:
        raise RuntimeError(
            "napari is required for interactive image review; install it with "
            "`pip install biosim-lab[imaging]`. The instruments themselves run "
            "and export results without it."
        )
    return napari


def view_segmentation(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    points: pd.DataFrame | None = None,
    name: str = "biosim-lab",
    show: bool = True,
) -> Any:
    """Open a Napari viewer with the image, its label mask and detected centroids.

    Parameters
    ----------
    image:
        ``(y, x)`` or ``(t, y, x)`` array.
    labels:
        Integer segmentation mask of the same shape.
    points:
        Table with ``y``/``x`` (and optionally ``frame``) columns.
    """
    napari = _require_napari()
    viewer = napari.Viewer(title=name, show=show)
    viewer.add_image(np.asarray(image), name="image", colormap="gray")
    if labels is not None:
        viewer.add_labels(np.asarray(labels).astype(int), name="segmentation")
    if points is not None and len(points):
        cols = [c for c in ("frame", "y", "x") if c in points.columns]
        viewer.add_points(points[cols].to_numpy(), name="centroids", size=8,
                          face_color="#eb6834")
    return viewer


def view_tracks(
    image: np.ndarray, tracks: pd.DataFrame, *, name: str = "biosim-lab tracks",
    show: bool = True,
) -> Any:
    """Open a Napari viewer with a tracks layer built from a trackpy table.

    *tracks* must have ``particle``, ``frame``, ``y``, ``x`` columns — the
    column names ``trackpy.link`` produces.
    """
    napari = _require_napari()
    required = {"particle", "frame", "y", "x"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks table is missing columns: {sorted(missing)}")
    viewer = napari.Viewer(title=name, show=show)
    viewer.add_image(np.asarray(image), name="image", colormap="gray")
    viewer.add_tracks(
        tracks[["particle", "frame", "y", "x"]].to_numpy(), name="tracks"
    )
    return viewer


__all__ = ["napari_available", "view_segmentation", "view_tracks"]
