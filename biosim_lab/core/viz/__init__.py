"""Visualisation layer shared by every instrument.

* :mod:`biosim_lab.core.viz.theme` — one palette, used by every figure.
* :mod:`biosim_lab.core.viz.curves` — Plotly figures (curves, heatmaps, histograms).
* :mod:`biosim_lab.core.viz.fields3d` — PyVista 3-D field rendering and animation.
* :mod:`biosim_lab.core.viz.dashboard` — Panel scaffolding.
* :mod:`biosim_lab.core.viz.napari_layers` — optional Napari layers for image data.
"""

from biosim_lab.core.viz.theme import PALETTE, color_for, plotly_layout

__all__ = ["PALETTE", "color_for", "plotly_layout"]
