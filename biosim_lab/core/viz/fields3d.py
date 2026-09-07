"""PyVista rendering of 2-D/3-D fields and trajectory animations.

PyVista needs a rendering backend.  In containers and CI there is no display, so
every entry point here calls :func:`configure_offscreen` first, which switches
VTK to off-screen software rendering.  If even that is unavailable the functions
raise a clear message rather than crashing the interpreter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from biosim_lab.core.plugin import optional_import
from biosim_lab.core.viz.theme import SEQUENTIAL_BLUE, color_for


def configure_offscreen() -> Any:
    """Import PyVista in headless mode; returns the module or ``None``."""
    pv = optional_import("pyvista", purpose="3-D field rendering needs PyVista")
    if pv is None:
        return None
    try:
        pv.OFF_SCREEN = True
        pv.global_theme.notebook = False
    except Exception:  # noqa: BLE001 - theme tweaks must never be fatal
        pass
    # On a headless Linux CI runner VTK also needs a virtual framebuffer; call
    # pyvista.start_xvfb() there. It is not invoked automatically because it
    # spawns a process, which is the wrong default for a library import.
    return pv


def _require_pyvista() -> Any:
    pv = configure_offscreen()
    if pv is None:
        raise RuntimeError(
            "PyVista is required for 3-D rendering. Install it with "
            "`pip install biosim-lab[viz]`, or use the Plotly figures in "
            "biosim_lab.core.viz.curves instead."
        )
    return pv


def grid_to_structured(field: xr.Dataset, variable: str, *, z_scale: float = 0.0) -> Any:
    """Convert a gridded ``(x, y)`` dataset into a PyVista ``StructuredGrid``.

    Parameters
    ----------
    z_scale:
        When non-zero the field value is used as the ``z`` coordinate as well,
        producing a relief surface — useful for pressure fields where the
        shape of the standing wave is the point.
    """
    pv = _require_pyvista()
    xs = field["x"].values
    ys = field["y"].values
    values = field[variable].values
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = values * z_scale if z_scale else np.zeros_like(gx)
    grid = pv.StructuredGrid(gx, gy, gz)
    grid[variable] = values.ravel(order="F")
    return grid


def render_field(
    field: xr.Dataset,
    variable: str = "p_abs",
    *,
    output: str | Path | None = None,
    z_scale: float = 0.0,
    window_size: tuple[int, int] = (1100, 500),
    title: str | None = None,
) -> Path | Any:
    """Render a 2-D field as a coloured surface; save a PNG when *output* is given."""
    pv = _require_pyvista()
    if variable not in field and {"p_real", "p_imag"} <= set(field.data_vars):
        field = field.assign(
            p_abs=(("x", "y"), np.hypot(field["p_real"].values, field["p_imag"].values))
        )
    grid = grid_to_structured(field, variable, z_scale=z_scale)
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.add_mesh(
        grid,
        scalars=variable,
        cmap=list(SEQUENTIAL_BLUE),
        show_edges=False,
        scalar_bar_args={"title": title or variable, "vertical": True},
    )
    plotter.view_xy()
    plotter.set_background("#fcfcfb")
    if output is None:
        return plotter
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output))
    plotter.close()
    return output


def render_force_vectors(
    force_field: xr.Dataset,
    *,
    output: str | Path | None = None,
    stride: int = 8,
    window_size: tuple[int, int] = (1100, 500),
) -> Path | Any:
    """Quiver plot of the Gor'kov force field over the pressure magnitude."""
    pv = _require_pyvista()
    xs = force_field["x"].values[::stride]
    ys = force_field["y"].values[::stride]
    fx = force_field["F_x"].values[::stride, ::stride]
    fy = force_field["F_y"].values[::stride, ::stride]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    vectors = np.column_stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)])
    magnitude = np.linalg.norm(vectors, axis=1)
    scale = (xs[1] - xs[0]) * 1.5 / max(magnitude.max(), 1e-30) if len(xs) > 1 else 1.0

    cloud = pv.PolyData(points)
    cloud["force"] = vectors * scale
    cloud["magnitude"] = magnitude
    arrows = cloud.glyph(orient="force", scale="force", factor=1.0)

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.add_mesh(grid_to_structured(force_field, "p_abs"), scalars="p_abs",
                     cmap=list(SEQUENTIAL_BLUE), opacity=0.85)
    plotter.add_mesh(arrows, color="#0b0b0b")
    plotter.view_xy()
    plotter.set_background("#fcfcfb")
    if output is None:
        return plotter
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output))
    plotter.close()
    return output


def animate_trajectories(
    trajectories: xr.Dataset,
    output: str | Path,
    *,
    channel_width: float,
    channel_length: float,
    fps: int = 20,
    max_particles: int = 300,
    window_size: tuple[int, int] = (1100, 420),
) -> Path:
    """Write an MP4 (or GIF) of cells travelling down the channel.

    The container is chosen from the file suffix: ``.mp4`` uses PyVista's movie
    writer (needs ``imageio-ffmpeg``), ``.gif`` uses the GIF writer, which has
    no extra dependency.
    """
    pv = _require_pyvista()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pos = trajectories["position"].values
    labels = np.asarray(trajectories["label"].values, dtype=str)
    radius = np.asarray(trajectories["radius"].values, dtype=float)
    if pos.shape[0] > max_particles:
        keep = np.linspace(0, pos.shape[0] - 1, max_particles).astype(int)
        pos, labels, radius = pos[keep], labels[keep], radius[keep]

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    if output.suffix.lower() == ".gif":
        plotter.open_gif(str(output), fps=fps)
    else:
        plotter.open_movie(str(output), framerate=fps)

    unique = sorted(set(labels))
    clouds: dict[str, Any] = {}
    for label in unique:
        sel = labels == label
        cloud = pv.PolyData(np.column_stack([pos[sel, 0, 2], pos[sel, 0, 0],
                                             pos[sel, 0, 1]]))
        cloud["radius"] = radius[sel]
        glyphs = cloud.glyph(scale="radius", geom=pv.Sphere(radius=1.0), factor=6.0)
        plotter.add_mesh(glyphs, color=color_for(label), label=label)
        clouds[label] = (cloud, sel)

    plotter.add_legend(bcolor="#fcfcfb", face="circle")
    plotter.set_background("#fcfcfb")
    plotter.camera_position = "xy"
    plotter.reset_camera()

    n_frames = pos.shape[1]
    for frame in range(n_frames):
        for cloud, sel in clouds.values():
            cloud.points = np.column_stack(
                [pos[sel, frame, 2], pos[sel, frame, 0], pos[sel, frame, 1]]
            )
        plotter.write_frame()
    plotter.close()
    return output


__all__ = [
    "configure_offscreen",
    "grid_to_structured",
    "render_field",
    "render_force_vectors",
    "animate_trajectories",
]
