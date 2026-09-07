"""Gmsh geometry templates with a dependency-free fallback.

Every generator returns a :class:`MeshBundle`: a ``skfem`` mesh plus the mapping
from *physical group name* to boundary facets, which is what
:class:`~biosim_lab.core.solver.BoundaryCondition` refers to.

Gmsh is the default mesher, but the templates that are simple tensor products
(a straight rectangular channel) also have an exact structured fallback, so the
core FEM path keeps working if Gmsh cannot start — for instance in a minimal
container without OpenGL.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from biosim_lab.core.plugin import MissingBackendWarning, optional_import


@dataclass
class MeshBundle:
    """A mesh plus its named boundary facet sets.

    Attributes
    ----------
    mesh:
        ``skfem.Mesh`` instance (``MeshTri`` in 2-D).
    boundaries:
        ``{group_name: facet_index_array}``.
    subdomains:
        ``{group_name: element_index_array}`` for multi-material meshes
        (e.g. ``"fluid"`` vs ``"pdms"``).
    meta:
        Geometry parameters that produced the mesh, for provenance.
    """

    mesh: Any
    boundaries: dict[str, np.ndarray] = field(default_factory=dict)
    subdomains: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def p(self) -> np.ndarray:
        """Vertex coordinates, shape ``(dim, n_vertices)``."""
        return self.mesh.p

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MeshBundle {type(self.mesh).__name__} "
            f"nv={self.mesh.p.shape[1]} nel={self.mesh.t.shape[1]} "
            f"boundaries={sorted(self.boundaries)}>"
        )


def gmsh_available() -> tuple[bool, str]:
    """Return ``(usable, reason)`` for the Gmsh Python API."""
    gmsh = optional_import("gmsh")
    if gmsh is None:
        return False, (
            "python module 'gmsh' not importable (pip install biosim-lab[mesh]); "
            "the structured-mesh fallback covers the straight-channel template"
        )
    try:
        gmsh.initialize()
        version = gmsh.option.getString("General.Version")
        gmsh.finalize()
    except Exception as exc:  # noqa: BLE001
        return False, f"gmsh failed to initialise: {exc}"
    return True, f"gmsh {version}"


@contextlib.contextmanager
def _gmsh_session(name: str, verbosity: int = 0):
    """Initialise/finalise Gmsh around a model, even if the body raises."""
    gmsh = optional_import("gmsh")
    if gmsh is None:
        raise RuntimeError("gmsh is not installed")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.model.add(name)
        yield gmsh
    finally:
        gmsh.finalize()


def _read_gmsh(path: Path) -> Any:
    """Load a ``.msh`` file into a skfem mesh."""
    from skfem.io.meshio import from_file  # imported late: pulls in meshio

    # skfem's signature is from_file(filename, out, **kwargs); `out` collects extra
    # tags (boundaries/subdomains) and may be None when we recover groups ourselves.
    return from_file(path, None)


def _facets_where(mesh: Any, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Boundary facets whose midpoint satisfies *predicate*."""
    facets = mesh.boundary_facets()
    mid = mesh.p[:, mesh.facets[:, facets]].mean(axis=1)
    return facets[predicate(mid)]


def _all_facets_where(mesh: Any, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """*Any* facet (interior included) whose midpoint satisfies *predicate*.

    Needed for material interfaces: once a PDMS wall surrounds the channel, the
    fluid/wall interface is an interior facet set, not a boundary.
    """
    mid = mesh.p[:, mesh.facets].mean(axis=1)
    return np.flatnonzero(predicate(mid))


def _elements_where(mesh: Any, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Element indices whose centroid satisfies *predicate*."""
    centroid = mesh.p[:, mesh.t].mean(axis=1)
    return np.flatnonzero(predicate(centroid))


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


def straight_channel_2d(
    width: float,
    height: float,
    *,
    resolution: int = 24,
    wall_thickness: float = 0.0,
    use_gmsh: bool | None = None,
    order: int = 1,
) -> MeshBundle:
    """Rectangular microchannel cross-section in the ``(x, y)`` plane.

    The channel occupies ``0 <= x <= width`` and ``0 <= y <= height``.  ``x`` is
    the acoustic (transverse) direction across which the standing wave forms;
    ``y`` is the channel depth.  Through-flow is along ``z`` (out of plane) and
    is handled analytically by :mod:`biosim_lab.instruments.saw_sorter.flow`.

    Parameters
    ----------
    width, height:
        Channel cross-section in metres, e.g. ``300e-6`` x ``50e-6``.
    resolution:
        Target number of elements across the *width*.
    wall_thickness:
        If > 0, a PDMS wall layer of this thickness is added on the two side
        walls and the top, meshed as the ``"pdms"`` subdomain.  Requires Gmsh.
    use_gmsh:
        ``None`` (default) picks Gmsh when available and a wall is requested or
        the aspect ratio is awkward, otherwise the structured fallback.
    order:
        1 for straight-sided triangles (only 1 is supported by the fallback).

    Boundary groups
    ---------------
    ``left``, ``right``, ``bottom`` (the SAW-coupling substrate side), ``top``,
    plus ``outer`` when a PDMS wall is present.
    """
    if width <= 0 or height <= 0:
        raise ValueError("channel width and height must be positive")

    want_gmsh = use_gmsh
    if want_gmsh is None:
        want_gmsh = wall_thickness > 0
    if want_gmsh:
        ok, why = gmsh_available()
        if not ok:
            if wall_thickness > 0:
                raise RuntimeError(
                    f"a PDMS wall layer requires gmsh, but {why}. Install it with "
                    "`pip install biosim-lab[mesh]`; a wall-free channel works without it."
                )
            want_gmsh = False

    if not want_gmsh:
        return _structured_channel(width, height, resolution)

    return _gmsh_channel(width, height, resolution, wall_thickness, order)


def _structured_channel(width: float, height: float, resolution: int) -> MeshBundle:
    """Exact tensor-product triangulation of the rectangle (no Gmsh needed)."""
    from skfem import MeshTri

    nx = max(int(resolution), 2)
    ny = max(int(round(resolution * height / width)), 2)
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, width, nx + 1), np.linspace(0.0, height, ny + 1)
    )
    tol = 1e-9 * max(width, height)
    boundaries = {
        "left": _facets_where(mesh, lambda m: m[0] < tol),
        "right": _facets_where(mesh, lambda m: m[0] > width - tol),
        "bottom": _facets_where(mesh, lambda m: m[1] < tol),
        "top": _facets_where(mesh, lambda m: m[1] > height - tol),
    }
    return MeshBundle(
        mesh=mesh,
        boundaries=boundaries,
        subdomains={"fluid": np.arange(mesh.t.shape[1])},
        meta={
            "template": "straight_channel_2d",
            "backend": "structured",
            "width_m": width,
            "height_m": height,
            "nx": nx,
            "ny": ny,
        },
    )


def _gmsh_channel(
    width: float, height: float, resolution: int, wall_thickness: float, order: int
) -> MeshBundle:
    """Gmsh version, optionally with a PDMS wall layer around the fluid."""
    lc = width / max(resolution, 2)
    with _gmsh_session("straight_channel") as gmsh:
        occ = gmsh.model.occ
        fluid = occ.addRectangle(0.0, 0.0, 0.0, width, height)
        if wall_thickness > 0:
            t = wall_thickness
            outer = occ.addRectangle(-t, 0.0, 0.0, width + 2 * t, height + t)
            # Fragment so the two surfaces share a conforming interface.
            parts, _ = occ.fragment([(2, outer)], [(2, fluid)])
            occ.synchronize()
            surfaces = [tag for dim, tag in parts if dim == 2]
            fluid_tag, wall_tags = _classify_surfaces(gmsh, surfaces, width, height)
            gmsh.model.addPhysicalGroup(2, [fluid_tag], name="fluid")
            gmsh.model.addPhysicalGroup(2, wall_tags, name="pdms")
        else:
            occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [fluid], name="fluid")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "channel.msh"
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(out))
            mesh = _read_gmsh(out)

    tol = 1e-6 * max(width, height)
    outer_half = width / 2 + wall_thickness
    # With a wall the channel edges become material interfaces (interior facets);
    # without one they are true domain boundaries.
    pick = _all_facets_where if wall_thickness > 0 else _facets_where
    boundaries = {
        "left": pick(mesh, lambda m: (np.abs(m[0]) < tol) & (m[1] <= height + tol)),
        "right": pick(mesh, lambda m: (np.abs(m[0] - width) < tol) & (m[1] <= height + tol)),
        "bottom": _facets_where(mesh, lambda m: np.abs(m[1]) < tol),
        "top": pick(
            mesh,
            lambda m: (np.abs(m[1] - height) < tol) & (m[0] >= -tol) & (m[0] <= width + tol),
        ),
    }
    if wall_thickness > 0:
        boundaries["outer"] = _facets_where(
            mesh,
            lambda m: (np.abs(np.abs(m[0] - width / 2) - outer_half) < tol)
            | (np.abs(m[1] - (height + wall_thickness)) < tol),
        )
        boundaries["channel_bottom"] = _facets_where(
            mesh, lambda m: (np.abs(m[1]) < tol) & (m[0] > tol) & (m[0] < width - tol)
        )
        subdomains = {
            "fluid": _elements_where(
                mesh,
                lambda c: (c[0] > 0) & (c[0] < width) & (c[1] > 0) & (c[1] < height),
            ),
            "pdms": _elements_where(
                mesh,
                lambda c: ~((c[0] > 0) & (c[0] < width) & (c[1] > 0) & (c[1] < height)),
            ),
        }
    else:
        subdomains = {"fluid": np.arange(mesh.t.shape[1])}
    boundaries = {k: v for k, v in boundaries.items() if v.size}
    return MeshBundle(
        mesh=mesh,
        boundaries=boundaries,
        subdomains=subdomains,
        meta={
            "template": "straight_channel_2d",
            "backend": "gmsh",
            "width_m": width,
            "height_m": height,
            "wall_thickness_m": wall_thickness,
            "lc_m": lc,
        },
    )


def _classify_surfaces(
    gmsh: Any, surfaces: list[int], width: float, height: float
) -> tuple[int, list[int]]:
    """Split fragmented surfaces into the fluid rectangle and the wall pieces."""
    # Gmsh pads OCC bounding boxes by Geometry.Tolerance (1e-7 by default), so the
    # comparison tolerance must be a fraction of the geometry, not machine epsilon.
    tol = 0.05 * min(width, height)
    fluid_tag, wall_tags = None, []
    for tag in surfaces:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.occ.getBoundingBox(2, tag)
        inside = (
            abs(xmin) < tol
            and abs(xmax - width) < tol
            and abs(ymin) < tol
            and abs(ymax - height) < tol
        )
        if inside and fluid_tag is None:
            fluid_tag = tag
        else:
            wall_tags.append(tag)
    if fluid_tag is None:  # pragma: no cover - geometry sanity guard
        raise RuntimeError("could not identify the fluid surface after fragmentation")
    return fluid_tag, wall_tags


def idt_electrodes_2d(
    period: float,
    n_pairs: int,
    *,
    gap: float,
    domain_height: float,
    finger_width_ratio: float = 0.5,
    resolution: int = 20,
) -> MeshBundle:
    """Cross-section of an interdigital transducer / electrode array.

    Used by ``impedance_rtca`` for the electro-quasistatic problem above a
    coplanar gold IDT, and by ``saw_sorter`` documentation figures.

    Parameters
    ----------
    period:
        Spatial period ``p`` of the electrode pattern [m]. For a SAW IDT the
        acoustic wavelength equals ``p``.
    n_pairs:
        Number of finger pairs.
    gap:
        Height of the electrolyte/fluid domain above the electrode plane [m].
    domain_height:
        Total simulated height [m]; must exceed *gap*.
    finger_width_ratio:
        Finger width as a fraction of the half-period (0.5 = equal metal/space).

    Boundary groups
    ---------------
    ``electrode_a``, ``electrode_b`` (alternating fingers, the two terminals),
    ``substrate`` (insulating gaps on the electrode plane), ``top``,
    ``left``, ``right``.
    """
    if domain_height <= 0 or gap <= 0:
        raise ValueError("gap and domain_height must be positive")
    if not 0 < finger_width_ratio < 1:
        raise ValueError("finger_width_ratio must lie in (0, 1)")

    from skfem import MeshTri

    total_width = period * n_pairs
    nx = max(int(resolution * n_pairs * 2), 8)
    ny = max(int(resolution), 4)
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, total_width, nx + 1), np.linspace(0.0, domain_height, ny + 1)
    )
    tol = 1e-9 * max(total_width, domain_height)
    half = period / 2.0
    finger = half * finger_width_ratio

    def _on_bottom(m: np.ndarray) -> np.ndarray:
        return m[1] < tol

    def _phase(m: np.ndarray) -> np.ndarray:
        return np.mod(m[0], period)

    boundaries = {
        "electrode_a": _facets_where(mesh, lambda m: _on_bottom(m) & (_phase(m) < finger)),
        "electrode_b": _facets_where(
            mesh, lambda m: _on_bottom(m) & (_phase(m) >= half) & (_phase(m) < half + finger)
        ),
        "top": _facets_where(mesh, lambda m: m[1] > domain_height - tol),
        "left": _facets_where(mesh, lambda m: m[0] < tol),
        "right": _facets_where(mesh, lambda m: m[0] > total_width - tol),
    }
    covered = np.concatenate([boundaries["electrode_a"], boundaries["electrode_b"]])
    bottom_all = _facets_where(mesh, _on_bottom)
    boundaries["substrate"] = np.setdiff1d(bottom_all, covered)

    return MeshBundle(
        mesh=mesh,
        boundaries={k: v for k, v in boundaries.items() if v.size},
        subdomains={"electrolyte": np.arange(mesh.t.shape[1])},
        meta={
            "template": "idt_electrodes_2d",
            "period_m": period,
            "n_pairs": n_pairs,
            "gap_m": gap,
            "domain_height_m": domain_height,
            "finger_width_ratio": finger_width_ratio,
        },
    )


#: SBS-standard microplate geometry. Well pitch is the only dimension that is
#: standardised across vendors (ANSI/SLAS 4-2004); diameters vary by supplier.
WELL_PLATE_FORMATS: dict[int, dict[str, Any]] = {
    6: {"rows": 2, "cols": 3, "pitch_m": 39.12e-3, "well_diameter_m": 34.8e-3},
    24: {"rows": 4, "cols": 6, "pitch_m": 19.30e-3, "well_diameter_m": 15.6e-3},
    96: {"rows": 8, "cols": 12, "pitch_m": 9.00e-3, "well_diameter_m": 6.4e-3},
    384: {"rows": 16, "cols": 24, "pitch_m": 4.50e-3, "well_diameter_m": 3.3e-3},
    1536: {"rows": 32, "cols": 48, "pitch_m": 2.25e-3, "well_diameter_m": 1.5e-3},
}


def well_plate(n_wells: int = 96) -> dict[str, Any]:
    """Return the layout of an SBS-format microplate.

    Well pitch follows ANSI/SLAS 4-2004 (9 mm for 96, 4.5 mm for 384).  Well
    diameters are vendor dependent; the values here are typical flat-bottom
    numbers and are marked as such in the returned ``notes`` field.

    Returns
    -------
    dict
        ``rows``, ``cols``, ``pitch_m``, ``well_diameter_m``, ``labels``
        (``["A1", "A2", ...]``) and ``centres_m`` as an ``(n, 2)`` array.
    """
    if n_wells not in WELL_PLATE_FORMATS:
        raise ValueError(
            f"unsupported plate format {n_wells}; available: {sorted(WELL_PLATE_FORMATS)}"
        )
    spec = dict(WELL_PLATE_FORMATS[n_wells])
    rows, cols, pitch = spec["rows"], spec["cols"], spec["pitch_m"]
    labels, centres = [], []
    for r in range(rows):
        for c in range(cols):
            labels.append(f"{chr(ord('A') + r)}{c + 1}")
            centres.append((c * pitch, r * pitch))
    spec.update(
        n_wells=n_wells,
        labels=labels,
        centres_m=np.asarray(centres),
        notes="Pitch follows ANSI/SLAS 4-2004. Well diameter is a typical flat-bottom "
        "value and is vendor dependent (ASSUMPTION).",
    )
    return spec


__all__ = [
    "MeshBundle",
    "gmsh_available",
    "straight_channel_2d",
    "idt_electrodes_2d",
    "well_plate",
    "WELL_PLATE_FORMATS",
    "MissingBackendWarning",
]
