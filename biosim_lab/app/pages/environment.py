"""The environment page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from biosim_lab import __version__
from biosim_lab.app.shared import (
    note,
)
from biosim_lab.registry import discovery_source


def page_environment() -> None:
    st.title("Environment")
    note(
        "What is installed here, what is not, and — the part that usually gets "
        "left out — which of the missing pieces <b>could not work in a hosted "
        "container even if they were installed</b>."
    )

    from biosim_lab.core.geometry import gmsh_available
    from biosim_lab.core.plugin import optional_import
    from biosim_lab.core.solver import solver_report
    from biosim_lab.instruments.cell_counter.segmentation import available_backends

    left, right = st.columns([1, 2])
    left.metric("biosim-lab version", __version__)
    right.caption(f"Instruments found via {discovery_source()}")

    # (module, what it unlocks, whether it can function in a headless container)
    components: list[tuple[str, str, str]] = [
        ("numpy", "core numerics", "yes"),
        ("scipy", "core numerics", "yes"),
        ("xarray", "gridded results", "yes"),
        ("pandas", "tables", "yes"),
        ("pydantic", "configuration validation", "yes"),
        ("pint", "unit checking", "yes"),
        ("skfem", "built-in Helmholtz / Stokes / electro solvers", "yes"),
        ("plotly", "every figure in this app", "yes"),
        ("skimage", "segmentation", "yes"),
        ("trackpy", "track linking", "yes"),
        ("streamlit", "this interface", "yes"),
        ("gmsh", "the PDMS-wall mesh template (the built-in single-material "
                  "Helmholtz solver does not consume it yet)", "yes"),
        ("meshio", "importing Gmsh meshes", "yes"),
        ("pyvista", "3-D field rendering (off-screen)", "yes"),
        ("imageio", "reading and writing image stacks", "yes"),
        ("matplotlib", "colour maps used by scikit-image", "yes"),
        ("h5py", "HDF5 back-end for NetCDF", "yes"),
        ("h5netcdf", "NetCDF result files", "yes"),
        ("netCDF4", "NetCDF result files (alternative back-end)", "yes"),
        ("pyarrow", "Parquet result files", "yes"),
        ("panel", "the desktop dashboards (this app replaces them)", "yes"),
        ("bokeh", "Panel's rendering back-end", "yes"),
        ("typer", "the command-line interface", "yes"),
        ("rich", "command-line formatting", "yes"),
        ("kaleido", "exporting figures as static PNG", "yes"),
        ("napari", "interactive image review",
         "no — needs Qt and a display, which a hosted container has neither of"),
        ("cellpose", "deep-learning segmentation",
         "only with a CPU-only PyTorch build; the default wheel pulls ~2.5 GB of CUDA"),
        ("stardist", "star-convex segmentation",
         "only with TensorFlow (~600 MB); heavy for a 1 GB tier"),
    ]

    rows = []
    for module, unlocks, hosted in components:
        installed = optional_import(module) is not None
        rows.append({
            "component": module,
            "status": "installed" if installed else "not installed",
            "usable here": hosted,
            "what it unlocks": unlocks,
        })

    ok, why = gmsh_available()
    rows.append({"component": "gmsh (runtime check)",
                 "status": "working" if ok else "not working",
                 "usable here": "yes", "what it unlocks": why})
    for name, (avail, detail) in available_backends().items():
        rows.append({"component": f"segmentation: {name}",
                     "status": "installed" if avail else "not installed",
                     "usable here": "yes" if name == "classical" else "see above",
                     "what it unlocks": detail})
    for row in solver_report():
        rows.append({
            "component": f"solver: {row['name']}",
            "status": "available" if row["available"] else "not available",
            "usable here": "yes" if row["name"].startswith("builtin")
            else "no — needs an external binary that is not installable on a "
                 "managed host",
            "what it unlocks": f"{row['kind']} — {row['detail']}",
        })

    table = pd.DataFrame(rows)
    missing_but_usable = table[
        (table["status"].isin(["not installed", "not working", "not available"]))
        & (table["usable here"] == "yes")
    ]
    cols = st.columns(3)
    cols[0].metric("Installed", int((table["status"].isin(
        ["installed", "working", "available"])).sum()))
    cols[1].metric("Missing but fixable", len(missing_but_usable))
    cols[2].metric("Cannot work here", int((table["usable here"] != "yes").sum()))

    if len(missing_but_usable) == 0:
        st.success(
            "Everything that can work in a hosted container is installed. The "
            "remaining rows are not oversights — they are things a managed host "
            "cannot provide.", icon="✅",
        )
    else:
        st.warning(
            "These are installable and are not present: "
            + ", ".join(missing_but_usable["component"])
            + ". Add them to `requirements.txt` and redeploy.",
            icon="⚠️",
        )

    st.dataframe(table, width="stretch", hide_index=True, height=560)

    st.subheader("Why some things cannot be installed here")
    st.markdown(
        """
Three of the optional components are not missing by accident, and adding them to
`requirements.txt` would not help:

- **Napari** needs Qt and a display server. It would install, report itself
  present, and still be unable to open a viewer. Use it locally instead —
  `pip install -e ".[imaging]"`, then `instrument.view_napari()`.
- **OpenFOAM** and **Elmer** are external binaries, not Python packages. They
  come from apt repositories a managed host does not carry. Their absence means
  acoustic streaming is not modelled and the analytic Rayleigh approximation is
  used instead, and the piezoelectric problem is replaced by the documented
  voltage calibration. Stages 1–3 are complete without them; the project ships
  `docker/Dockerfile.openfoam` and `docker/Dockerfile.elmer` for when you need
  them.
- **Cellpose** and **StarDist** are installable but pull a deep-learning
  runtime. The default PyTorch wheel bundles CUDA and is roughly 2.5 GB. To use
  Cellpose on a free tier you would need the CPU-only build:

  ```
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch
  cellpose>=3.0
  ```

  The classical watershed back-end needs none of this and is what every figure
  in this app uses.
"""
    )


