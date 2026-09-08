"""biosim-lab --- Streamlit front end.

A browser interface to the same simulation core the CLI and the Panel dashboards
use. Nothing physical lives here: every number on screen comes from
``biosim_lab``, and every figure is built by
:mod:`biosim_lab.core.viz.curves`, which returns plain Plotly figures and is
therefore framework-agnostic.

This file is the entry point and the router only. Each page lives in
:mod:`biosim_lab.app.pages`, the cached simulation wrappers in
:mod:`biosim_lab.app.runners`, and the presentation helpers in
:mod:`biosim_lab.app.shared`.

Deployment notes (see docs/USER_MANUAL.md for the full walkthrough)
------------------------------------------------------------------
* Entry point for Streamlit Community Cloud: this file, at the repository root.
* Dependencies come from ``requirements.txt``, which is deliberately *smaller*
  than the ones in ``pyproject.toml``: PyVista/VTK, Gmsh, Napari, Panel and the
  NetCDF back-ends are all left out. The core degrades to a structured mesh, and
  results download as CSV instead of NetCDF. That keeps the image inside the
  memory budget of a free hosting tier.
* Because the package is not ``pip install``-ed there, entry-point discovery
  finds nothing; :mod:`biosim_lab.registry` falls back to importing the built-in
  instruments directly. The Environment page reports which route was taken.
* Every simulation is wrapped in ``st.cache_data`` keyed on its parameters, so
  dragging a slider back to a previous value is instant and the host is not
  asked to recompute the same thing twice.
"""

from __future__ import annotations

import streamlit as st

# set_page_config must be the first Streamlit call in the process, so it runs
# before the page modules are imported.
st.set_page_config(
    page_title="biosim-lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from biosim_lab import __version__  # noqa: E402
from biosim_lab.app.pages import (  # noqa: E402
    counter,
    environment,
    materials,
    overview,
    rtca,
    sorter,
    tracker,
)
from biosim_lab.app.shared import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

PAGES = {
    "Overview": overview.page_overview,
    "SAW cell sorter": sorter.page_sorter,
    "Impedance (RTCA)": rtca.page_rtca,
    "Cell counter": counter.page_counter,
    "Cell tracker": tracker.page_tracker,
    "Material provenance": materials.page_materials,
    "Environment": environment.page_environment,
}


def main() -> None:
    with st.sidebar:
        st.markdown("### 🔬 biosim-lab")
        choice = st.radio("Instrument", list(PAGES), label_visibility="collapsed")
        st.divider()

    PAGES[choice]()

    with st.sidebar:
        st.divider()
        st.caption(
            f"biosim-lab {__version__} · MIT licence\n\n"
            "Every number on screen is computed live from the same code the "
            "command line runs. Nothing is pre-baked."
        )


if __name__ == "__main__":
    main()
