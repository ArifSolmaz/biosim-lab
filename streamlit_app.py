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
* Dependencies come from ``requirements.txt``. There is deliberately **no**
  ``packages.txt``: any such file makes the host run ``apt-get update``, and a
  single expired release file in its base image then fails the whole deploy —
  which leaves the previous process serving stale code rather than the app
  simply losing a feature. Nothing here needs system libraries. Gmsh cannot
  import without OpenGL, so meshing falls back to the structured template; that
  path is exercised in CI and by ``tests/test_deployment.py``.
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
from biosim_lab.app.shared import CSS, warn_if_stale  # noqa: E402

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

    # Before anything else on the page: a stale process invalidates every
    # number below it, so this must not be tucked away on one tab.
    warn_if_stale()

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
