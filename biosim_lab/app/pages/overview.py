"""The overview page."""

from __future__ import annotations

import streamlit as st


def page_overview() -> None:
    st.title("biosim-lab")
    st.markdown(
        "#### Open-source virtual laboratory instruments — "
        "simulations of what commercial bio-instruments measure"
    )
    st.markdown(
        """
Four instruments share one core. Pick one from the sidebar; every control
re-runs the real simulation, not a lookup table.

| Instrument | Commercial equivalent | What it answers |
|---|---|---|
| **SAW cell sorter** | acoustic separators | can I pull tumour cells from blood, how pure? |
| **Impedance (RTCA)** | xCELLigence | how fast are they growing, what dose kills half? |
| **Cell counter** | Countess, Cellometer | how many cells per mL, how many alive? |
| **Cell tracker** | Incucyte | how fast do they crawl, and in a direction or not? |
"""
    )

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Read this before trusting a number")
        st.markdown(
            """
- **The model tells you when it is out of its depth.** Orange warning banners
  are the simulation flagging that an assumption behind it has been stretched.
  They are not errors and they are not decoration — read them.
- **Every physical constant has a source.** The *Material provenance* page lists
  all 65, split into the 27 with a published DOI and the 38 that are explicit
  assumptions, each with the reason.
- **The fast model is the optimistic one.** The analytic sorter mode applies the
  sound strength at the chip surface to cells at every height. The real field
  weakens upwards. Switch the sorter to *FEM* mode to see the honest version.
"""
        )
    with right:
        st.subheader("What this is not")
        st.markdown(
            """
- It is **not** a replacement for an experiment. Its value is telling you in
  seconds that a design will not work, before you fabricate the chip.
- It does **not** predict how loud the sound is for a given drive voltage.
  That needs a piezoelectric solve this project has not implemented; there is a
  documented linear calibration standing in for it.
- Some effects are deliberately left out — acoustic streaming, cell–cell
  interaction, cell deformability. The full list is in the manual.
"""
        )

    st.divider()
    st.markdown(
        "<span class='biosim-doi'>Gor'kov radiation force: doi:10.1039/c2lc21068a"
        " &nbsp;·&nbsp; SSAW node spacing: doi:10.1039/b910595f"
        " &nbsp;·&nbsp; impedance model: doi:10.1073/pnas.88.17.7896"
        " &nbsp;·&nbsp; particle linking: doi:10.1006/jcis.1996.0217</span>",
        unsafe_allow_html=True,
    )


