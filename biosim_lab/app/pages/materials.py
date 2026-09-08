"""The materials page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from biosim_lab.app.shared import (
    download_frame,
    note,
)
from biosim_lab.core.materials import all_values, audit


def page_materials() -> None:
    st.title("Material provenance")
    note(
        "Every physical constant in this software carries either a published "
        "reference or an explicit label saying it was assumed, with the reason. "
        "There is no third category, and an automated test refuses to let anyone "
        "add one."
    )

    assumptions = audit()
    total = len(all_values())

    cols = st.columns(3)
    cols[0].metric("Values in the library", total)
    cols[1].metric("With a published DOI", total - len(assumptions))
    cols[2].metric("Flagged as assumptions", len(assumptions))

    st.subheader("Everything, with its source")
    everything = pd.DataFrame(
        [
            {
                "group": group, "material": key, "property": prop,
                "value": f"{v.magnitude:g} {v.unit}",
                "kind": "ASSUMPTION" if v.prov.is_assumption else "published",
                "source": (v.prov.assumption if v.prov.is_assumption
                           else f"doi:{v.prov.doi} — {v.prov.citation}"),
            }
            for group, key, prop, v in all_values()
        ]
    )
    only_assumptions = st.checkbox(
        "Show only the assumptions (the list a methods section should disclose)",
        value=False,
    )
    view = everything[everything["kind"] == "ASSUMPTION"] if only_assumptions else everything
    st.dataframe(view, width="stretch", hide_index=True, height=460)
    download_frame(everything, "biosim_lab_materials.csv", "Download the whole library (CSV)")

    st.subheader("The single biggest assumption")
    st.warning(
        "Nothing in this software predicts how loud the sound is for a given drive "
        "voltage — that needs a piezoelectric solve this project has not "
        "implemented. A straight-line calibration stands in for it: 15 Vpp gives "
        "0.45 MPa. It scales **every** acoustic force here. If you build the "
        "device, measure the pressure by tracking calibration beads and enter that "
        "number instead of a voltage.",
        icon="⚠️",
    )


