"""Presentation helpers shared by every page.

Nothing physical lives here: formatting, download buttons and the one place
that decides how a model warning is rendered.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
import streamlit as st

from biosim_lab.core.plugin import ConfigurationError
from biosim_lab.core.viz.theme import PALETTE

UL_MIN = 1e-9 / 60.0  # m^3/s per uL/min

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
      [data-testid="stMetricValue"] {{ font-size: 1.9rem; }}
      .biosim-note {{
        font-size: 0.82rem; color: {PALETTE['text_secondary_light']};
        border-left: 3px solid {PALETTE['grid_light']}; padding: 0.1rem 0 0.1rem 0.7rem;
        margin: 0.4rem 0 0.9rem 0;
      }}
      .biosim-doi {{ font-size: 0.76rem; color: {PALETTE['text_secondary_light']}; }}
    </style>
    """


def note(text: str) -> None:
    """A small muted caption used to explain what a number means."""
    st.markdown(f"<div class='biosim-note'>{text}</div>", unsafe_allow_html=True)


def show_warnings(caught: list[dict[str, Any]]) -> None:
    """Surface the model's own warnings instead of swallowing them.

    Takes the *serialised* form the cached runners return --- ``st.cache_data``
    has to pickle its result, and a ``warnings.WarningMessage`` does not
    survive that, so each one is reduced to its category name and message.

    A model that has been pushed outside its validated range still returns
    numbers, and they still render as confident-looking metrics. This is the
    only thing standing between the user and believing them.
    """
    for w in caught:
        if w["category"] == "RegimeWarning":
            st.warning(f"**Model outside its validated range** — {w['message']}", icon="⚠️")
        elif w["category"] == "MissingBackendWarning":
            st.info(f"**Optional back-end missing** — {w['message']}", icon="ℹ️")


@contextmanager
def explained_settings() -> Iterator[None]:
    """Render a settings problem as guidance rather than as a stack trace.

    Some widget combinations describe a device that cannot exist --- a cell
    taller than the channel it must flow through, a linking radius so wide that
    the assignment is ambiguous. Those are the user's to fix, and the model says
    how in the exception text.

    Only :class:`ConfigurationError` is caught. A genuine bug still surfaces as
    a real traceback, because hiding one behind a friendly message is how a
    broken simulation gets published.
    """
    try:
        yield
    except ConfigurationError as exc:
        st.error(f"**These settings cannot be simulated** — {exc}", icon="🚫")
        st.stop()


def download_frame(df: pd.DataFrame, filename: str, label: str) -> None:
    """CSV download button. CSV rather than Parquet/NetCDF so the hosted app
    needs neither pyarrow nor an HDF5 stack."""
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
        mime="text/csv", width="content",
    )


__all__ = [
    "CSS",
    "PLOTLY_CONFIG",
    "UL_MIN",
    "download_frame",
    "explained_settings",
    "note",
    "show_warnings",
]
