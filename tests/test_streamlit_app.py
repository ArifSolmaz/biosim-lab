"""Every page of the front end must actually execute.

Importing a page module only proves it parses. These tests run the real script
through Streamlit's own harness and select each page in turn, which is the only
way a broken widget call, a renamed metric key or a missing import gets caught.

This matters most after the app was split from one 1500-line file into a module
per page: a name that silently moved out of scope produces a page that raises at
render time and imports perfectly.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute: AppTest resolves a relative path against the working directory,
# which pytest does not guarantee.
APP = str(pathlib.Path(__file__).resolve().parent.parent / "streamlit_app.py")

PAGES = [
    "Overview",
    "SAW cell sorter",
    "Impedance (RTCA)",
    "Cell counter",
    "Cell tracker",
    "Material provenance",
    "Environment",
]


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"the app failed to start: {at.exception}"
    return at


def test_the_app_starts(app: AppTest) -> None:
    assert app.sidebar.radio, "the instrument selector is missing"
    assert list(app.sidebar.radio[0].options) == PAGES


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders_without_raising(page: str) -> None:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"page {page!r} raised: {at.exception}"


@pytest.mark.slow
def test_the_sorter_page_reports_real_numbers(app: AppTest) -> None:
    """A page that renders but shows nothing is still broken."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("SAW cell sorter").run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Recovery", "Purity", "Enrichment"} <= labels, f"got metrics: {sorted(labels)}"
