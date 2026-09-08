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


@pytest.mark.slow
def test_every_widget_option_is_reachable_without_crashing() -> None:
    """Exercise each widget's option space, not just the default position.

    Rendering a page with its default values proves almost nothing: the bugs
    live at the settings nobody clicked. Three shipped failures were found this
    way and none of them were reachable from the defaults --- a scalar-vs-array
    sampling bug on one inlet option, a channel height at its own slider minimum
    that no cell could fit through, and a linking radius at its own slider
    maximum that made the assignment ambiguous.

    Discrete widgets get every option; continuous ones get both ends, which is
    where the degenerate cases are. ``select_slider`` is included deliberately:
    it was missed on the first pass because it is not a ``slider``.
    """
    pages = ["SAW cell sorter", "Impedance (RTCA)", "Cell counter",
             "Cell tracker", "Material provenance"]

    def fresh(page: str) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=900)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        return at

    failures = []
    for page in pages:
        base = fresh(page)
        plan: list[tuple[str, int, str, object]] = []
        for i, w in enumerate(base.selectbox):
            plan += [("selectbox", i, w.label, v) for v in w.options]
        for i, w in enumerate(base.radio):
            if w.label != "Instrument":
                plan += [("radio", i, w.label, v) for v in w.options]
        for i, w in enumerate(base.select_slider):
            plan += [("select_slider", i, w.label, v) for v in w.options]
        for i, w in enumerate(base.slider):
            plan += [("slider", i, w.label, w.min), ("slider", i, w.label, w.max)]
        for i, w in enumerate(base.number_input):
            plan += [("number_input", i, w.label, w.min),
                     ("number_input", i, w.label, w.max)]
        for i, w in enumerate(base.checkbox):
            plan += [("checkbox", i, w.label, True), ("checkbox", i, w.label, False)]

        for kind, idx, label, value in plan:
            at = fresh(page)
            getattr(at, kind)[idx].set_value(value).run()
            if at.exception:
                failures.append(f"[{page}] {label}={value!r}: {at.exception[0].value}")

    assert not failures, "widget settings that crash:\n" + "\n".join(failures)


@pytest.mark.slow
def test_impossible_settings_are_explained_not_crashed() -> None:
    """A device that cannot exist must produce guidance, not a stack trace."""
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.sidebar.radio[0].set_value("Cell tracker").run()
    index = next(i for i, w in enumerate(at.slider) if "Search range" in w.label)
    at.slider[index].set_value(at.slider[index].max).run()

    assert not at.exception
    assert at.error, "no explanation was rendered"
    assert "cannot be simulated" in at.error[0].value
