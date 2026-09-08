"""Guards for the hosted-deployment manifests.

These files are not read by Python, so nothing else in the test suite would ever
notice if they were malformed --- the failure shows up minutes later in a
hosting provider's build log, which is the worst place to discover it.

Two accidents, both of which have actually happened here.

**A ``packages.txt`` at all is a liability.** Its presence makes the host run
``apt-get update`` before anything else, and the base image ships repositories
outside this project's control. One expired ``Release`` file --- Debian
bullseye-security, expired by twelve hours --- made ``apt-get update`` return
non-zero, which failed the entire deploy. The consequence is worse than losing a
feature: the new instance never starts, so the *previous* process keeps serving,
executing old bytecode against the newly pulled source. A fix then appears to
change nothing. Nothing in this app needs a system library, so the file is gone
and must stay gone.

**Its parser has no comment support**, which is why the file existed in a broken
state before that. It splits on whitespace and hands every token to ``apt-get
install``, so one explanatory comment became a wall of ``E: Unable to locate
package OpenGL,``. That check is kept below in case the file is ever
reintroduced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ROOT / "packages.txt"
REQUIREMENTS = ROOT / "requirements.txt"

#: Debian package names: lower-case alphanumerics plus ``+ - .``
DEBIAN_NAME = re.compile(r"^[a-z0-9][a-z0-9+.\-]*$")


def test_there_is_no_packages_txt():
    """Its mere presence makes the host run apt-get update, which can fail.

    Not a style preference: an expired release file in the base image took the
    whole deployment down, and nothing here needs a system library.
    """
    assert not PACKAGES.exists(), (
        "packages.txt is back. Its presence triggers `apt-get update` on the "
        "host, and a single expired repository in the base image then fails the "
        "deploy and leaves the previous process serving stale code. Add a system "
        "dependency only if something genuinely cannot work without it."
    )


def test_manifests_exist():
    assert REQUIREMENTS.exists(), "requirements.txt is what the host installs with pip"


def test_packages_txt_contains_only_bare_package_names():
    """No comments, no blank lines, no inline annotations. See the module docstring.

    Skipped while the file is absent, which is the intended state; it exists so
    that reintroducing the file cannot reintroduce the parsing accident too.
    """
    if not PACKAGES.exists():
        pytest.skip("no packages.txt, which is the intended state")
    lines = PACKAGES.read_text(encoding="utf-8").splitlines()
    assert lines, "packages.txt must not be empty"
    for number, line in enumerate(lines, start=1):
        assert line == line.strip(), (
            f"packages.txt line {number}: leading or trailing whitespace"
        )
        assert line, (
            f"packages.txt line {number}: blank line — apt is handed the raw file"
        )
        assert not line.startswith("#"), (
            f"packages.txt line {number}: comments are NOT supported. Streamlit "
            "Cloud passes every whitespace-separated token to apt-get, so this "
            "line would be read as several package names."
        )
        assert DEBIAN_NAME.match(line), (
            f"packages.txt line {number}: {line!r} is not a bare Debian package name"
        )


def test_packages_txt_has_no_carriage_returns():
    """A CRLF file makes every package name end in \\r, and apt finds none of them."""
    if not PACKAGES.exists():
        pytest.skip("no packages.txt, which is the intended state")
    assert b"\r" not in PACKAGES.read_bytes()


def test_nothing_in_requirements_needs_a_system_library_to_import():
    """The deploy has no apt step, so every dependency must import unaided.

    gmsh is the deliberate exception: it links against OpenGL and cannot import
    on a bare host. That is allowed *because* the code treats it as optional and
    falls back to a structured mesh --- which this asserts rather than assumes,
    since an unhandled failure here takes the whole app down.
    """
    from biosim_lab.core.geometry import gmsh_available
    from biosim_lab.core.plugin import optional_import

    for module in ("numpy", "scipy", "pandas", "xarray", "skfem", "plotly",
                   "streamlit", "skimage", "trackpy", "pint", "pydantic"):
        assert optional_import(module) is not None, (
            f"{module} is in requirements.txt but does not import"
        )

    # Whatever gmsh does here, asking must not raise.
    available, message = gmsh_available()
    assert isinstance(available, bool)
    assert message, "gmsh_available must explain itself either way"

def _requirement_names(text: str) -> set[str]:
    names = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower())
    return names


def test_requirements_cover_what_the_app_imports():
    names = _requirement_names(REQUIREMENTS.read_text(encoding="utf-8"))
    for required in (
        "streamlit", "plotly", "numpy", "scipy", "pandas", "xarray",
        "pint", "pydantic", "pyyaml", "scikit-fem", "scikit-image", "trackpy",
    ):
        assert required in names, f"{required} missing from requirements.txt"


def test_heavy_optional_backends_stay_commented_out():
    """Torch and TensorFlow must not be uncommented by accident.

    The default PyTorch wheel bundles CUDA at roughly 2.5 GB, which does not fit
    a free hosting tier. The CPU-only opt-in is documented at the bottom of
    requirements.txt and has to be a deliberate act.
    """
    names = _requirement_names(REQUIREMENTS.read_text(encoding="utf-8"))
    for heavy in ("torch", "tensorflow", "stardist"):
        assert heavy not in names, (
            f"{heavy} is uncommented in requirements.txt; that will exceed the "
            "free-tier build limits"
        )


def test_napari_is_not_requested_for_the_hosted_app():
    """It would install, report itself present, and still be unusable without a display."""
    names = _requirement_names(REQUIREMENTS.read_text(encoding="utf-8"))
    assert "napari" not in names


@pytest.mark.parametrize("path", [PACKAGES, REQUIREMENTS])
def test_manifests_end_with_a_newline(path: Path):
    """A missing final newline silently drops the last entry in some parsers."""
    if not path.exists():
        pytest.skip(f"{path.name} is absent, which is the intended state")
    assert path.read_bytes().endswith(b"\n"), f"{path.name} must end with a newline"
