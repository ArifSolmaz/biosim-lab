"""Guards for the hosted-deployment manifests.

These files are not read by Python, so nothing else in the test suite would ever
notice if they were malformed --- the failure shows up minutes later in a
hosting provider's build log, which is the worst place to discover it.

The specific accident this exists to prevent: **Streamlit Community Cloud's
``packages.txt`` parser has no comment support.** It splits the file on
whitespace and hands every token to ``apt-get install``, so a single explanatory
comment turns into a wall of

    E: Unable to locate package OpenGL,
    E: Unable to locate package needed
    E: Unable to locate package by

and the whole deployment fails. One package name per line, nothing else.
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


def test_manifests_exist():
    assert PACKAGES.exists(), "packages.txt is what installs the OpenGL/X11 libraries"
    assert REQUIREMENTS.exists(), "requirements.txt is what the host installs with pip"


def test_packages_txt_contains_only_bare_package_names():
    """No comments, no blank lines, no inline annotations. See the module docstring."""
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
    assert b"\r" not in PACKAGES.read_bytes()


def test_the_libraries_pyvista_and_gmsh_link_against_are_present():
    """Both are compiled against OpenGL and X11 even when rendering off-screen."""
    packages = set(PACKAGES.read_text(encoding="utf-8").split())
    for required in ("libgl1", "libglu1-mesa", "libxrender1", "libxext6"):
        assert required in packages, (
            f"{required} missing: PyVista/VTK and Gmsh fail to import without it"
        )
    assert "xvfb" in packages, "PyVista needs a virtual framebuffer on a headless host"
    assert "libgomp1" in packages, "scikit-image and SciPy need the OpenMP runtime"


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
    assert path.read_bytes().endswith(b"\n"), f"{path.name} must end with a newline"
