"""Mesh templates and their named boundary groups."""

import pytest

from biosim_lab.core.geometry import (
    WELL_PLATE_FORMATS,
    gmsh_available,
    idt_electrodes_2d,
    straight_channel_2d,
    well_plate,
)

# Declared here rather than imported from conftest: `tests` is not an importable
# package, and it must not need to be for the suite to run from any rootdir
# (the Docker image runs `pytest` from /app).
requires_gmsh = pytest.mark.skipif(
    not gmsh_available()[0], reason="gmsh not usable here (optional: biosim-lab[mesh])"
)


def test_structured_channel_has_the_four_walls():
    bundle = straight_channel_2d(300e-6, 50e-6, resolution=20)
    assert set(bundle.boundaries) == {"left", "right", "bottom", "top"}
    assert all(len(v) > 0 for v in bundle.boundaries.values())
    assert bundle.meta["backend"] == "structured"
    assert bundle.mesh.p[0].max() == pytest.approx(300e-6)
    assert bundle.mesh.p[1].max() == pytest.approx(50e-6)


def test_resolution_controls_element_count():
    coarse = straight_channel_2d(300e-6, 50e-6, resolution=10)
    fine = straight_channel_2d(300e-6, 50e-6, resolution=40)
    assert fine.mesh.t.shape[1] > 4 * coarse.mesh.t.shape[1]


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        straight_channel_2d(-1.0, 50e-6)


def test_pdms_wall_without_gmsh_fails_loudly():
    """A wall layer needs Gmsh; if it is missing the user must be told, not
    silently given a wall-free mesh."""
    ok, _ = gmsh_available()
    if ok:
        pytest.skip("gmsh is available here")
    with pytest.raises(RuntimeError, match="requires gmsh"):
        straight_channel_2d(300e-6, 50e-6, wall_thickness=40e-6)


@requires_gmsh
def test_gmsh_channel_with_a_pdms_wall_splits_the_subdomains():
    bundle = straight_channel_2d(300e-6, 50e-6, resolution=16, wall_thickness=40e-6)
    assert bundle.meta["backend"] == "gmsh"
    assert set(bundle.subdomains) == {"fluid", "pdms"}
    assert bundle.subdomains["fluid"].size > 0
    assert bundle.subdomains["pdms"].size > 0
    assert "outer" in bundle.boundaries


def test_idt_electrodes_alternate_and_cover_the_bottom():
    bundle = idt_electrodes_2d(100e-6, 3, gap=50e-6, domain_height=50e-6, resolution=8)
    for group in ("electrode_a", "electrode_b", "substrate", "top"):
        assert group in bundle.boundaries, group
    a = set(bundle.boundaries["electrode_a"].tolist())
    b = set(bundle.boundaries["electrode_b"].tolist())
    assert not (a & b), "the two terminals must not share facets"


def test_idt_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        idt_electrodes_2d(100e-6, 2, gap=-1.0, domain_height=50e-6)
    with pytest.raises(ValueError):
        idt_electrodes_2d(100e-6, 2, gap=50e-6, domain_height=50e-6,
                          finger_width_ratio=1.5)


def test_well_plate_pitch_follows_the_slas_standard():
    p96 = well_plate(96)
    assert p96["n_wells"] == 96
    assert p96["pitch_m"] == pytest.approx(9.0e-3)
    assert p96["labels"][0] == "A1" and p96["labels"][-1] == "H12"
    assert p96["centres_m"].shape == (96, 2)
    assert well_plate(384)["pitch_m"] == pytest.approx(4.5e-3)


def test_unsupported_plate_format_is_rejected():
    with pytest.raises(ValueError, match="unsupported plate format"):
        well_plate(42)
    assert set(WELL_PLATE_FORMATS) >= {6, 24, 96, 384}


@requires_gmsh
def test_gmsh_works_off_the_main_thread():
    """Gmsh must initialise inside a worker thread.

    ``gmsh.initialize()`` installs a SIGINT handler, which Python only allows on
    the main thread. Anything that runs user code on a worker — a Streamlit
    script thread, a Jupyter kernel, a Dask worker, a web request handler —
    would otherwise get ``signal only works in main thread of the main
    interpreter`` and see Gmsh as permanently unavailable.
    """
    import threading

    captured: dict[str, object] = {}

    def work() -> None:
        captured["available"] = gmsh_available()
        bundle = straight_channel_2d(
            300e-6, 50e-6, resolution=12, wall_thickness=40e-6
        )
        captured["subdomains"] = sorted(bundle.subdomains)
        captured["elements"] = int(bundle.mesh.t.shape[1])

    thread = threading.Thread(target=work)
    thread.start()
    thread.join(timeout=120)

    assert captured["available"][0] is True, captured["available"]
    assert captured["subdomains"] == ["fluid", "pdms"]
    assert captured["elements"] > 0
