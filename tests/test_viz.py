"""Figure construction and the colour policy.

The figures are not compared pixel by pixel; what is checked is the policy that
makes them readable: stable colour assignment, a legend for multi-series
figures, one y-axis, and a neutral midpoint for diverging scales.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from biosim_lab.core.viz import curves, theme


def test_colour_follows_the_entity_not_its_rank():
    """Dropping a series must not repaint the survivors."""
    first = theme.color_for("mcf7")
    assert theme.color_for("rbc") != first
    assert theme.color_for("mcf7") == first
    # An unknown label gets a stable slot too.
    unknown = theme.color_for("my_line")
    assert theme.color_for("my_line") == unknown


def test_categorical_palette_has_no_duplicate_hues():
    assert len(set(theme.CATEGORICAL_LIGHT)) == len(theme.CATEGORICAL_LIGHT)
    assert len(theme.CATEGORICAL_DARK) == len(theme.CATEGORICAL_LIGHT)


def test_sequential_ramp_is_monotonically_darker():
    def luminance(hex_color: str) -> float:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lum = [luminance(c) for c in theme.SEQUENTIAL_BLUE]
    assert all(b < a for a, b in zip(lum, lum[1:])), "a sequential ramp must be monotone"


def test_diverging_scale_has_a_neutral_midpoint():
    scale = theme.diverging_colorscale()
    assert scale[1][0] == 0.5
    mid = scale[1][1]
    r, g, b = (int(mid[i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) < 12, "the midpoint must read as neutral, not hued"


def test_layout_never_creates_a_second_y_axis():
    layout = theme.plotly_layout("t", xaxis_title="x", yaxis_title="y")
    assert "yaxis2" not in layout
    assert layout["yaxis"]["title"]["text"] == "y"


def _cells():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "label": ["mcf7"] * 30 + ["rbc"] * 30,
            "x_outlet_m": np.r_[rng.normal(150e-6, 8e-6, 30), rng.normal(40e-6, 20e-6, 30)],
            "radius_m": np.r_[rng.normal(9e-6, 1e-6, 30), rng.normal(2.8e-6, 0.2e-6, 30)],
            "x_initial_m": rng.uniform(0, 300e-6, 60),
        }
    )


def test_outlet_histogram_has_one_trace_per_population_and_a_legend():
    fig = curves.outlet_histogram_figure(
        _cells(), channel_width=300e-6, collection_bounds=(100e-6, 200e-6)
    )
    assert len(fig.data) == 2
    assert fig.layout.showlegend
    assert {t.name for t in fig.data} == {"mcf7", "rbc"}


def test_size_distribution_figure_builds():
    fig = curves.size_distribution_figure(_cells())
    assert len(fig.data) == 2


def test_force_profile_figure_labels_each_series():
    x = np.linspace(0, 300e-6, 50)
    fig = curves.force_profile_figure(
        x, {"mcf7": np.sin(x * 1e4), "rbc": 0.1 * np.sin(x * 1e4)},
        node_positions=[150e-6],
    )
    assert {t.name for t in fig.data} == {"mcf7", "rbc"}


def test_field_heatmap_uses_a_neutral_midpoint_when_diverging():
    grid = xr.Dataset(
        {"F_x": (("x", "y"), np.linspace(-1, 1, 20).reshape(5, 4))},
        coords={"x": np.arange(5.0), "y": np.arange(4.0)},
    )
    fig = curves.field_heatmap_figure(grid, "F_x", diverging=True)
    assert fig.data[0].zmid == 0
    assert fig.data[0].zmin == -fig.data[0].zmax


def test_field_heatmap_derives_p_abs_from_the_complex_parts():
    grid = xr.Dataset(
        {
            "p_real": (("x", "y"), np.ones((3, 2))),
            "p_imag": (("x", "y"), np.zeros((3, 2))),
        },
        coords={"x": np.arange(3.0), "y": np.arange(2.0)},
    )
    fig = curves.field_heatmap_figure(grid, "p_abs")
    assert np.allclose(fig.data[0].z, 1.0)


def test_missing_variable_is_reported():
    grid = xr.Dataset({"a": (("x",), np.ones(3))}, coords={"x": np.arange(3.0)})
    with pytest.raises(KeyError):
        curves.field_heatmap_figure(grid, "nope")


def test_well_plate_heatmap_places_wells_correctly():
    fig = curves.well_plate_heatmap({"A1": 1.0, "H12": 2.0}, rows=8, cols=12)
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z[0, 0] == 1.0
    assert z[7, 11] == 2.0
    assert np.isnan(z[3, 3])


def test_bode_and_nyquist_are_separate_figures():
    """Magnitude and phase never share a y-axis."""
    f = np.logspace(2, 6, 20)
    z = 100 / (1 + 1j * f / 1e4)
    mag = curves.bode_magnitude_figure(f, z)
    phase = curves.bode_phase_figure(f, z)
    assert mag.layout.yaxis.title.text.startswith("|Z|")
    assert phase.layout.yaxis.title.text.startswith("phase")
    nyq = curves.nyquist_figure(f, z)
    assert np.allclose(nyq.data[0].y, -z.imag)


def test_dose_response_marks_the_ic50():
    c = np.logspace(-2, 2, 9)
    y = 1 / (1 + (c / 1.0) ** 1.5)
    fig = curves.dose_response_figure(c, y, ic50=1.0)
    assert fig.layout.xaxis.type == "log"
    assert any("IC50" in str(a.text) for a in fig.layout.annotations)


def test_trajectory_figure_thins_but_keeps_the_legend():
    ds = xr.Dataset(
        {"position": (("particle", "time", "axis"), np.random.default_rng(0).normal(
            size=(120, 10, 3)) * 1e-5)},
        coords={
            "particle": np.arange(120),
            "time": np.linspace(0, 1, 10),
            "axis": ["x", "y", "z"],
            "label": ("particle", np.array(["mcf7"] * 60 + ["rbc"] * 60)),
        },
    )
    fig = curves.trajectory_figure(ds, node_positions=[150e-6], channel_width=300e-6,
                                   max_tracks_per_population=10)
    shown = [t for t in fig.data if t.showlegend]
    assert len(shown) == 2, "exactly one legend entry per population"
