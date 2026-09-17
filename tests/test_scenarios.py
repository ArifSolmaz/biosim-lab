"""Every named scenario must describe a device the page can actually hold.

A preset that seeds a widget with a value outside its own range, or that puts
the app into the one settings pair the model refuses, would turn the
convenience into a crash. These are cheap checks against the real
``configs/`` files, so a config edited later cannot silently break the page.
"""

from __future__ import annotations

import pytest

from biosim_lab.app.scenarios import SOURCES, scenario

#: (min, max, step or None) of the sorter sidebar widget each key feeds.
RANGES = {
    "sorter_frequency_mhz": (1.0, 40.0, None),
    "sorter_voltage_pp": (1.0, 40.0, 0.5),
    "sorter_flow_ul_min": (0.5, 60.0, 0.5),
    "sorter_width_um": (100.0, 800.0, 10.0),
    "sorter_height_um": (20.0, 200.0, 5.0),
    "sorter_length_mm": (0.2, 10.0, None),
    "sorter_temperature_c": (4.0, 45.0, 0.5),
    "sorter_inlet_viability": (0.5, 1.0, None),
    "sorter_tilt_angle_deg": (-89.0, 89.0, None),
    "sorter_collection_fraction": (0.05, 0.9, None),
    "sorter_split_position": (0.05, 0.95, None),
    "sorter_n_cells": (50, 600, 50),
    "sorter_fem_resolution": (16, 64, 8),
    "sorter_seed": (0, 999_999, None),
}
def _cell_keys() -> set[str]:
    from biosim_lab.core.materials import CELL_TYPES

    return set(CELL_TYPES)


CHOICES = {
    "sorter_target": _cell_keys(),
    "sorter_background": _cell_keys(),
    "sorter_inlet": {"sheath_sides", "uniform", "centre", "side"},
    "sorter_inlet_side": {"left", "right"},
    "sorter_outlet_layout": {"centre_band", "lateral_split"},
    "sorter_collect_side": {"right", "left"},
    "sorter_mode": {"analytic", "fem"},
}
NAMES = [n for n in SOURCES if SOURCES[n] is not None]


def test_the_default_scenario_leaves_the_widgets_alone() -> None:
    default = next(n for n in SOURCES if SOURCES[n] is None)
    assert scenario(default) is None


@pytest.mark.parametrize("name", NAMES)
def test_every_value_a_scenario_seeds_is_one_its_widget_can_hold(name: str) -> None:
    loaded = scenario(name)
    assert loaded is not None
    for key, value in loaded["widgets"].items():
        if key in CHOICES:
            assert value in CHOICES[key], f"{name}: {key}={value!r}"
            continue
        low, high, step = RANGES[key]
        assert low <= value <= high, f"{name}: {key}={value} outside [{low}, {high}]"
        if step:
            assert abs((value - low) / step - round((value - low) / step)) < 1e-6, (
                f"{name}: {key}={value} is not on the widget's {step} step"
            )


@pytest.mark.parametrize("name", NAMES)
def test_no_scenario_asks_for_the_pair_the_model_refuses(name: str) -> None:
    """A tilted pattern with the cross-section FEM is exactly what crashed the app."""
    widgets = scenario(name)["widgets"]
    if widgets.get("sorter_tilt_angle_deg"):
        assert widgets["sorter_mode"] == "analytic", name


@pytest.mark.parametrize("name", NAMES)
def test_a_scenario_declares_what_it_could_not_express(name: str) -> None:
    """The sidebar carries one background; a reduced sample must say so."""
    loaded = scenario(name)
    kind, ref = SOURCES[name]
    if kind != "config":
        return
    from biosim_lab.app.scenarios import CONFIG_DIR
    from biosim_lab.core.config import ExperimentConfig
    from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams

    params = ExperimentConfig.from_yaml(CONFIG_DIR / ref).validated_params(SAWSorterParams)
    backgrounds = [p for p in params.populations if not p.target]
    if len(backgrounds) > 1:
        assert any("one background population" in n for n in loaded["notes"]), (
            f"{name} drops {len(backgrounds) - 1} populations without saying so"
        )


def test_the_li_2015_scenario_carries_the_published_numbers() -> None:
    """The point of that preset is the comparison, so the figures must be on screen."""
    loaded = scenario("Li et al. 2015 — tilted-angle CTC sorter")
    assert "recovery" in loaded["description"]
    assert loaded["widgets"]["sorter_tilt_angle_deg"] != 0.0
    assert loaded["widgets"]["sorter_outlet_layout"] == "lateral_split"
