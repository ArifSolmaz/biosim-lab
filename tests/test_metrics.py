"""Paper metric definitions, the RF-power drive, and cells pressed against a wall."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biosim_lab.instruments.saw_sorter import metrics
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    dbm_to_watt,
    pressure_from_rf_power,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation


@pytest.fixture()
def cells():
    # 10 targets: 8 collected, 1 waste, 1 never left. 20 background: 2 collected.
    return pd.DataFrame({
        "label": ["ctc"] * 10 + ["pbmc"] * 20,
        "is_target": [True] * 10 + [False] * 20,
        "outlet": ["collect"] * 8 + ["waste"] * 2 + ["collect"] * 2 + ["waste"] * 18,
        "exited": [True] * 9 + [False] + [True] * 20,
        "x_outlet_m": [500e-6] * 10 + [100e-6] * 20,
        "x_initial_m": [50e-6] * 30,
    })


def test_capture_efficiency_counts_every_cell_that_entered(cells):
    # doi:10.3390/ijms24043338 eq. 1: CCs_captured / CCs_in
    assert metrics.capture_efficiency(cells) == pytest.approx(80.0)


def test_contamination_rate(cells):
    # doi:10.3390/ijms24043338 eq. 2: PBMCs_captured / PBMCs_in
    assert metrics.contamination_rate(cells) == pytest.approx(10.0)


def test_recovery_rate_counts_only_what_came_out(cells):
    # doi:10.1073/pnas.1504484112 p. 4973: collection / (collection + waste).
    # The cell stuck in the device is invisible to the outlet dishes.
    assert metrics.recovery_rate(cells) == pytest.approx(100.0 * 8 / 9)


def test_removal_rate_and_separation_distance(cells):
    assert metrics.removal_rate(cells) == pytest.approx(90.0)
    assert metrics.separation_distance(cells) == pytest.approx(400.0)


def test_separation_distance_is_negative_when_the_background_overtakes(cells):
    flipped = cells.assign(x_outlet_m=np.where(cells["is_target"], 100e-6, 500e-6))
    assert metrics.separation_distance(flipped) < 0


def test_rf_power_to_pressure_scalings():
    kw = dict(reference_pressure=0.4e6, reference_power_dbm=35.0, reference_idt_length=10e-3)
    assert pressure_from_rf_power(35.0, idt_length=10e-3, **kw) == pytest.approx(0.4e6)
    assert pressure_from_rf_power(38.0103, idt_length=10e-3, **kw) == pytest.approx(
        0.4e6 * np.sqrt(2), rel=1e-4)
    assert pressure_from_rf_power(35.0, idt_length=20e-3, **kw) == pytest.approx(
        0.4e6 / np.sqrt(2))
    assert dbm_to_watt(30.0) == pytest.approx(1.0)


def _li(**kw):
    base = dict(
        mode="tassaw", frequency="19.573 MHz", tilt_angle_deg=-5.0, channel_width="800 um",
        channel_height="110 um", channel_length="10 mm", flow_rate="75 uL/min",
        sheath_ratio=2.5, inlet="side", inlet_side="left", fluid="pbs",
        outlet_layout="lateral_split", split_position=0.5, collect_side="right",
        power_drive=dict(power_dbm=35.0, reference_pressure="0.44 MPa",
                         reference_power_dbm=35.0, reference_idt_length="10 mm",
                         reference_source="test"),
        track_viability=False, seed=4,
        populations=[dict(cell_type="mcf7", count=15, target=True),
                     dict(cell_type="wbc", count=15)],
    )
    base.update(kw)
    return SAWSorterParams.model_validate(base)


def test_power_drive_sets_p0_and_the_heating_input():
    p = _li(power_drive=dict(power_dbm=38.0103, reference_pressure="0.44 MPa",
                             reference_power_dbm=35.0, reference_idt_length="10 mm",
                             reference_source="test"))
    assert p.p0 == pytest.approx(0.44e6 * np.sqrt(2), rel=1e-4)


def test_cells_driven_into_the_far_wall_still_reach_the_outlet():
    """Hard over-drive of a tilted device presses targets into the far wall.

    With the cell CENTRE clamped onto the wall plane the no-slip velocity there
    is zero and the cell never exits --- a numerical artefact. With the centre
    held one radius off the wall it flows out along the wall, as a real cell does.
    """
    out = SAWSorterSimulation(_li(power_drive=dict(
        power_dbm=40.0, reference_pressure="0.44 MPa", reference_power_dbm=35.0,
        reference_idt_length="10 mm", reference_source="test"))).run()
    assert out.metrics["all_cells_exited"]
    target = out.cells[out.cells["is_target"]]
    assert (target["x_outlet_m"] <= 800e-6 - target["radius_m"] + 1e-12).all()
    assert out.metrics["recovery_rate_percent"] == pytest.approx(100.0)
