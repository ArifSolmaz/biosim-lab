"""Published device configurations, so a run can be compared against real data.

Each function returns the parameters of a device someone actually built and
measured, together with what they reported. That makes them the closest thing
this package has to validation against an experiment rather than against
another calculation: run one, and the metrics either land near the published
figures or they do not.

Parameters not stated in a paper are marked ASSUMPTION in the returned
``reference`` block. There are always some --- a paper gives the channel width
and the frequency, rarely the acoustic pressure.
"""

from __future__ import annotations

from typing import Any

from biosim_lab.core.materials import Provenance
from biosim_lab.core.samples import rbc_lysis, whole_blood_with_ctc
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams

LI_2015 = Provenance(
    doi="10.1073/pnas.1504484112",
    citation="Li et al. (2015), Acoustic separation of circulating tumor cells, "
    "PNAS 112:4970",
)


def li_2015_tassaw(
    *,
    cancer_cell: str = "mcf7",
    simulated_count: int = 300,
    ctc_per_ml: float = 100.0,
) -> dict[str, Any]:
    """The tilted-angle SAW CTC separator of Li et al. (2015).

    Reported: **>83 % cancer-cell recovery** (83-96 % across lines) with **~90 %
    of leukocytes removed**, on RBC-lysed blood at up to 20 uL/min sample flow.

    Stated in the paper and used here: 19.573 MHz, an 800 x 110 um channel, a
    tilt angle of about 5 degrees, and blood whose red cells were lysed and the
    remainder resuspended in an equal volume of PBS. Note ~100 cells/mL is the
    spike concentration they separated, not a physiological CTC burden.

    Not stated, and therefore assumed:

    * the **acoustic pressure**. The paper gives an RF input of 35-38 dBm, and
      nothing in this package converts drive power to pressure --- that is the
      project's single largest assumption. 15 Vpp is used via the documented
      voltage calibration, so absolute forces here are indicative.
    * the **active channel length**. The IDTs are 8-10 mm; 10 mm is taken.
    * the **inlet**, taken as one sheath-focused side stream, which is what a
      2.5:1 sheath-to-sample ratio implies.

    A 800 um channel at 19.573 MHz holds about eight pressure nodes, so the
    multi-node warning fires. That is correct and is the point of the design:
    a tilted device works by carrying cells ACROSS many nodes, not by parking
    them on one.
    """
    lysed = rbc_lysis(whole_blood_with_ctc(cancer_cell, ctc_per_ml=ctc_per_ml))

    params = SAWSorterParams(
        frequency="19.573 MHz",
        voltage_pp="15 V",
        channel_width="800 um",
        channel_height="110 um",
        channel_length="10 mm",
        flow_rate="20 uL/min",
        fluid="pbs",
        substrate="linbo3_128yx",
        inlet="side",
        inlet_side="left",
        tilt_angle_deg=-5.0,
        outlet_layout="lateral_split",
        split_position=0.5,
        collect_side="right",
        mode="analytic",
        seed=20260907,
        populations=[
            {
                "cell_type": key,
                "count": simulated_count,
                "abundance_per_ml": value,
                "target": key == cancer_cell,
            }
            for key, value in lysed.items()
        ],
    )
    return {
        "params": params,
        "composition": lysed,
        "reference": LI_2015,
        "reported": {
            "cancer_recovery_percent": 83.0,
            "cancer_recovery_range_percent": (83.0, 96.0),
            "wbc_removal_percent": 90.0,
            "wbc_depletion_log10": 1.0,
            "sample_flow_ul_min": 20.0,
        },
        "calibrated_pressure_note": (
            "Sweeping the drive to reproduce the published 83 % recovery and 90 % "
            "leukocyte removal lands at p0 ~ 0.24-0.30 MPa, NOT the 0.45 MPa the "
            "project's 15 Vpp calibration gives. Read that as evidence the "
            "voltage-to-pressure assumption is roughly 1.5x too high, and as the "
            "closest thing here to an external check on it."
        ),
        "over_driving_note": (
            "Recovery saturates near 73 % above ~15 Vpp while leukocyte removal "
            "collapses from 91 % to 27 %: past the point where the target is fully "
            "deflected, more power only pushes the background across the divider "
            "too. The useful operating point is the lowest drive that still "
            "collects the target, which is the opposite of the intuition that a "
            "stronger field separates harder."
        ),
        "assumptions": [
            "acoustic pressure: 35-38 dBm RF input is not convertible to a "
            "pressure amplitude by anything in this package; 15 Vpp assumed",
            "active channel length: IDTs are 8-10 mm, 10 mm assumed",
            "inlet: one side stream, implied by the 2.5:1 sheath ratio",
            "residual red cells after lysis: 0.1 % assumed",
        ],
    }


__all__ = ["LI_2015", "li_2015_tassaw"]
