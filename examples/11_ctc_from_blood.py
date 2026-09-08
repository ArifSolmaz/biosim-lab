"""Isolating tumour cells from blood, at the composition a real sample has.

The other examples separate 300 tumour cells from 300 red cells. A patient
sample is nothing like that: about 5e9 red cells per mL against single-digit
circulating tumour cells, a ratio near 1 : 1e9, at 45 % cells by volume. Three
things follow, and each is a thing the 50:50 demo hides.

**Whole blood cannot go in.** Not a throughput limit --- a modelling one. At 45 %
by volume the cells sit about two radii apart, where particle-particle
scattering and acoustic streaming dominate, and none of that is in this model.
Every published acoustophoretic CTC protocol lyses the red cells first, which is
what this script does (doi:10.1073/pnas.1504484112,
doi:10.1021/acs.analchem.1c04050).

**The real ratio cannot be simulated.** Ten tumour cells at that abundance means
tracking ten billion red cells. What transfers instead is each population's
*probability* of reaching the collection outlet, applied afterwards to the real
abundances.

**Purity is the wrong headline.** At these ratios every device looks terrible on
purity and the number that matters is log10 depletion of the background.

Run::

    python examples/11_ctc_from_blood.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from biosim_lab.core import samples
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.protocols import li_2015_tassaw
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    # -- 1. why nobody runs whole blood ----------------------------------
    blood = samples.whole_blood_with_ctc("mcf7")
    lysed = samples.rbc_lysis(blood)

    print("Sample preparation")
    print("-" * 66)
    rows = []
    for name, comp in (("whole blood", blood), ("after RBC lysis", lysed)):
        report = samples.dilution_report(comp)
        rows.append({
            "sample": name,
            "cells_per_ml": sum(comp.values()),
            "volume_fraction_percent": 100 * report.volume_fraction,
            "separation_radii": report.mean_separation_radii,
            "is_dilute": report.is_dilute,
        })
        print(f"  {name:<18} {report}")
    print(
        "\n  Lysis is not a convenience: it is what brings the sample inside the\n"
        "  dilute-suspension assumption this model is built on."
    )

    # -- 2. the published device, as published ---------------------------
    proto = li_2015_tassaw()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        outcome = SAWSorterSimulation(proto["params"]).run()
    physio = outcome.diagnostics["physiological"]
    reported = proto["reported"]

    print("\n\nLi et al. 2015 taSSAW device, run at the composition it was fed")
    print("-" * 66)
    print(f"  {'quantity':<30} {'model':>9} {'published':>11}")
    wbc_removed = 100.0 * (1.0 - physio["collection_probability"].get("wbc", 0.0))
    print(f"  {'cancer-cell recovery':<30} {physio['recovery_percent']:8.1f}% "
          f"{reported['cancer_recovery_percent']:10.0f}%")
    print(f"  {'leukocytes removed':<30} {wbc_removed:8.1f}% "
          f"{reported['wbc_removal_percent']:10.0f}%")
    print(f"  {'background depletion (log10)':<30} "
          f"{physio['background_depletion_log10']:8.2f} "
          f"{reported['wbc_depletion_log10']:10.1f}")

    print("\n  Assumed, because the paper does not state it:")
    for item in proto["assumptions"]:
        print(f"    - {item}")

    # -- 3. what the drive does, which is not what you expect ------------
    print("\n\nDrive sweep: more power is not better")
    print("-" * 66)
    base = proto["params"].model_dump(mode="python")
    base.pop("pressure_amplitude", None)
    drive_rows = []
    for voltage in (8, 10, 12, 15, 18, 22):
        config = dict(base)
        config["voltage_pp"] = float(voltage)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            sim = SAWSorterSimulation(SAWSorterParams(**config))
            result = sim.run()
        ph = result.diagnostics["physiological"]
        removed = 100.0 * (1.0 - ph["collection_probability"].get("wbc", 0.0))
        drive_rows.append({
            "voltage_pp": voltage,
            "pressure_MPa": sim.params.p0 / 1e6,
            "recovery_percent": ph["recovery_percent"],
            "wbc_removed_percent": removed,
            "depletion_log10": ph["background_depletion_log10"],
            "demonstrable_log10": ph["background_depletion_log10_demonstrated"],
        })
    frame = pd.DataFrame(drive_rows)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(
        "\n  Recovery saturates once the target is fully deflected; past that\n"
        "  point more power only pushes leukocytes across the divider as well,\n"
        "  so removal collapses. The useful setting is the LOWEST drive that\n"
        "  still collects the target."
    )
    print(f"\n  {proto['calibrated_pressure_note']}")

    print(
        "\n  Note the last column. Depletion beyond what the simulated ensemble\n"
        "  can resolve is not evidence: with a few hundred cells per population\n"
        "  nothing past ~2 logs can be demonstrated, however good the point\n"
        "  estimate looks."
    )

    out = ASSETS / "11_ctc_from_blood.csv"
    pd.concat([
        pd.DataFrame(rows).assign(part="sample_preparation"),
        frame.assign(part="drive_sweep"),
    ]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
