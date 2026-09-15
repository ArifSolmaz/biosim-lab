"""Cases (a)-(f) and the counterfactual of benchmark_02: Zhang et al. (2023).

doi:10.3390/ijms24043338. The device comes from ``config.yaml``; stated values
to compare against come from ``reference.yaml``.

What is calibrated, and from what
---------------------------------
The paper gives no acoustic energy density for either mode. Both are derived
from sentences in its Sec. 4.2 / 2.2, never from a result compared below:

* ``E_1MHz`` --- the logarithmic midpoint of the interval allowed by the
  stated W/6 design rule
  (:func:`~biosim_lab.instruments.saw_sorter.design.separation_rule_energy`).
* ``E_3MHz`` --- the level that returns a PBMC to its node in the ~1 s after
  which contamination stops improving (:func:`~...design.return_energy`). This
  one IS informed by the Fig. 2a statement, so case (a)'s "minimum beyond 1 s"
  trend is labelled calibration-informed.

The acoustic region length is not stated either; case (d) sweeps it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.common import Check, load_yaml, parallel_map
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import ConfigurationError, RegimeWarning
from biosim_lab.instruments.saw_sorter.design import (
    operating_window,
    return_energy,
    separation_rule_energy,
)
from biosim_lab.instruments.saw_sorter.simulate import (
    SAWSorterParams,
    SAWSorterSimulation,
    SortingOutcome,
)

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
REFERENCE = HERE / "reference.yaml"

T3_SWEEP_S = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.0, 3.0)
T1_SWEEP_S = (0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4)
V1_SWEEP_VPP = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0)
FLOWS_UL_H = (150.0, 300.0, 900.0)
LENGTHS_MM = (10.0, 20.0, 30.0, 40.0)

#: Window criteria for "high capture, low contamination" (Table 1 scale: ~95 % / ~1 %).
WINDOW = {"capture_min": 90.0, "contamination_max": 5.0}

ZHANG_KAPPA_NOTE = (
    "ASSUMPTION: cancer-line compressibility 4.3e-10 1/Pa, the level of the box plots "
    "in Zhang 2023 Fig. 1 (not a stated number)"
)

LINES = {
    # label: (cell_type, overrides) — Table 1 lines, sized per the stated 14.8-19.6 um range
    "MCF7": ("mcf7", {"compressibility": 4.3e-10, "override_source": ZHANG_KAPPA_NOTE}),
    "HCT116": ("hct116", {}),
    "A549": ("a549", {"compressibility": 4.3e-10, "override_source": ZHANG_KAPPA_NOTE}),
}


def reference() -> dict[str, Any]:
    return load_yaml(REFERENCE)


def base_config() -> dict[str, Any]:
    return dict(ExperimentConfig.from_yaml(CONFIG).params)


def make_params(overrides: dict[str, Any] | None = None) -> SAWSorterParams:
    """Device from ``config.yaml``; phase keys ``T1, T3, V1, V3, E1, E3`` edit the drive."""
    data = base_config()
    ov = dict(overrides or {})
    sw = dict(data["switching"])
    phases = [dict(p) for p in sw["phases"]]
    for key, (i, field) in {
        "T1": (0, "duration"), "T3": (1, "duration"),
        "V1": (0, "voltage_pp"), "V3": (1, "voltage_pp"),
        "E1": (0, "reference_energy_density"), "E3": (1, "reference_energy_density"),
    }.items():
        if key in ov:
            phases[i][field] = ov.pop(key)
    for key in ("entry", "entry_time_in_cycle", "time_step", "integrator"):
        if key in ov:
            sw[key] = ov.pop(key)
    sw["phases"] = phases
    data["switching"] = sw
    data.update(ov)
    return SAWSorterParams.model_validate(data)


def calibration(params: SAWSorterParams | None = None) -> dict[str, Any]:
    """Recompute both calibrated energy densities for the configured cells."""
    params = params or make_params()
    sim = SAWSorterSimulation(params)
    target = next(p for p in params.populations if p.target)
    background = next(p for p in params.populations if not p.target)
    ph1 = params.switching.phases[0]  # type: ignore[union-attr]
    rule = separation_rule_energy(
        sim.cell_for(target), sim.cell_for(background), sim.fluid,
        width=params.channel_width, duration=ph1.duration,
    )
    e3 = return_energy(sim.cell_for(background), sim.fluid, width=params.channel_width,
                       duration=1.0, start_fraction=0.9, end_fraction=0.1)
    return {"E1_J_m3": rule.chosen, "E3_J_m3": e3, "rule": rule.as_dict(),
            "E3_rule": "mean PBMC from 90 % to 10 % of node-antinode distance in 1.0 s"}


def _run(params: SAWSorterParams) -> SortingOutcome:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return SAWSorterSimulation(params).run()


def point(args: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """One sorting run; returns the swept keys and the Zhang metrics."""
    overrides, seed = args
    try:
        params = make_params({**overrides, "seed": seed})
        m = _run(params).metrics
    except ConfigurationError as exc:
        return {**_scalar(overrides), "error": str(exc)}
    return {
        **_scalar(overrides),
        "capture_efficiency_percent": m["capture_efficiency_percent"],
        "contamination_rate_percent": m["contamination_rate_percent"],
        "all_cells_exited": m["all_cells_exited"],
    }


def _scalar(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, (list, dict))}


def sweep(key: str, values: tuple[Any, ...], extra: dict[str, Any] | None = None,
          *, seed: int = 5) -> pd.DataFrame:
    jobs = [({**(extra or {}), key: v}, seed) for v in values]
    return pd.DataFrame(parallel_map(point, jobs))


# ---------------------------------------------------------------------------
# (a) 3 MHz duration
# ---------------------------------------------------------------------------


def case_a(values: tuple[float, ...] = T3_SWEEP_S) -> tuple[pd.DataFrame, list[Check]]:
    df = sweep("T3", values)
    cap = df["capture_efficiency_percent"].to_numpy()
    con = df["contamination_rate_percent"].to_numpy()
    t3 = df["T3"].to_numpy()
    beyond = con[t3 > 1.0]
    before = con[t3 < 1.0]
    checks = [
        Check("2a", "capture changes little with the 3 MHz duration", "trend",
              f"{cap.min():.0f}-{cap.max():.0f} % over {t3.min():g}-{t3.max():g} s",
              "spread <= 10 points", passed=bool(cap.max() - cap.min() <= 10.0)),
        Check("2a", "contamination falls with the 3 MHz duration", "trend",
              f"{con[0]:.1f} -> {con[-1]:.1f} %", "decreasing",
              passed=bool(con[0] > con[-1] and np.all(np.diff(con) <= 2.0))),
        Check("2a", "contamination minimum lies beyond T3 = 1 s", "trend",
              f"min {con.min():.1f} % at {t3[int(np.argmin(con))]:g} s; "
              f"{beyond.max():.1f} % worst beyond 1 s vs {before.min():.1f} % best before",
              "minimum at T3 > 1 s",
              passed=bool(t3[int(np.argmin(con))] > 1.0 and beyond.max() < before.min()),
              note="calibration-informed: E_3MHz was derived from this statement"),
        Check("2a", "contamination is FLAT beyond 1 s (plateau reading)", "claim",
              f"{beyond.max():.1f} -> {beyond.min():.1f} % over 1.2-{t3.max():g} s",
              "plateau (within 3 points)", passed=bool(beyond.max() - beyond.min() <= 3.0),
              note=("the model keeps improving: a longer 3 MHz phase gives more of the "
                    "PBMCs near the W/3 edge of the stream a full return to W/6 before their "
                    "first 1 MHz push, and that fraction keeps growing past 1 s")),
    ]
    return df, checks


# ---------------------------------------------------------------------------
# (b) 1 MHz duration and amplitude, and the operating window
# ---------------------------------------------------------------------------


def case_b(t1: tuple[float, ...] = T1_SWEEP_S, v1: tuple[float, ...] = V1_SWEEP_VPP
           ) -> tuple[dict[str, pd.DataFrame], list[Check], dict[str, Any]]:
    out = {"T1": sweep("T1", t1), "V1": sweep("V1", v1)}
    checks: list[Check] = []
    windows: dict[str, Any] = {}
    for key, df in out.items():
        cap = df["capture_efficiency_percent"].to_numpy()
        con = df["contamination_rate_percent"].to_numpy()
        win = operating_window(df, key, **WINDOW)
        windows[key] = win
        best = df[key].to_numpy() == win["best_setting"]
        i_best = int(np.argmax(best))
        d_cap, d_con = cap[i_best] - cap[0], con[i_best] - con[0]
        unit = "s" if key == "T1" else "Vpp"
        checks += [
            Check(f"2{'b' if key == 'T1' else 'c'}", f"both metrics rise with 1 MHz {key}",
                  "trend", f"capture {cap[0]:.0f}->{cap[-1]:.0f} %, contamination "
                  f"{con[0]:.1f}->{con[-1]:.1f} %", "increasing",
                  passed=bool(cap[-1] > cap[0] and con[-1] > con[0]
                              and np.all(np.diff(cap) >= -3.0))),
            Check(f"2{'b' if key == 'T1' else 'c'}",
                  f"contamination rises much less than capture (up to the best {key})",
                  "trend",
                  f"+{d_cap:.0f} vs +{d_con:.1f} points (to {win['best_setting']:g} {unit})",
                  "d(contamination) <= d(capture)/3", passed=bool(d_con <= d_cap / 3.0)),
            Check(f"2{'b' if key == 'T1' else 'c'}",
                  f"a high-capture / low-contamination window in {key}",
                  "claim",
                  (f"{win['window'][0]:g}-{win['window'][1]:g} {unit}" if win["window"]
                   else f"none (best J = {win['best_youden_percent']:.0f} % at "
                        f"{win['best_setting']:g} {unit}: "
                        f"{win['best_capture_percent']:.0f} % / "
                        f"{win['best_contamination_percent']:.1f} %)"),
                  f"capture >= {WINDOW['capture_min']:g} %, contamination <= "
                  f"{WINDOW['contamination_max']:g} %",
                  passed=bool(win["window"])),
        ]
    return out, checks, windows


# ---------------------------------------------------------------------------
# (c) Fig. 8 trajectories and forces
# ---------------------------------------------------------------------------


def fig8_case(entry_time: float, *, n: int = 12) -> SortingOutcome:
    """Monodisperse MCF-7 and PBMC entering together at one point of the cycle."""
    pops = []
    for pop in base_config()["populations"]:
        src = pop.get("override_source")
        note = "monodisperse cells for the trajectory figure (Zhang 2023 Fig. 8)"
        pops.append(dict(pop, count=n, diameter_cv=0.0,
                         override_source=f"{src}; {note}" if src else note))
    # The figure's premise is the stated y0 < W/3 (Sec. 4.2): starting points
    # spread evenly across it, mid-height, so every part of the stream is shown.
    params = make_params({
        "entry": "fixed", "entry_time_in_cycle": entry_time, "time_step": 2e-3,
        "populations": pops, "record_forces": True,
        "inlet_x": list(np.linspace(0.01, 1.0 / 3.0 - 0.01, n)),
        "n_time_samples": 4000, "seed": 8,
    })
    return _run(params)


def case_c() -> tuple[dict[str, SortingOutcome], list[Check]]:
    params = make_params()
    t1 = params.switching.phases[0].duration  # type: ignore[union-attr]
    t3 = params.switching.phases[1].duration  # type: ignore[union-attr]
    period = t1 + t3
    outcomes = {
        "a_start_of_1MHz": fig8_case(0.0),
        "b_end_of_3MHz": fig8_case(t1 + 0.85 * t3),
    }
    w = params.channel_width
    checks: list[Check] = []
    summary: dict[str, dict[str, float]] = {}
    for key, out in outcomes.items():
        tr = out.tracks.trajectories
        t_after = 2.0 * period
        x = tr["position"].sel(axis="x").interp(time=t_after).values / w
        labels = tr["label"].values
        mcf = x[labels == "MCF7"]
        pb = x[labels == "PBMC"]
        summary[key] = {
            "mcf7_at_midline": float(np.mean(np.abs(mcf - 0.5) < 0.05)),
            "pbmc_near_W6": float(np.mean(np.abs(pb - 1.0 / 6.0) < 0.05)),
            "pbmc_at_midline": float(np.mean(np.abs(pb - 0.5) < 0.05)),
        }
    a, b = summary["a_start_of_1MHz"], summary["b_end_of_3MHz"]
    checks += [
        Check("8b", "entering at the end of 3 MHz: complete separation after 2 cycles",
              "claim", f"MCF7 at midline {100 * b['mcf7_at_midline']:.0f} %, PBMC at "
              f"midline {100 * b['pbmc_at_midline']:.0f} %", "100 % / 0 %",
              passed=bool(b["mcf7_at_midline"] == 1.0 and b["pbmc_at_midline"] == 0.0),
              note=("a PBMC that enters just below the W/3 antinode in the last 15 % of the "
                    "3 MHz phase is not pulled back before the next 1 MHz phase with the "
                    "calibrated E_3MHz; the paper does not state which y0 it simulated")),
        Check("8", "after 2 cycles most PBMCs sit around the W/6 node", "claim",
              f"{100 * a['pbmc_near_W6']:.0f} % (entry at 1 MHz start), "
              f"{100 * b['pbmc_near_W6']:.0f} % (entry at 3 MHz end)", "most",
              passed=bool(min(a["pbmc_near_W6"], b["pbmc_near_W6"]) > 0.5)),
        Check("8a", "entering at the start of 1 MHz: some PBMCs reach the midline",
              "claim", f"{100 * a['pbmc_at_midline']:.0f} % of PBMCs at midline",
              "a small portion", passed=bool(0.0 < a["pbmc_at_midline"] <= 0.5)),
        Check("8", "entry at the start of 1 MHz is the worse case for contamination",
              "trend",
              f"{100 * a['pbmc_at_midline']:.0f} % vs {100 * b['pbmc_at_midline']:.0f} %",
              "a > b", passed=bool(a["pbmc_at_midline"] > b["pbmc_at_midline"])),
    ]
    return outcomes, checks


# ---------------------------------------------------------------------------
# (d) flow rate, with the unstated acoustic length swept
# ---------------------------------------------------------------------------


def durations_for(flow_ul_h: float, length_mm: float) -> tuple[float, float, float]:
    """Stated durations, shortened only as much as the two-cycle rule requires.

    Zhang et al. say the durations "should be reduced correspondingly" at
    higher flow but not to what. This takes the least reduction consistent with
    their other stated requirement --- every cell sees two cycles --- keeping
    the 0.8 : 1.4 ratio. Returns ``(T1, T3, scale)``.
    """
    params = make_params({"flow_rate": f"{flow_ul_h} uL/h",
                          "channel_length": f"{length_mm} mm"})
    sim = SAWSorterSimulation(params)
    t1, t3 = (p.duration for p in params.switching.phases)  # type: ignore[union-attr]
    t_fast = params.channel_length / sim.flow.max_velocity
    scale = min(1.0, t_fast / (2.0 * (t1 + t3)))
    # Snap down to the RK4 grid so the phases stay whole steps.
    dt = params.switching.time_step  # type: ignore[union-attr]
    return (max(dt, np.floor(t1 * scale / dt) * dt), max(dt, np.floor(t3 * scale / dt) * dt),
            scale)


def flow_point(args: tuple[str, float, float, int]) -> dict[str, Any]:
    line, flow, length, seed = args
    key, extra = LINES[line]
    t1, t3, scale = durations_for(flow, length)
    pbmc = next(p for p in base_config()["populations"] if not p.get("target"))
    pops = [{"cell_type": key, "count": 200, "target": True, "label": line, **extra}, pbmc]
    row = point(({"flow_rate": f"{flow} uL/h", "channel_length": f"{length} mm",
                  "T1": t1, "T3": t3, "populations": pops}, seed))
    return {**row, "line": line, "flow_ul_h": flow, "length_mm": length,
            "T1_s": t1, "T3_s": t3, "duration_scale": scale}


def case_d(flows: tuple[float, ...] = FLOWS_UL_H,
           lengths: tuple[float, ...] = LENGTHS_MM) -> tuple[pd.DataFrame, list[Check]]:
    ref = reference()["results"]["fig4_flow"]
    length0 = base_config()["channel_length"]
    length0_mm = float(str(length0).split()[0])
    jobs = [("A549", q, length0_mm, 9) for q in flows]
    jobs += [("A549", 900.0, mm, 9) for mm in lengths if mm != length0_mm]
    df = pd.DataFrame(parallel_map(flow_point, jobs))
    main = df[df["length_mm"] == length0_mm].set_index("flow_ul_h")
    checks: list[Check] = []
    for q in flows:
        paper = ref["capture_percent"][str(int(q))]["mean"]
        row = main.loc[q]
        checks.append(Check(
            "4", f"A549 capture at {q:g} uL/h", "quantity",
            row["capture_efficiency_percent"], paper, "%", tolerance=5.0,
            note=f"durations {row['T1_s']:.2f} s / {row['T3_s']:.2f} s "
                 f"(x{row['duration_scale']:.2f} of stated)",
            cause=_flow_cause(q),
        ))
        checks.append(Check(
            "4", f"PBMC contamination at {q:g} uL/h", "quantity",
            row["contamination_rate_percent"], ref["contamination_percent"], "%", tolerance=5.0,
            cause=CONTAMINATION_CAUSE,
        ))
    cap = [main.loc[q, "capture_efficiency_percent"] for q in flows]
    checks.append(Check(
        "4", "capture holds from 150 to 300 uL/h, then drops at 900", "trend",
        " -> ".join(f"{c:.0f}" for c in cap) + " %", "flat, then lower",
        passed=bool(abs(cap[1] - cap[0]) <= 5.0 and cap[2] < min(cap[0], cap[1]) - 5.0),
    ))
    return df, checks


def _flow_cause(flow: float) -> str:
    if flow < 900.0:
        return ("E_1MHz is calibrated to give the MCF7/PBMC rule a symmetric margin; "
                "A549 is larger than MCF7 here, so it clears the rule with room to spare")
    return ("the acoustic region length and the reduced durations used at 900 uL/h are both "
            "unstated; with the least duration cut the two-cycle rule allows, the result "
            "depends strongly on the assumed length (see the length sweep)")


CONTAMINATION_CAUSE = (
    "PBMCs that start in the upper part of the sample stream, near its W/3 edge, and meet "
    "a 1 MHz phase before a full 3 MHz phase has pulled them back to W/6, cross W/3; the "
    "paper's 1:2 ratio puts that edge exactly on the 3 MHz antinode, an unstable point. "
    "Confining the inlet to y0 < W/6, as the paper says would be ideal, removes most of "
    "it (contamination-sensitivity table)"
)


# ---------------------------------------------------------------------------
# (e) Table 1: best window per line
# ---------------------------------------------------------------------------


def line_point(args: tuple[str, float, int]) -> dict[str, Any]:
    line, v1, seed = args
    key, extra = LINES[line]
    pbmc = next(p for p in base_config()["populations"] if not p.get("target"))
    pops = [{"cell_type": key, "count": 200, "target": True, "label": line, **extra}, pbmc]
    return {**point(({"V1": v1, "populations": pops}, seed)), "line": line}


def case_e(v1: tuple[float, ...] = V1_SWEEP_VPP) -> tuple[pd.DataFrame, list[Check],
                                                          dict[str, Any]]:
    """Table 1 at each line's own best 1 MHz amplitude, as the paper did (Fig. S1)."""
    ref = reference()["results"]["table1"]["lines"]
    jobs = [(line, v, 13) for line in LINES for v in v1]
    df = pd.DataFrame(parallel_map(line_point, jobs))
    checks: list[Check] = []
    windows: dict[str, Any] = {}
    for line in LINES:
        sub = df[df["line"] == line]
        win = operating_window(sub, "V1", **WINDOW)
        windows[line] = win
        cap_ref, con_ref = ref[line]["capture"][0], ref[line]["contamination"][0]
        note = f"at the Youden-best 1 MHz amplitude, {win['best_setting']:g} Vpp"
        checks.append(Check("Table 1", f"{line} capture", "quantity",
                            win["best_capture_percent"], cap_ref, "%", tolerance=5.0,
                            note=note, cause=(
                                "capture at the Youden optimum is pulled up by the high "
                                "contamination penalty elsewhere; see contamination")))
        checks.append(Check("Table 1", f"{line} PBMC contamination", "quantity",
                            win["best_contamination_percent"], con_ref, "%", tolerance=5.0,
                            note=note, cause=CONTAMINATION_CAUSE))
    return df, checks, windows


# ---------------------------------------------------------------------------
# contamination sensitivity: which assumption the baseline deviation rests on
# ---------------------------------------------------------------------------


def contamination_sensitivity() -> pd.DataFrame:
    base = base_config()
    pbmc = next(p for p in base["populations"] if not p.get("target"))
    tight = dict(pbmc, diameter_cv=0.10,
                 override_source="sensitivity: PBMC size CV 10 % instead of 15 %")
    variants = {
        "baseline (config.yaml)": {},
        "inlet band exactly y0 < W/3, as stated": {"sheath_ratio": None, "inlet_band": 1 / 3},
        "inlet confined to y0 < W/6 (sheath 5:1)": {"sheath_ratio": 5.0},
        "every cell enters at the end of 3 MHz": {"entry": "fixed",
                                                  "entry_time_in_cycle": 0.8 + 0.85 * 1.4},
        "PBMC size CV 10 %": {"populations": [base["populations"][0], tight]},
    }
    rows = parallel_map(point, [(ov, 5) for ov in variants.values()])
    df = pd.DataFrame(rows)
    df.insert(0, "variant", list(variants))
    return df[["variant", "capture_efficiency_percent", "contamination_rate_percent"]]


# ---------------------------------------------------------------------------
# counterfactual: similar sizes, with and without the compressibility difference
# ---------------------------------------------------------------------------

#: Patient-like sizes from the stated "PBMC/CTC ~10-13 um" band with < 2 um between them.
CF_SIZES_UM = {"CTC": 12.0, "PBMC": 10.5}
CF_KAPPA = {"PBMC": 4.0e-10, "CTC_measured": 4.3e-10, "CTC_reversed": 3.7e-10}
CF_V1 = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 17.0, 20.0, 25.0)
CF_SUCCESS = {"capture_min": 80.0, "contamination_max": 10.0}

CF_ARMS = {
    "size only (equal compressibility)": CF_KAPPA["PBMC"],
    "size + measured compressibility (CTC more compressible)": CF_KAPPA["CTC_measured"],
    "size + reversed compressibility (CTC less compressible)": CF_KAPPA["CTC_reversed"],
}


def cf_point(args: tuple[str, float, int]) -> dict[str, Any]:
    arm, v1, seed = args
    note = ("counterfactual: patient-like sizes from the project spec's 10-13 um band "
            "(Zhang 2023 Fig. 1 region, not stated numbers); CV 10 %")
    pops = [
        {"cell_type": "mcf7", "count": 200, "target": True, "label": "CTC",
         "diameter": f"{CF_SIZES_UM['CTC']} um", "diameter_cv": 0.10, "density": 1068.0,
         "compressibility": CF_ARMS[arm], "override_source": note},
        {"cell_type": "pbmc", "count": 400, "label": "PBMC",
         "diameter": f"{CF_SIZES_UM['PBMC']} um", "diameter_cv": 0.10, "density": 1068.0,
         "compressibility": CF_KAPPA["PBMC"], "override_source": note},
    ]
    return {**point(({"V1": v1, "populations": pops}, seed)), "arm": arm}


def counterfactual(v1: tuple[float, ...] = CF_V1) -> tuple[pd.DataFrame, list[Check],
                                                           dict[str, Any]]:
    """Does compressibility rescue separation when sizes are similar?

    Zhang et al. claim separation "even when the size of CTCs is similar to
    that of PBMCs". Density is set equal in every arm so the only acoustic
    difference besides size is compressibility. For each arm the best
    achievable setting of the 1 MHz amplitude is found; "separated" means some
    setting reaches capture >= 80 % with contamination <= 10 %.
    """
    jobs = [(arm, v, 17) for arm in CF_ARMS for v in v1]
    df = pd.DataFrame(parallel_map(cf_point, jobs))
    best: dict[str, Any] = {}
    for arm in CF_ARMS:
        best[arm] = operating_window(df[df["arm"] == arm], "V1", **CF_SUCCESS)
    names = list(CF_ARMS)
    size_only, measured, reversed_ = (best[a] for a in names)
    j = [best[a]["best_youden_percent"] for a in names]
    checks = [
        Check("CF", "size-only separation fails for < 2 um difference", "claim",
              _cf_text(size_only), "fails", passed=not size_only["window"]),
        Check("CF", "adding the measured compressibility difference makes it succeed",
              "claim", _cf_text(measured), "succeeds", passed=bool(measured["window"]),
              note="the paper's central claim; see REPORT.md for why the model cannot "
                   "support it with the Fig. 1 ordering"),
        Check("CF", "model follows Gor'kov: best J ordered reversed > size-only > measured",
              "trend", " > ".join(f"{x:.0f}" for x in (j[2], j[0], j[1])) + " %",
              "reversed > size-only > measured",
              passed=bool(j[2] > j[0] > j[1])),
    ]
    return df, checks, best


def _cf_text(win: dict[str, Any]) -> str:
    return (f"best J {win['best_youden_percent']:.0f} % "
            f"({win['best_capture_percent']:.0f} % / {win['best_contamination_percent']:.1f} % "
            f"at {win['best_setting']:g} Vpp)")


__all__ = [
    "reference", "base_config", "make_params", "calibration", "point", "sweep",
    "case_a", "case_b", "case_c", "case_d", "case_e", "contamination_sensitivity",
    "counterfactual", "durations_for", "fig8_case",
]
