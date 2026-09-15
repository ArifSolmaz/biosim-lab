"""Cases (a)-(e) of benchmark_01: Li et al. (2015), doi:10.1073/pnas.1504484112.

Everything is built from ``config.yaml`` (the device) and checked against
``reference.yaml`` (the paper's stated numbers). One number is fitted ---
the pressure amplitude at the paper's simulation power --- and it is fitted to
exactly one stated condition, the optimum tilt of Fig. 2A. All other paper
values are predictions of the fitted model.

The separation-distance calculations (Fig. 2A/B, Fig. S2) use *representative*
cells: mean-sized, monodisperse, a dozen of each at flux-weighted inlet
positions, as a design simulation does. The sorting-performance cases (Fig. 3,
Table 1, beads) use full log-normal populations, because recovery is a
statement about a distribution.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.common import Check, load_yaml, parallel_map
from biosim_lab.core.config import ExperimentConfig
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    max_trappable_tilt,
    pressure_from_rf_power,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
REFERENCE = HERE / "reference.yaml"

#: Tilt grid used to locate the optimum during calibration [deg].
CALIBRATION_TILTS = tuple(np.round(np.arange(1.0, 12.01, 0.5), 2))
FLOWS_UL_MIN = (25.0, 50.0, 75.0, 100.0, 125.0)
TILTS = tuple(sorted({*np.round(np.arange(0.5, 10.01, 0.5), 2), 12.0, 15.0, 20.0, 25.0,
                      30.0, 35.0, 40.0, 45.0}))
IDT_LENGTHS_MM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 25, 30)
POWERS_DBM = tuple(np.round(np.arange(29.0, 40.01, 1.0), 1))

MONODISPERSE = (
    "representative mean-sized cells: the separation-distance curves of Li 2015 Fig. 2 "
    "are a single-cell design calculation, not a population statistic"
)


def reference() -> dict[str, Any]:
    return load_yaml(REFERENCE)


def base_config() -> dict[str, Any]:
    """The ``params`` block of ``config.yaml``, as a plain dict."""
    return dict(ExperimentConfig.from_yaml(CONFIG).params)


def make_params(overrides: dict[str, Any] | None = None, *,
                representative: int | None = None) -> SAWSorterParams:
    """Device from ``config.yaml`` with *overrides* applied.

    ``power_drive`` keys given as ``power_dbm=...``/``reference_pressure=...``
    are merged into the drive block rather than replacing it.
    """
    data = base_config()
    ov = dict(overrides or {})
    drive = dict(data["power_drive"])
    for key in ("power_dbm", "reference_pressure"):
        if key in ov:
            drive[key] = ov.pop(key)
    data["power_drive"] = drive
    data.update(ov)
    if representative is not None:
        pops = []
        for pop in data["populations"]:
            pop = dict(pop, count=representative, diameter_cv=0.0)
            src = pop.get("override_source")
            pop["override_source"] = f"{src}; {MONODISPERSE}" if src else MONODISPERSE
            pops.append(pop)
        data["populations"] = pops
    return SAWSorterParams.model_validate(data)


def _run(params: SAWSorterParams) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        return SAWSorterSimulation(params).run().metrics


# ---------------------------------------------------------------------------
# separation distance (Fig. 2A, 2B, S2)
# ---------------------------------------------------------------------------


def separation_distance(args: tuple[float, float, float, float, float, int]) -> float:
    """``Delta Y`` [um] for ``(tilt_deg, flow_ul_min, power_dbm, idt_mm, p_ref_Pa, n)``."""
    tilt, flow, power, idt_mm, p_ref, n = args
    params = make_params({
        "tilt_angle_deg": -float(tilt),
        "flow_rate": f"{flow} uL/min",
        "channel_length": f"{idt_mm} mm",
        "power_dbm": float(power),
        "reference_pressure": float(p_ref),
    }, representative=int(n))
    return float(_run(params)["separation_distance_um"])


def tilt_curve(flow: float, p_ref: float, *, power: float = 35.0, idt_mm: float = 10.0,
               tilts: tuple[float, ...] = TILTS, n: int = 12) -> pd.DataFrame:
    jobs = [(t, flow, power, idt_mm, p_ref, n) for t in tilts]
    dy = parallel_map(separation_distance, jobs)
    return pd.DataFrame({"flow_ul_min": flow, "power_dbm": power, "tilt_deg": tilts,
                         "separation_distance_um": dy})


def optimum(curve: pd.DataFrame, x: str = "tilt_deg") -> tuple[float, float]:
    """``(x at the maximum, maximum Delta Y)`` of a curve."""
    i = int(curve["separation_distance_um"].idxmax())
    return float(curve.loc[i, x]), float(curve.loc[i, "separation_distance_um"])


def calibrate(*, n: int = 12, lo: float = 2.0e5, hi: float = 8.0e5,
              iterations: int = 9) -> dict[str, Any]:
    """Pressure at 35 dBm / 10 mm for which the Fig. 2A optimum at 75 uL/min is 5 deg.

    The optimum tilt grows with pressure (stronger trapping holds cells at
    steeper angles), so the pressures at which the grid optimum first reaches
    5 deg and first passes it bracket the answer; each edge is found by
    bisection and the midpoint is used. Only the stated optimum is fitted ---
    the ~600 um separation that the paper quotes alongside it is left as a
    prediction.
    """
    target = float(reference()["results"]["fig2a_optimum_tilt_deg"]["value"])

    def best_tilt(p: float) -> float:
        return optimum(tilt_curve(75.0, p, tilts=CALIBRATION_TILTS, n=n))[0]

    def edge(strict: bool) -> float:
        a, b = lo, hi
        for _ in range(iterations):
            mid = 0.5 * (a + b)
            t = best_tilt(mid)
            if (t > target) if strict else (t >= target):
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    first, past = edge(strict=False), edge(strict=True)
    p_star = 0.5 * (first + past)
    return {
        "reference_pressure_Pa": p_star,
        "bracket_Pa": [first, past],
        "target_optimum_tilt_deg": target,
        "tilt_grid_deg": list(CALIBRATION_TILTS),
        "reference_power_dbm": 35.0,
        "reference_idt_length_mm": 10.0,
        "method": "bisection on the grid optimum of Delta Y(theta) at 75 uL/min",
    }


def case_a(p_ref: float, *, n: int = 12, tilts: tuple[float, ...] = TILTS,
           flows: tuple[float, ...] = FLOWS_UL_MIN) -> tuple[pd.DataFrame, list[Check]]:
    """Fig. 2A: Delta Y against tilt at five flow rates."""
    ref = reference()["results"]
    curves = pd.concat([tilt_curve(q, p_ref, tilts=tilts, n=n) for q in flows],
                       ignore_index=True)
    opt = {q: optimum(curves[curves["flow_ul_min"] == q]) for q in flows}
    best_t = [opt[q][0] for q in flows]
    above = [q for q in sorted(flows) if q >= 75.0]
    checks = [
        Check("2A", "optimum tilt at 75 uL/min", "calibration", opt[75.0][0],
              ref["fig2a_optimum_tilt_deg"]["value"], "deg",
              note="fitted: the reference pressure was chosen to put it here"),
        Check("2A", "Delta Y at the optimum, 75 uL/min", "quantity", opt[75.0][1],
              ref["fig2a_separation_distance_um"]["value"], "um", tolerance=60.0,
              note="tolerance: 10 % of the stated value (not a percentage metric)",
              cause=(
                  "the lateral room is bounded by the 800 um channel: MCF-7 cells "
                  "entering across the ~200 um sample stream reach the far wall, so "
                  "the gap between population MEANS saturates near 500 um. Li's "
                  "2-D single-trajectory model measured it between two cells from "
                  "one inlet point; the sample-stream width (2.5:1 sheath ratio) is "
                  "included here and not there. Table S1 cell properties unknown."
              )),
        Check("2A", "optimum tilt falls as flow rises", "trend",
              ", ".join(f"{t:g}" for t in best_t) + " deg at "
              + ", ".join(f"{q:g}" for q in flows) + " uL/min",
              "decreasing",
              passed=bool(np.all(np.diff(best_t) <= 0.0) and best_t[0] > best_t[-1])),
        Check("2A", "above 75 uL/min Delta Y falls even at the optimum", "trend",
              " -> ".join(f"{opt[q][1]:.0f}" for q in above) + " um at "
              + ", ".join(f"{q:g}" for q in above) + " uL/min",
              "decreasing", passed=bool(len(above) > 1 and np.all(
                  np.diff([opt[q][1] for q in above]) < 0.0))),
    ]
    return curves, checks


def case_s2(p_ref: float, *, n: int = 12,
            powers: tuple[float, ...] = (33.0, 35.0, 37.0)) -> tuple[pd.DataFrame, list[Check]]:
    """Fig. S2 (text): at fixed flow, the optimum tilt increases with power."""
    curves = pd.concat([tilt_curve(75.0, p_ref, power=pw, tilts=CALIBRATION_TILTS, n=n)
                        for pw in powers], ignore_index=True)
    best = [optimum(curves[curves["power_dbm"] == pw])[0] for pw in powers]
    return curves, [Check(
        "S2", "optimum tilt rises with power", "trend",
        f"{', '.join(f'{t:g}' for t in best)} deg at {', '.join(f'{p:g}' for p in powers)} dBm",
        "increasing", passed=bool(np.all(np.diff(best) >= 0.0) and best[-1] > best[0]),
    )]


def case_b(p_ref: float, *, n: int = 12,
           lengths: tuple[float, ...] = IDT_LENGTHS_MM) -> tuple[pd.DataFrame, list[Check]]:
    """Fig. 2B: Delta Y against IDT length at 35 dBm, 75 uL/min, 5 deg."""
    ref = reference()["results"]["fig2b_optimum_idt_length_mm"]["value"]
    jobs = [(5.0, 75.0, 35.0, float(mm), p_ref, n) for mm in lengths]
    dy = parallel_map(separation_distance, jobs)
    curve = pd.DataFrame({"idt_length_mm": lengths, "separation_distance_um": dy})
    best_l, best_dy = optimum(curve, "idt_length_mm")
    # A plateau: every length within 2 % of the maximum is "at the maximum".
    plateau = curve.loc[curve["separation_distance_um"] >= 0.98 * best_dy, "idt_length_mm"]
    checks = [
        Check("2B", "IDT length at maximum Delta Y", "range", best_l, tuple(ref), "mm",
              note=f"lengths within 2 % of the maximum: {plateau.min():g}-{plateau.max():g} mm",
              cause=(
                  "the fall-off at long IDTs rests on the ASSUMED scaling p0^2 ~ 1/L "
                  "(the paper states that longer IDTs lower the energy density, not "
                  "how); a weaker dependence moves the maximum to longer IDTs"
              )),
        Check("2B", "an interior maximum exists", "trend",
              f"max at {best_l:g} mm", "interior",
              passed=bool(lengths[0] < best_l < lengths[-1])),
    ]
    return curve, checks


# ---------------------------------------------------------------------------
# sorting performance (Fig. 3, Table 1, beads)
# ---------------------------------------------------------------------------


def sort_point(args: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """Full-population run at ``overrides``; returns the metrics of interest."""
    overrides, seed = args
    params = make_params({**overrides, "seed": seed})
    m = _run(params)
    per = m["per_population"]
    return {
        **{k: v for k, v in overrides.items() if not isinstance(v, list)},
        "recovery_rate_percent": m["recovery_rate_percent"],
        "background_removal_percent": m["background_removal_percent"],
        "capture_efficiency_percent": m["capture_efficiency_percent"],
        "separation_distance_um": m["separation_distance_um"],
        **{f"collected_{k}": v["collected_fraction"] for k, v in per.items()},
    }


def case_c(p_ref: float, *, powers: tuple[float, ...] = POWERS_DBM,
           count: int = 300) -> tuple[pd.DataFrame, list[Check]]:
    """Fig. 3: recovery and WBC removal against power (trend only, as specified)."""
    ref = reference()["results"]
    pops = [dict(p, count=count) for p in base_config()["populations"]]
    jobs = [({"power_dbm": float(pw), "reference_pressure": p_ref, "populations": pops}, 7)
            for pw in powers]
    df = pd.DataFrame(parallel_map(sort_point, jobs))
    rec = df["recovery_rate_percent"].to_numpy()
    rem = df["background_removal_percent"].to_numpy()
    low = ref["fig3_low_power"]
    high = ref["fig3_high_power"]
    # The paper's low-power regime: WBC removal ~99 % while recovery is 60-80 %.
    in_low = (rem >= low["wbc_removal_percent"] - 2.0) & (rec >= 55.0) & (rec <= 85.0)
    # Its high-power regime: recovery > 90 % while removal has dropped to ~90 %.
    in_high = (rec > high["recovery_min_percent"]) & (rem < low["wbc_removal_percent"] - 2.0)
    checks = [
        Check("3", "recovery rises with power", "trend",
              f"{rec[0]:.0f} -> {rec[-1]:.0f} %", "increasing",
              passed=bool(np.all(np.diff(rec) >= -3.0) and rec[-1] > rec[0])),
        Check("3", "WBC removal falls with power", "trend",
              f"{rem[0]:.0f} -> {rem[-1]:.0f} %", "decreasing",
              passed=bool(np.all(np.diff(rem) <= 3.0) and rem[-1] < rem[0])),
        Check("3", "a low-power regime: removal ~99 %, recovery 60-80 %", "trend",
              _regime(df, in_low), "exists", passed=bool(in_low.any())),
        Check("3", "a high-power regime: recovery > 90 %, removal ~90 %", "trend",
              _regime(df, in_high), "exists", passed=bool(in_high.any())),
    ]
    return df, checks


def _regime(df: pd.DataFrame, mask: np.ndarray) -> str:
    if not mask.any():
        return "none"
    sel = df[mask]
    return (f"{sel['power_dbm'].min():g}-{sel['power_dbm'].max():g} dBm "
            f"(rec {sel['recovery_rate_percent'].min():.0f}-"
            f"{sel['recovery_rate_percent'].max():.0f} %, removal "
            f"{sel['background_removal_percent'].min():.0f}-"
            f"{sel['background_removal_percent'].max():.0f} %)")


BEAD_POWERS_DBM = tuple(np.round(np.arange(29.0, 40.01, 0.5), 1))


def case_d(p_ref: float, *, powers: tuple[float, ...] = BEAD_POWERS_DBM,
           count: int = 300) -> tuple[pd.DataFrame, list[Check]]:
    """Beads: the 10 um calibration bead, and 9.9 vs 7.3 um separation >= 97 %."""
    ref = reference()["results"]["beads_separation"]
    pops = [
        {"cell_type": "ps_9p9um", "count": count, "target": True, "label": "PS 9.9 um"},
        {"cell_type": "ps_7p3um", "count": count, "label": "PS 7.3 um"},
    ]
    jobs = [({"power_dbm": float(pw), "reference_pressure": p_ref, "populations": pops}, 11)
            for pw in powers]
    df = pd.DataFrame(parallel_map(sort_point, jobs))
    df["separation_efficiency_percent"] = np.minimum(
        df["recovery_rate_percent"], df["background_removal_percent"]
    )
    best = df.loc[df["separation_efficiency_percent"].idxmax()]

    # The 10 um calibration bead: can a node hold it at the design tilt?
    sim = SAWSorterSimulation(make_params({"reference_pressure": p_ref, "power_dbm": 35.0,
                                           "populations": [{"cell_type": "ps_10um",
                                                            "target": True}]}))
    from biosim_lab.core.materials import get_cell

    bead = get_cell("ps_10um")
    flow_speed = sim.params.flow_rate / (sim.params.channel_width * sim.params.channel_height)
    held = float(np.degrees(max_trappable_tilt(
        bead.r, p0=sim.params.p0, kappa_f=sim.fluid.kappa, wavelength=sim.wavelength,
        phi=sim.phi_for(bead), viscosity=sim.fluid.mu, flow_speed=flow_speed,
    )))
    checks = [
        Check("beads", "9.9 vs 7.3 um PS: best min(recovery, removal)", "claim",
              f"{best['separation_efficiency_percent']:.1f} % at {best['power_dbm']:g} dBm",
              f">= {ref['efficiency_min_percent']:g} %",
              passed=bool(best["separation_efficiency_percent"]
                          >= ref["efficiency_min_percent"]),
              note=("tested on the Li 2015 geometry: the Ding 2014 device the result comes "
                    "from is not specified in this paper, so this checks feasibility only")),
        Check("beads", "10 um PS bead held by a node at the 5 deg design tilt (35 dBm)",
              "claim", f"largest holdable tilt {held:.1f} deg", "held (deflected)",
              passed=bool(held > 5.0),
              note="Li calibrated their model on these beads (SI Fig. S1, not available), "
                   "which requires them to be deflected at the design point; this checks "
                   "only that, not the calibration data"),
    ]
    return df, checks


LINES = {
    # label: (cell_type, overrides)
    "MCF-7": ("mcf7", {}),
    "HeLa": ("hela", {
        "diameter": "16 um",
        "override_source": "Li 2015 p. 4975: cancer lines 'average diameters of 16 or "
        "20 um'; 16 um assigned to HeLa (ASSUMPTION: the paper does not map them)",
    }),
    "UACC903M-GFP": ("uacc903m", {}),
    "LNCaP": ("lncap", {}),
}


def case_e(p_ref: float, *, count: int = 300,
           power: float = 37.5) -> tuple[pd.DataFrame, list[Check]]:
    """Table 1: rare-cell recovery for four lines against leukocytes at ~37.5 dBm."""
    ref = reference()["results"]
    wbc = next(p for p in base_config()["populations"] if not p.get("target"))
    jobs = []
    for label, (key, extra) in LINES.items():
        target = {"cell_type": key, "count": count, "target": True, "label": label, **extra}
        jobs.append(({"power_dbm": power, "reference_pressure": p_ref,
                      "populations": [target, dict(wbc, count=count)]}, 23))
    rows = parallel_map(sort_point, jobs)
    df = pd.DataFrame(rows)
    df.insert(0, "line", list(LINES))
    band = tuple(ref["recovery_band_percent"]["value"])
    checks: list[Check] = []
    for label, row in zip(LINES, rows):
        table = ref["table1"][label]
        paper_mean = float(np.mean([r[3] for r in table]))
        checks.append(Check(
            "Table 1", f"{label} recovery vs paper mean of {len(table)} run(s)", "quantity",
            row["recovery_rate_percent"], round(paper_mean, 1), "%", tolerance=5.0,
            cause=_recovery_cause(label, row["recovery_rate_percent"], paper_mean),
        ))
        checks.append(Check(
            "Table 1", f"{label} recovery in the stated 83-96 % band", "range",
            row["recovery_rate_percent"], band, "%",
            cause=_recovery_cause(label, row["recovery_rate_percent"], paper_mean),
        ))
    removal = float(np.mean([r["background_removal_percent"] for r in rows]))
    checks.append(Check(
        "Table 1", "WBC removal (mean over the four runs)", "quantity", removal,
        ref["wbc_removal_percent"]["value"], "%", tolerance=5.0,
        cause=("leukocyte size spread is the library's 18 % CV around the stated 12 um "
               "(an ASSUMPTION); the paper attributes lost removal to large WBCs "
               "such as monocytes, so the tail of that distribution sets this number"),
    ))
    return df, checks


def _recovery_cause(label: str, ours: float, paper: float) -> str:
    size = ("cell diameter is an ASSUMPTION for this line (the paper gives only '16 or 20 "
            "um' without mapping them)") if label != "MCF-7" else (
            "MCF-7 diameter is the library's 18 um (Hartono 2011), between the paper's "
            "16 and 20 um")
    direction = "over" if ours > paper else "under"
    return (f"model {direction}-predicts; {size}; Table S1 compressibility unavailable; "
            "the divider position is an ASSUMPTION (channel midline)")


__all__ = [
    "reference", "base_config", "make_params", "separation_distance", "tilt_curve",
    "optimum", "calibrate", "case_a", "case_s2", "case_b", "case_c", "case_d",
    "case_e", "pressure_from_rf_power",
]
