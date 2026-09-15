"""Run benchmark_01 (Li et al. 2015, taSSAW) and write its results and figures.

    python -m benchmarks.benchmark_01_tassaw.run            # full, ~5 min on 8 cores
    python -m benchmarks.benchmark_01_tassaw.run --quick    # coarse grids, ~1 min
    python -m benchmarks.benchmark_01_tassaw.run --no-calibrate   # reuse config.yaml's value

Outputs land in ``benchmarks/benchmark_01_tassaw/results/``: one CSV per case,
``comparison.csv`` (every Check), ``summary.json`` (read by
``benchmarks/make_report.py``) and ``figures/*.{html,png}``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from benchmarks.benchmark_01_tassaw import cases
from benchmarks.common import Check, checks_table, save_figure, write_summary
from biosim_lab.core.viz.theme import SEQUENTIAL_BLUE, color_for, plotly_layout

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"

QUICK = {
    "tilts": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 45.0),
    "flows": (25.0, 75.0, 125.0),
    "lengths": (2, 5, 8, 10, 12, 20, 30),
    "powers": (31.0, 33.0, 35.0, 37.0, 39.0),
    "count": 120,
    "n": 8,
}


def _line_fig(title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plotly_layout(title=title, xaxis_title=x_title, yaxis_title=y_title))
    return fig


def figure_2a(curves: pd.DataFrame, ref: dict[str, Any]) -> go.Figure:
    fig = _line_fig("Li 2015 Fig. 2A — separation distance against tilt (35 dBm)",
                    "IDT tilt angle θ (deg)", "separation distance ΔY (µm)")
    flows = sorted(curves["flow_ul_min"].unique())
    # Flow rate is ordered, so it gets the sequential ramp (light = slow), not hues.
    ramp = [SEQUENTIAL_BLUE[int(i)] for i in np.linspace(3, len(SEQUENTIAL_BLUE) - 1,
                                                            len(flows))]
    for colour, q in zip(ramp, flows):
        grp = curves[curves["flow_ul_min"] == q]
        fig.add_trace(go.Scatter(x=grp["tilt_deg"], y=grp["separation_distance_um"],
                                 mode="lines+markers", name=f"{q:g} µL/min",
                                 line={"color": colour, "width": 2}))
    r = ref["results"]
    fig.add_trace(go.Scatter(
        x=[r["fig2a_optimum_tilt_deg"]["value"]],
        y=[r["fig2a_separation_distance_um"]["value"]],
        mode="markers", name="stated in text: ~600 µm at ~5° (75 µL/min)",
        marker={"symbol": "x", "size": 13, "color": "#0b0b0b"}))
    return fig


def figure_2b(curve: pd.DataFrame, ref: dict[str, Any]) -> go.Figure:
    fig = _line_fig("Li 2015 Fig. 2B — separation distance against IDT length "
                    "(35 dBm, 75 µL/min, 5°)", "IDT length (mm)", "separation distance ΔY (µm)")
    lo, hi = ref["results"]["fig2b_optimum_idt_length_mm"]["value"]
    fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(82,81,78,0.12)", line_width=0,
                  annotation_text="stated maximum: 8–10 mm", annotation_position="top left")
    fig.add_trace(go.Scatter(x=curve["idt_length_mm"], y=curve["separation_distance_um"],
                             mode="lines+markers", name="this model",
                             line={"color": color_for("target"), "width": 2}))
    return fig


def figure_3(df: pd.DataFrame) -> go.Figure:
    fig = _line_fig("Li 2015 Fig. 3 — sorting performance against power (MCF-7 / WBC)",
                    "RF power (dBm)", "percent")
    fig.add_trace(go.Scatter(x=df["power_dbm"], y=df["recovery_rate_percent"],
                             mode="lines+markers", name="MCF-7 recovery",
                             line={"color": color_for("MCF-7"), "width": 2}))
    fig.add_trace(go.Scatter(x=df["power_dbm"], y=df["background_removal_percent"],
                             mode="lines+markers", name="WBC removal",
                             line={"color": color_for("WBC"), "width": 2}))
    return fig


def figure_table1(df: pd.DataFrame, ref: dict[str, Any], title: str) -> go.Figure:
    fig = _line_fig(title, "cell line", "recovery rate (%)")
    table = ref["results"]["table1"]
    paper = [float(np.mean([r[3] for r in table[line]])) for line in df["line"]]
    fig.add_trace(go.Bar(x=df["line"], y=paper, name="paper (Table 1 mean)",
                         marker={"color": "#9ec5f4"}))
    fig.add_trace(go.Bar(x=df["line"], y=df["recovery_rate_percent"], name="this model",
                         marker={"color": color_for("target")}))
    fig.add_hrect(y0=83, y1=96, fillcolor="rgba(82,81,78,0.10)", line_width=0,
                  annotation_text="stated band 83–96 %", annotation_position="top left")
    fig.update_layout(barmode="group")
    return fig


def figure_beads(df: pd.DataFrame) -> go.Figure:
    fig = _line_fig("9.9 µm vs 7.3 µm polystyrene on the Li 2015 geometry",
                    "RF power (dBm)", "percent")
    fig.add_trace(go.Scatter(x=df["power_dbm"], y=df["recovery_rate_percent"],
                             mode="lines+markers", name="9.9 µm recovery",
                             line={"color": color_for("ps_bead"), "width": 2}))
    fig.add_trace(go.Scatter(x=df["power_dbm"], y=df["background_removal_percent"],
                             mode="lines+markers", name="7.3 µm removal",
                             line={"color": color_for("lipid"), "width": 2}))
    fig.add_hline(y=97, line={"dash": "dot", "color": "#52514e"},
                  annotation_text="stated ≥ 97 % (earlier device)")
    return fig


def reanchored(power_df: pd.DataFrame, p_ref: float, count: int) -> dict[str, Any]:
    """Table 1 again, at the power where this model's WBC removal is the stated 90 %.

    A sensitivity, not a comparison: it asks how much of the Table 1 deviation
    is one offset between the paper's simulated and experimental power scales.
    """
    d = power_df.sort_values("power_dbm")
    rem = d["background_removal_percent"].to_numpy()
    pw = d["power_dbm"].to_numpy()
    # removal falls with power: interpolate on the reversed arrays
    p90 = float(np.interp(90.0, rem[::-1], pw[::-1]))
    df, _ = cases.case_e(p_ref, count=count, power=p90)
    return {"power_dbm": p90, "offset_db": 37.5 - p90, "table": df}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse grids, fewer cells")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="use config.yaml's reference_pressure instead of refitting")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    q = QUICK if args.quick else {}
    out: Path = args.out
    figs = out / "figures"
    t0 = time.time()
    ref = cases.reference()

    configured = float(cases.make_params().power_drive.reference_pressure)  # type: ignore[union-attr]
    if args.no_calibrate or args.quick:
        calib = {"reference_pressure_Pa": configured, "method": "taken from config.yaml"}
    else:
        calib = cases.calibrate()
    p_ref = float(calib["reference_pressure_Pa"])
    print(f"[01] reference pressure {p_ref / 1e6:.4f} MPa "
          f"(config.yaml: {configured / 1e6:.4f})")

    n = q.get("n", 12)
    curves, ca = cases.case_a(p_ref, n=n, tilts=q.get("tilts", cases.TILTS),
                              flows=q.get("flows", cases.FLOWS_UL_MIN))
    s2, cs2 = cases.case_s2(p_ref, n=n)
    curve_b, cb = cases.case_b(p_ref, n=n, lengths=q.get("lengths", cases.IDT_LENGTHS_MM))
    count = q.get("count", 300)
    power_df, cc = cases.case_c(p_ref, powers=q.get("powers", cases.POWERS_DBM), count=count)
    beads, cd = cases.case_d(p_ref, powers=q.get("powers", cases.BEAD_POWERS_DBM), count=count)
    table1, ce = cases.case_e(p_ref, count=count)
    anchor = reanchored(power_df, p_ref, count)

    checks: list[Check] = [*ca, *cs2, *cb, *cc, *cd, *ce]
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {"fig2a_tilt": curves, "figS2_power": s2, "fig2b_idt_length": curve_b,
                     "fig3_power": power_df, "beads": beads, "table1": table1,
                     "table1_reanchored": anchor["table"]}.items():
        df.drop(columns=[c for c in df.columns if c == "populations"]).to_csv(
            out / f"{name}.csv", index=False)

    figures = {
        "fig2a": save_figure(figure_2a(curves, ref), figs, "fig2a_tilt"),
        "fig2b": save_figure(figure_2b(curve_b, ref), figs, "fig2b_idt_length"),
        "fig3": save_figure(figure_3(power_df), figs, "fig3_power"),
        "table1": save_figure(figure_table1(table1, ref, "Li 2015 Table 1 — rare-cell recovery "
                                            "at 37.5 dBm"), figs, "table1"),
        "table1_reanchored": save_figure(figure_table1(
            anchor["table"], ref,
            f"Table 1 re-anchored: {anchor['power_dbm']:.1f} dBm (model WBC removal = 90 %)"),
            figs, "table1_reanchored"),
        "beads": save_figure(figure_beads(beads), figs, "beads"),
    }
    payload = {
        "benchmark": "benchmark_01_tassaw",
        "paper": ref["paper"],
        "quick": bool(args.quick),
        "calibration": calib,
        "configured_reference_pressure_Pa": configured,
        "reanchored": {"power_dbm": anchor["power_dbm"], "offset_db": anchor["offset_db"],
                       "table": anchor["table"][["line", "recovery_rate_percent",
                                                 "background_removal_percent"]]
                       .to_dict(orient="records")},
        "table1": table1[["line", "recovery_rate_percent", "background_removal_percent"]]
        .to_dict(orient="records"),
        "figures": figures,
        "runtime_s": time.time() - t0,
    }
    write_summary(out, payload, checks)
    print(checks_table(checks))
    print(f"[01] done in {payload['runtime_s']:.0f} s -> {out}")
    return payload


if __name__ == "__main__":
    main()
