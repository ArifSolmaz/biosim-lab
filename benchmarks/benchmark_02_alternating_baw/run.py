"""Run benchmark_02 (Zhang et al. 2023, alternating-frequency BAW) and write results.

    python -m benchmarks.benchmark_02_alternating_baw.run           # full, ~6 min on 8 cores
    python -m benchmarks.benchmark_02_alternating_baw.run --quick   # coarse grids

Outputs land in ``benchmarks/benchmark_02_alternating_baw/results/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from benchmarks.benchmark_02_alternating_baw import cases
from benchmarks.common import Check, checks_table, save_figure, write_summary
from biosim_lab.core.viz.curves import force_timeline_figure, lateral_timeline_figure
from biosim_lab.core.viz.theme import SEQUENTIAL_BLUE, color_for, plotly_layout
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterSimulation

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

QUICK = {
    "t3": (0.2, 0.6, 1.0, 1.4, 3.0),
    "t1": (0.2, 0.6, 0.8, 1.0, 1.4),
    "v1": (5.0, 7.0, 9.0, 11.0, 14.0, 22.0),
    "lengths": (20.0, 40.0),
    "cf_v1": (3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0, 17.0),
}


def _fig(title: str, x: str, y: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plotly_layout(title=title, xaxis_title=x, yaxis_title=y))
    return fig


def metric_figure(df: pd.DataFrame, key: str, title: str, x_title: str,
                  window: dict[str, Any] | None = None) -> go.Figure:
    fig = _fig(title, x_title, "percent")
    fig.add_trace(go.Scatter(x=df[key], y=df["capture_efficiency_percent"],
                             mode="lines+markers", name="capture efficiency (MCF7)",
                             line={"color": color_for("mcf7"), "width": 2}))
    fig.add_trace(go.Scatter(x=df[key], y=df["contamination_rate_percent"],
                             mode="lines+markers", name="PBMC contamination",
                             line={"color": color_for("pbmc"), "width": 2}))
    if window is not None:
        if window["window"]:
            lo, hi = window["window"]
            fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(82,81,78,0.12)", line_width=0,
                          annotation_text="operating window", annotation_position="top left")
        fig.add_vline(x=window["best_setting"], line={"dash": "dot", "color": "#52514e"},
                      annotation_text=f"best J = {window['best_youden_percent']:.0f} %",
                      annotation_position="bottom right")
    return fig


def flow_figure(df: pd.DataFrame, ref: dict[str, Any], length_mm: float) -> go.Figure:
    fig = _fig("Zhang 2023 Fig. 4 — A549 capture against flow rate", "total flow rate (µL/h)",
               "capture efficiency (%)")
    main = df[df["length_mm"] == length_mm].sort_values("flow_ul_h")
    paper = ref["results"]["fig4_flow"]["capture_percent"]
    fig.add_trace(go.Scatter(
        x=[float(k) for k in paper], y=[v["mean"] for v in paper.values()],
        error_y={"type": "data", "array": [v["sd"] for v in paper.values()]},
        mode="markers", name="paper (stated, mean ± SD)",
        marker={"symbol": "x", "size": 12, "color": "#0b0b0b"}))
    fig.add_trace(go.Scatter(x=main["flow_ul_h"], y=main["capture_efficiency_percent"],
                             mode="lines+markers", name=f"this model, {length_mm:g} mm region",
                             line={"color": color_for("a549"), "width": 2}))
    sens = df[df["flow_ul_h"] == 900.0].sort_values("length_mm")
    ramp = [SEQUENTIAL_BLUE[int(i)] for i in np.linspace(4, 12, len(sens))]
    for colour, (_, row) in zip(ramp, sens.iterrows()):
        if row["length_mm"] == length_mm:
            continue
        fig.add_trace(go.Scatter(x=[900.0], y=[row["capture_efficiency_percent"]],
                                 mode="markers",
                                 name=f"900 µL/h, {row['length_mm']:g} mm region",
                                 marker={"color": colour, "size": 10}))
    fig.update_xaxes(type="log")
    return fig


def table1_figure(windows: dict[str, Any], ref: dict[str, Any]) -> go.Figure:
    fig = _fig("Zhang 2023 Table 1 — at each line's best 1 MHz amplitude", "cell line",
               "percent")
    lines = list(windows)
    tab = ref["results"]["table1"]["lines"]
    fig.add_trace(go.Bar(x=lines, y=[tab[x]["capture"][0] for x in lines],
                         error_y={"type": "data",
                                  "array": [tab[x]["capture"][1] for x in lines]},
                         name="capture, paper", marker={"color": "#9ec5f4"}))
    fig.add_trace(go.Bar(x=lines, y=[windows[x]["best_capture_percent"] for x in lines],
                         name="capture, this model", marker={"color": color_for("mcf7")}))
    fig.add_trace(go.Bar(x=lines, y=[tab[x]["contamination"][0] for x in lines],
                         error_y={"type": "data",
                                  "array": [tab[x]["contamination"][1] for x in lines]},
                         name="contamination, paper", marker={"color": "#f5c3a9"}))
    fig.add_trace(go.Bar(x=lines, y=[windows[x]["best_contamination_percent"] for x in lines],
                         name="contamination, this model", marker={"color": color_for("pbmc")}))
    fig.update_layout(barmode="group")
    return fig


def counterfactual_figure(df: pd.DataFrame) -> go.Figure:
    fig = _fig("Counterfactual — CTC 12 µm vs patient PBMC 10.5 µm",
               "1 MHz amplitude (Vpp)", "Youden J = capture − contamination (%)")
    for i, (arm, grp) in enumerate(df.groupby("arm", sort=False)):
        g = grp.sort_values("V1")
        fig.add_trace(go.Scatter(
            x=g["V1"], y=g["capture_efficiency_percent"] - g["contamination_rate_percent"],
            mode="lines+markers", name=arm,
            line={"color": ("#52514e", color_for("target"), color_for("background"))[i % 3],
                  "width": 2}))
    fig.add_hline(y=70, line={"dash": "dot", "color": "#52514e"},
                  annotation_text="success: capture ≥ 80 %, contamination ≤ 10 %")
    return fig


def fig8_figures(outcomes: dict[str, Any], figs: Path) -> dict[str, Any]:
    params = cases.make_params()
    sim = SAWSorterSimulation(params)
    schedule = sim.schedule
    assert schedule is not None
    w = params.channel_width
    t1 = params.switching.phases[0].duration  # type: ignore[union-attr]
    t3 = params.switching.phases[1].duration  # type: ignore[union-attr]
    entries = {"a_start_of_1MHz": 0.0, "b_end_of_3MHz": t1 + 0.85 * t3}
    t_max = 3.4 * schedule.period
    out: dict[str, Any] = {}
    refs = [(1 / 6, "W/6"), (1 / 3, "W/3"), (0.5, "W/2"), (5 / 6, "5W/6")]
    for key, oc in outcomes.items():
        timeline = schedule.timeline(0.0, t_max, offset=entries[key])
        title = ("Fig. 8a — cells enter at the start of the 1 MHz phase" if key.startswith("a")
                 else "Fig. 8b — cells enter at the end of the 3 MHz phase")
        out[f"fig8{key[0]}"] = save_figure(
            lateral_timeline_figure(oc.tracks.trajectories, channel_width=w, timeline=timeline,
                                    reference_lines=refs, t_max=t_max, title=title),
            figs, f"fig8{key[0]}_trajectories")
        if key.startswith("b") and oc.forces is not None:
            labels = np.asarray(oc.forces["label"].values, dtype=str)
            pick = [int(np.flatnonzero(labels == lab)[len(np.flatnonzero(labels == lab)) // 2])
                    for lab in ("MCF7", "PBMC")]
            out["fig8c"] = save_figure(
                force_timeline_figure(oc.forces, particles=pick, timeline=timeline,
                                      t_max=t_max,
                                      title="Fig. 8c — radiation and drag force (entry at "
                                            "the end of 3 MHz)"),
                figs, "fig8c_forces")
            try:
                from biosim_lab.core.viz.fields3d import animate_trajectories

                gif = animate_trajectories(
                    oc.tracks.trajectories.sel(time=slice(0.0, t_max)),
                    figs / "fig8b_trajectories.gif",
                    channel_width=w, channel_length=params.channel_length, fps=15)
                out["animation"] = str(gif)
            except Exception as exc:  # noqa: BLE001 - PyVista rendering is optional
                out["animation_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse grids")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    q = QUICK if args.quick else {}
    out: Path = args.out
    figs = out / "figures"
    t0 = time.time()
    ref = cases.reference()

    calib = cases.calibration()
    params = cases.make_params()
    configured = [ph.reference_energy_density for ph in params.switching.phases]  # type: ignore[union-attr]
    calib_checks = [
        Check("cal", "E_1MHz in config.yaml vs design-rule calibration", "calibration",
              configured[0], round(calib["E1_J_m3"], 2), "J/m^3",
              note="geometric mean of the W/6-rule interval " +
                   f"[{calib['rule']['target_minimum_J_m3']:.1f}, "
                   f"{calib['rule']['background_maximum_J_m3']:.1f}] J/m^3"),
        Check("cal", "E_3MHz in config.yaml vs return-time calibration", "calibration",
              configured[1], round(calib["E3_J_m3"], 2), "J/m^3", note=calib["E3_rule"]),
    ]
    print(f"[02] E1 {calib['E1_J_m3']:.2f} J/m^3, E3 {calib['E3_J_m3']:.2f} J/m^3 "
          f"(config: {configured[0]:.2f}, {configured[1]:.2f})")

    da, ca = cases.case_a(q.get("t3", cases.T3_SWEEP_S))
    db, cb, windows_b = cases.case_b(q.get("t1", cases.T1_SWEEP_S),
                                     q.get("v1", cases.V1_SWEEP_VPP))
    oc, cc = cases.case_c()
    dd, cd = cases.case_d(lengths=q.get("lengths", cases.LENGTHS_MM))
    de, ce, windows_e = cases.case_e(q.get("v1", cases.V1_SWEEP_VPP))
    sens = cases.contamination_sensitivity()
    dcf, ccf, best_cf = cases.counterfactual(q.get("cf_v1", cases.CF_V1))
    rule = SAWSorterSimulation(params).switching_report()["separation_rule"]
    rule_check = Check(
        "f", "design rule: MCF7 moves > W/6, PBMC < W/6 in the 1 MHz phase", "claim",
        ", ".join(f"{k} {v['displacement_um']:.0f} um" for k, v in rule["populations"].items())
        + f" (W/6 = {rule['threshold_um']:.0f} um)", "satisfied", passed=rule["satisfied"],
        note="satisfied by construction of E_1MHz; the integrator-level test is "
             "tests/test_baw.py::test_design_rule_holds_in_the_integrated_trajectory")

    checks: list[Check] = [*calib_checks, *ca, *cb, *cc, *cd, *ce, rule_check, *ccf]
    out.mkdir(parents=True, exist_ok=True)
    tables = {"fig2a_T3": da, "fig2b_T1": db["T1"], "fig2c_V1": db["V1"], "fig4_flow": dd,
              "table1_V1_sweep": de, "contamination_sensitivity": sens, "counterfactual": dcf}
    for name, df in tables.items():
        df.drop(columns=[c for c in df.columns if c == "populations"]).to_csv(
            out / f"{name}.csv", index=False)

    length0 = float(str(cases.base_config()["channel_length"]).split()[0])
    figures = {
        "fig2a": save_figure(metric_figure(da, "T3", "Zhang 2023 Fig. 2a — 3 MHz duration "
                                           "(1 MHz: 0.8 s, 9 Vpp)", "3 MHz duration (s)"),
                             figs, "fig2a_T3"),
        "fig2b": save_figure(metric_figure(db["T1"], "T1",
                                           "Zhang 2023 Fig. 2b — 1 MHz duration "
                                           "(9 Vpp; 3 MHz: 1.4 s)", "1 MHz duration (s)",
                                           windows_b["T1"]), figs, "fig2b_T1"),
        "fig2c": save_figure(metric_figure(db["V1"], "V1",
                                           "Zhang 2023 Fig. 2c — 1 MHz amplitude "
                                           "(0.8 s; 3 MHz: 1.4 s)", "1 MHz amplitude (Vpp)",
                                           windows_b["V1"]), figs, "fig2c_V1"),
        "fig4": save_figure(flow_figure(dd, ref, length0), figs, "fig4_flow"),
        "table1": save_figure(table1_figure(windows_e, ref), figs, "table1"),
        "counterfactual": save_figure(counterfactual_figure(dcf), figs, "counterfactual"),
        **fig8_figures(oc, figs),
    }
    payload = {
        "benchmark": "benchmark_02_alternating_baw",
        "paper": ref["paper"],
        "quick": bool(args.quick),
        "calibration": calib,
        "windows": {"fig2b_T1": windows_b["T1"], "fig2c_V1": windows_b["V1"], **windows_e},
        "contamination_sensitivity": sens.to_dict(orient="records"),
        "counterfactual": best_cf,
        "flow_rates": dd.drop(columns=[c for c in dd.columns if c == "populations"])
        .to_dict(orient="records"),
        "separation_rule": rule,
        "figures": figures,
        "runtime_s": time.time() - t0,
    }
    write_summary(out, payload, checks)
    print(checks_table(checks))
    print(f"[02] done in {payload['runtime_s']:.0f} s -> {out}")
    return payload


if __name__ == "__main__":
    main()
