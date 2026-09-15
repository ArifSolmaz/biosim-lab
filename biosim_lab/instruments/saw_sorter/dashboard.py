"""Interactive Panel dashboard for the SAW sorter.

Sliders re-run the (fast, analytic) simulation on every change, so the effect of
frequency, drive voltage and flow rate on separation efficiency and purity is
visible immediately.  FEM mode is available from a toggle but is deliberately
*not* wired to the live sliders: one Helmholtz solve takes seconds, which would
make the sliders feel broken.

The controls follow the operating mode: ``tassaw`` adds a tilt slider, and
``alternating_baw`` replaces frequency and voltage with the duration and
amplitude of each switched phase (:func:`build_baw_dashboard`), with the
trajectories and forces drawn against time and the phases shaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.viz.curves import (
    cross_section_figure,
    cumulative_count_figure,
    force_profile_figure,
    force_timeline_figure,
    lateral_timeline_figure,
    live_view_figure,
    outlet_histogram_figure,
    size_distribution_figure,
    trajectory_figure,
)
from biosim_lab.core.viz.dashboard import data_table, metric_tiles, require_panel, shell
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    node_positions,
    primary_radiation_force_1d,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

_TILE_FORMATS = {
    "efficiency_percent": "{:.1f} %",
    "purity_percent": "{:.1f} %",
    "live_purity_percent": "{:.1f} %",
    "viability_out_percent": "{:.1f} %",
    "enrichment_fold": "{:.2f}×",
    "n_collected": "{:d}",
    "n_cells": "{:d}",
    "capture_efficiency_percent": "{:.1f} %",
    "contamination_rate_percent": "{:.1f} %",
    "recovery_rate_percent": "{:.1f} %",
    "separation_distance_um": "{:.0f} µm",
}


def _force_profiles(sim: SAWSorterSimulation) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Force profile across the channel for each configured population."""
    x = np.linspace(0.0, sim.params.channel_width, 400)
    profiles: dict[str, np.ndarray] = {}
    for pop in sim.params.populations:
        cell = sim.cell_for(pop)
        volume = 4.0 / 3.0 * np.pi * cell.r**3
        profiles[pop.resolved_label()] = primary_radiation_force_1d(
            x,
            p0=sim.params.p0,
            volume=volume,
            kappa_f=sim.fluid.kappa,
            wavelength=sim.wavelength,
            phi=sim.phi_for(cell),
            node_offset=sim.node_offset,
        )
    return x, profiles


def build_dashboard(params: SAWSorterParams) -> Any:
    """Return a Panel view with live sliders bound to a re-running simulation."""
    import warnings

    if params.mode == "alternating_baw":
        return build_baw_dashboard(params)

    pn = require_panel()
    tilted = params.mode == "tassaw"
    tilt = pn.widgets.FloatInput(
        name="IDT tilt (deg)", value=params.tilt_angle_deg, step=0.5, start=-45.0, end=45.0,
        disabled=not tilted,
    )

    freq = pn.widgets.FloatSlider(
        name="frequency (MHz)", start=1.0, end=40.0, step=0.1,
        value=params.frequency / 1e6,
    )
    voltage = pn.widgets.FloatSlider(
        name="drive voltage (Vpp)", start=1.0, end=40.0, step=0.5,
        value=float(params.voltage_pp or 15.0),
    )
    flow = pn.widgets.FloatSlider(
        name="flow rate (µL/min)", start=0.5, end=60.0, step=0.5,
        value=params.flow_rate * 6e10,
    )
    length = pn.widgets.FloatSlider(
        name="active length (mm)", start=0.2, end=20.0, step=0.2,
        value=params.channel_length * 1e3,
    )
    count = pn.widgets.IntSlider(
        name="cells per population", start=50, end=1000, step=50,
        value=params.populations[0].count,
    )
    temperature = pn.widgets.FloatSlider(
        name="temperature (°C)", start=4.0, end=45.0, step=0.5,
        value=params.temperature - 273.15,
    )
    collection = pn.widgets.FloatSlider(
        name="collection outlet width (fraction)", start=0.05, end=0.9, step=0.05,
        value=params.collection_fraction,
    )

    def _simulate(f_mhz, v_pp, q_ul_min, l_mm, n, frac, temp_c, tilt_deg):
        updated = params.model_dump()
        if tilted and tilt_deg != 0.0:
            updated["tilt_angle_deg"] = tilt_deg
        updated.update(
            frequency=f_mhz * 1e6,
            voltage_pp=v_pp,
            pressure_amplitude=None,
            power_drive=None,
            flow_rate=q_ul_min / 6e10,
            channel_length=l_mm * 1e-3,
            collection_fraction=frac,
            temperature=temp_c + 273.15,
            populations=[{**p, "count": int(n)} for p in updated["populations"]],
        )
        new = SAWSorterParams.model_validate(updated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            sim = SAWSorterSimulation(new)
            return sim, sim.run()

    bound = pn.bind(_simulate, freq, voltage, flow, length, count, collection, temperature,
                    tilt)

    def _tiles(result):
        _, outcome = result
        return metric_tiles(
            outcome.metrics,
            ["efficiency_percent", "purity_percent", "live_purity_percent",
             "enrichment_fold", "viability_out_percent", "n_collected"]
            + (["separation_distance_um"] if tilted else []),
            _TILE_FORMATS,
        )

    def _live_view(result):
        sim, outcome = result
        return pn.pane.Plotly(
            live_view_figure(
                outcome.tracks.trajectories,
                channel_width=sim.params.channel_width,
                channel_length=sim.params.channel_length,
                node_positions=node_positions(
                    sim.params.channel_width, sim.wavelength, node_offset=sim.node_offset
                ),
                alive=outcome.cells["alive"].to_numpy(),
                collection_bounds=sim.collection_bounds,
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=580,
        )

    def _cross_section(result):
        sim, outcome = result
        return pn.pane.Plotly(
            cross_section_figure(
                outcome.tracks.trajectories,
                channel_width=sim.params.channel_width,
                channel_height=sim.params.channel_height,
                channel_length=sim.params.channel_length,
                alive=outcome.cells["alive"].to_numpy(),
                node_positions=node_positions(
                    sim.params.channel_width, sim.wavelength, node_offset=sim.node_offset
                ),
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=320,
        )

    def _live_count(result):
        sim, outcome = result
        return pn.pane.Plotly(
            cumulative_count_figure(
                outcome.tracks.trajectories, outcome.cells,
                channel_length=sim.params.channel_length,
                collection_bounds=sim.collection_bounds,
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=420,
        )

    def _trajectories(result):
        sim, outcome = result
        return pn.pane.Plotly(
            trajectory_figure(
                outcome.tracks.trajectories,
                node_positions=node_positions(
                    sim.params.channel_width, sim.wavelength, node_offset=sim.node_offset
                ),
                channel_width=sim.params.channel_width,
                channel_length=sim.params.channel_length,
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=440,
        )

    def _histogram(result):
        sim, outcome = result
        return pn.pane.Plotly(
            outlet_histogram_figure(
                outcome.cells,
                channel_width=sim.params.channel_width,
                collection_bounds=sim.collection_bounds,
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=440,
        )

    def _forces(result):
        sim, _ = result
        x, profiles = _force_profiles(sim)
        return pn.pane.Plotly(
            force_profile_figure(
                x,
                profiles,
                node_positions=node_positions(
                    sim.params.channel_width, sim.wavelength, node_offset=sim.node_offset
                ),
            ),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=440,
        )

    def _sizes(result):
        _, outcome = result
        return pn.pane.Plotly(
            size_distribution_figure(outcome.cells),
            config={"displayModeBar": False},
            sizing_mode="stretch_width",
            height=440,
        )

    def _table(result):
        sim, outcome = result
        rows = []
        for label, stats in outcome.metrics["per_population"].items():
            rows.append({"population": label, **stats})
        summary = pd.DataFrame(rows)
        return data_table(summary, height=220, page_size=10)

    def _diagnostics(result):
        sim, outcome = result
        d = outcome.diagnostics
        nodes = ", ".join(f"{v * 1e6:.0f}" for v in d["node_positions_m"])
        warn = ""
        if len(d["node_positions_m"]) > 1:
            suggestion = sim.saw_velocity / (2 * sim.params.channel_width) / 1e6
            warn = (
                f"<br><b>{len(d['node_positions_m'])} pressure nodes</b> across the channel — "
                f"two-outlet sorting assumes one. Try {suggestion:.2f} MHz."
            )
        return pn.pane.HTML(
            f"""
            <div style="font-size:12px;line-height:1.7">
              SAW wavelength <b>{d['saw_wavelength_m'] * 1e6:.1f} µm</b> ·
              node spacing <b>{d['node_spacing_m'] * 1e6:.1f} µm</b> ·
              nodes at x = <b>{nodes} µm</b><br>
              <b>{d['temperature_C']:.1f} °C</b> ·
              viscosity <b>{d['viscosity_Pa_s'] * 1e3:.3f} mPa·s</b> ·
              p₀ = <b>{d['pressure_amplitude_Pa'] / 1e6:.3f} MPa</b> ·
              mean flow <b>{d['mean_velocity_m_s'] * 1e3:.2f} mm/s</b> ·
              transit <b>{d['transit_time_s']:.2f} s</b> ·
              Re<sub>channel</sub> = <b>{outcome.metrics['channel_reynolds']:.3g}</b>
              {warn}
            </div>
            """
        )

    return shell(
        "SAW acoustophoretic cell sorter",
        "Standing surface-acoustic-wave separation — Gor'kov radiation force with "
        "analytic Poiseuille flow. Move a slider to re-run the simulation.",
        controls=[
            pn.Column(freq, voltage),
            pn.Column(flow, length),
            pn.Column(count, collection),
            pn.Column(temperature, *([tilt] if tilted else [])),
        ],
        tiles=pn.Column(
            pn.bind(_tiles, bound), pn.bind(_diagnostics, bound), sizing_mode="stretch_width"
        ),
        tabs=[
            ("Live view", pn.bind(_live_view, bound)),
            ("Cross-section", pn.bind(_cross_section, bound)),
            ("Live count", pn.bind(_live_count, bound)),
            ("Trajectories", pn.bind(_trajectories, bound)),
            ("Outlet histogram", pn.bind(_histogram, bound)),
            ("Force profile", pn.bind(_forces, bound)),
            ("Size distribution", pn.bind(_sizes, bound)),
            ("Per-population table", pn.bind(_table, bound)),
        ],
        footer="biosim-lab · MIT · Gor'kov force doi:10.1039/c2lc21068a · "
        "SSAW node spacing doi:10.1039/b910595f",
    )


def build_baw_dashboard(params: SAWSorterParams) -> Any:
    """Live dashboard for ``mode="alternating_baw"``: one slider per phase knob.

    Two simulations run per change: the population run behind the metric tiles
    and outlet histogram, and a small fixed-entry run of mean-sized cells for
    the time plots, which read like Zhang et al. 2023 Fig. 8
    (doi:10.3390/ijms24043338).
    """
    import warnings

    pn = require_panel()
    sw = params.switching
    assert sw is not None
    ph1, ph3 = sw.phases[0], sw.phases[1]
    t1 = pn.widgets.FloatSlider(name=f"{ph1.resolved_label()} duration (s)", start=0.1,
                                end=3.0, step=0.05, value=ph1.duration)
    v1 = pn.widgets.FloatSlider(name=f"{ph1.resolved_label()} amplitude (Vpp)", start=1.0,
                                end=30.0, step=0.5, value=ph1.voltage_pp)
    t3 = pn.widgets.FloatSlider(name=f"{ph3.resolved_label()} duration (s)", start=0.1,
                                end=3.0, step=0.05, value=ph3.duration)
    v3 = pn.widgets.FloatSlider(name=f"{ph3.resolved_label()} amplitude (Vpp)", start=10.0,
                                end=200.0, step=5.0, value=ph3.voltage_pp)
    flow = pn.widgets.FloatSlider(name="total flow rate (µL/h)", start=50.0, end=1200.0,
                                  step=25.0, value=params.flow_rate * 3.6e12)
    count = pn.widgets.IntSlider(name="cells per population", start=20, end=400, step=20,
                                 value=min(100, params.populations[0].count))

    def _with(t1_s, v1_vpp, t3_s, v3_vpp, q_ul_h, n, **extra):
        data = params.model_dump()
        phases = [dict(p) for p in data["switching"]["phases"]]
        phases[0].update(duration=t1_s, voltage_pp=v1_vpp)
        phases[1].update(duration=t3_s, voltage_pp=v3_vpp)
        data["switching"] = {**data["switching"], "phases": phases,
                             **extra.pop("switching", {})}
        data.update(flow_rate=q_ul_h / 3.6e12,
                    populations=[{**p, "count": int(n)} for p in data["populations"]],
                    **extra)
        return SAWSorterParams.model_validate(data)

    def _simulate(t1_s, v1_vpp, t3_s, v3_vpp, q_ul_h, n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            try:
                full = _with(t1_s, v1_vpp, t3_s, v3_vpp, q_ul_h, n)
                sim = SAWSorterSimulation(full)
                outcome = sim.run()
                demo_params = _with(
                    t1_s, v1_vpp, t3_s, v3_vpp, q_ul_h, 8,
                    switching={"entry": "fixed", "entry_time_in_cycle": 0.0},
                    inlet_x=list(np.linspace(0.02, 0.31, 8)), record_forces=True,
                    n_time_samples=1500,
                )
                demo = SAWSorterSimulation(demo_params).run()
            except ValueError as exc:  # ConfigurationError, e.g. < 2 cycles
                return None, None, None, str(exc)
        return sim, outcome, demo, ""

    bound = pn.bind(_simulate, t1, v1, t3, v3, flow, count)

    def _tiles(result):
        sim, outcome, _, error = result
        if error:
            return pn.pane.Alert(error, alert_type="warning")
        sw_rep = outcome.diagnostics["switching"]
        rule = sw_rep["separation_rule"]
        rule_txt = " · ".join(
            f"{k} moves {v['displacement_um']:.0f} µm" for k, v in rule["populations"].items())
        return pn.Column(
            metric_tiles(outcome.metrics, ["capture_efficiency_percent",
                                           "contamination_rate_percent", "n_collected",
                                           "n_cells"], _TILE_FORMATS),
            pn.pane.HTML(
                f"<div style='font-size:12px;line-height:1.7'>cycle "
                f"<b>{sw_rep['period_s']:.2f} s</b> · fastest cell sees "
                f"<b>{sw_rep['cycles_seen']['fastest_cell']:.1f}</b> cycles · W/6 rule: "
                f"{rule_txt} vs threshold <b>{rule['threshold_um']:.0f} µm</b> — "
                f"<b>{'satisfied' if rule['satisfied'] else 'NOT satisfied'}</b></div>"),
        )

    def _timeline(result):
        sim, _, demo, error = result
        if error:
            return pn.pane.Markdown("")
        schedule = sim.schedule
        t_max = 3.2 * schedule.period
        return pn.pane.Plotly(lateral_timeline_figure(
            demo.tracks.trajectories, channel_width=sim.params.channel_width,
            timeline=schedule.timeline(0.0, t_max), t_max=t_max,
            reference_lines=[(1 / 6, "W/6"), (1 / 3, "W/3"), (0.5, "W/2")],
            title="Mean-sized cells entering at the start of the first phase"),
            sizing_mode="stretch_width", height=460, config={"displayModeBar": False})

    def _force(result):
        sim, _, demo, error = result
        if error:
            return pn.pane.Markdown("")
        labels = np.asarray(demo.forces["label"].values, dtype=str)
        pick = [int(np.flatnonzero(labels == lab)[4]) for lab in dict.fromkeys(labels)]
        t_max = 3.2 * sim.schedule.period
        return pn.pane.Plotly(force_timeline_figure(
            demo.forces, particles=pick, timeline=sim.schedule.timeline(0.0, t_max),
            t_max=t_max), sizing_mode="stretch_width", height=460,
            config={"displayModeBar": False})

    def _histogram(result):
        sim, outcome, _, error = result
        if error:
            return pn.pane.Markdown("")
        return pn.pane.Plotly(outlet_histogram_figure(
            outcome.cells, channel_width=sim.params.channel_width,
            collection_bounds=sim.collection_bounds),
            sizing_mode="stretch_width", height=440, config={"displayModeBar": False})

    return shell(
        "Alternating-frequency BAW cell sorter",
        "One piezoceramic switched between the 1 MHz and 3 MHz resonances of the channel "
        "(doi:10.3390/ijms24043338). Move a slider to re-run.",
        controls=[pn.Column(t1, v1), pn.Column(t3, v3), pn.Column(flow, count)],
        tiles=pn.bind(_tiles, bound),
        tabs=[
            ("Trajectories vs time", pn.bind(_timeline, bound)),
            ("Forces vs time", pn.bind(_force, bound)),
            ("Outlet histogram", pn.bind(_histogram, bound)),
        ],
        footer="biosim-lab · MIT · force doi:10.1039/c2lc21068a · trajectory "
        "doi:10.1039/b920376a · device doi:10.3390/ijms24043338",
    )


__all__ = ["build_dashboard", "build_baw_dashboard"]
