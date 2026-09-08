"""Interactive Panel dashboard for the SAW sorter.

Sliders re-run the (fast, analytic) simulation on every change, so the effect of
frequency, drive voltage and flow rate on separation efficiency and purity is
visible immediately.  FEM mode is available from a toggle but is deliberately
*not* wired to the live sliders: one Helmholtz solve takes seconds, which would
make the sliders feel broken.
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
}


def _force_profiles(sim: SAWSorterSimulation) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Force profile across the channel for each configured population."""
    from biosim_lab.core.materials import get_cell

    x = np.linspace(0.0, sim.params.channel_width, 400)
    profiles: dict[str, np.ndarray] = {}
    for pop in sim.params.populations:
        cell = get_cell(pop.cell_type)
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

    pn = require_panel()

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

    def _simulate(f_mhz, v_pp, q_ul_min, l_mm, n, frac, temp_c):
        updated = params.model_dump()
        updated.update(
            frequency=f_mhz * 1e6,
            voltage_pp=v_pp,
            pressure_amplitude=None,
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

    bound = pn.bind(_simulate, freq, voltage, flow, length, count, collection, temperature)

    def _tiles(result):
        _, outcome = result
        return metric_tiles(
            outcome.metrics,
            ["efficiency_percent", "purity_percent", "live_purity_percent",
             "enrichment_fold", "viability_out_percent", "n_collected"],
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
            pn.Column(temperature),
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


__all__ = ["build_dashboard"]
