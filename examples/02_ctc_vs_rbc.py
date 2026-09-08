"""Example 2 — separating circulating tumour cells from erythrocytes.

The reference scenario: MCF-7 (r ~ 9 um) mixed with red blood cells
(r ~ 2.8 um volume-equivalent) enter a 300 x 50 um channel hydrodynamically
focused against the two side walls.  A single-node standing SAW pushes both
towards the centre, but the radiation force scales with ``r^3``, so the tumour
cells arrive at the central outlet within the 2 mm active length and the
erythrocytes do not.

Writes ``assets/02_ctc_vs_rbc.html`` (interactive) and a PNG when kaleido is
available.

Run::

    python examples/02_ctc_vs_rbc.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from biosim_lab.core.io import save_result
from biosim_lab.core.materials import get_cell
from biosim_lab.core.plugin import InstrumentResult, RegimeWarning
from biosim_lab.core.viz.curves import (
    cross_section_figure,
    cumulative_count_figure,
    force_profile_figure,
    live_view_figure,
    outlet_histogram_figure,
    size_distribution_figure,
    trajectory_figure,
)
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    node_positions,
    primary_radiation_force_1d,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"


def build_params() -> SAWSorterParams:
    return SAWSorterParams(
        frequency="6.632 MHz",     # c_SAW / (2 W): exactly one node, centred
        voltage_pp="15 V",         # -> p0 = 0.45 MPa via the documented calibration
        channel_width="300 um",
        channel_height="50 um",
        channel_length="2 mm",
        flow_rate="5 uL/min",
        fluid="water",
        substrate="linbo3_128yx",
        inlet="sheath_sides",
        collection_fraction=1 / 3,
        mode="analytic",
        populations=[
            {"cell_type": "mcf7", "count": 400, "target": True},
            {"cell_type": "rbc", "count": 400, "target": False},
        ],
        seed=20260907,
    )


def main() -> None:
    params = build_params()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RegimeWarning)
        sim = SAWSorterSimulation(params)
        outcome = sim.run()

    m = outcome.metrics
    print("=" * 72)
    print("MCF-7 vs erythrocytes — standing SAW separation")
    print("=" * 72)
    d = outcome.diagnostics
    print(f"  SAW wavelength      {d['saw_wavelength_m'] * 1e6:8.1f} um")
    print(f"  pressure node(s) at "
          f"{', '.join(f'{v * 1e6:.0f}' for v in d['node_positions_m'])} um")
    print(f"  drive amplitude p0  {d['pressure_amplitude_Pa'] / 1e6:8.3f} MPa")
    print(f"  mean flow speed     {d['mean_velocity_m_s'] * 1e3:8.2f} mm/s")
    print(f"  transit time        {d['transit_time_s']:8.2f} s")
    print(f"  channel Reynolds    {m['channel_reynolds']:8.3g}")
    print()
    print(f"  temperature         {d['temperature_C']:8.1f} C")
    print(f"  viscosity           {d['viscosity_Pa_s'] * 1e3:8.3f} mPa.s")
    print(f"  heating             {d['thermal_budget']['total_rise_K']:8.4f} K "
          f"(water absorbs {d['thermal_budget']['bulk_absorption_K']:.5f} K)")
    print()
    print(f"  recovery (efficiency) {m['efficiency_percent']:6.1f} %")
    print(f"  purity                {m['purity_percent']:6.1f} %")
    print(f"  live purity           {m['live_purity_percent']:6.1f} %")
    print(f"  enrichment            {m['enrichment_fold']:6.2f} x")
    print(f"  viability in -> out   {m['viability_in_percent']:6.1f} -> "
          f"{m['viability_out_percent']:.1f} %")
    print()
    print("  cell safety margins (how far from each published damage threshold):")
    print(f"    thermal    {m['thermal_margin']:11.3g} x")
    print(f"    shear      {m['shear_margin']:11.0f} x")
    print(f"    cavitation {m['cavitation_margin']:11.1f} x   "
          f"(MI = {m['mechanical_index']:.3f}, limit {m['mechanical_index_limit']})")
    print()
    header = f"  {'population':10s} {'n':>5s} {'collected':>10s} {'|dx| (um)':>11s} " \
             f"{'to node (um)':>13s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, stats in m["per_population"].items():
        print(f"  {label:10s} {stats['n']:5d} "
              f"{stats['collected_fraction'] * 100:9.1f}% "
              f"{stats['mean_abs_displacement_um']:11.1f} "
              f"{stats['median_distance_to_node_um']:13.1f}")
    print()
    for label, phi in outcome.diagnostics["contrast_factors"].items():
        cell = get_cell(label)
        print(f"  {label:10s} Phi = {phi['phi_classical']:.4f} (classical), "
              f"{phi['phi_effective_ssaw']:.4f} (SSAW effective), r = {cell.r * 1e6:.2f} um")
    for w in caught:
        print(f"\n  [{w.category.__name__}] {w.message}")

    # ---- figures ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = node_positions(params.channel_width, sim.wavelength,
                           node_offset=sim.node_offset)

    x = np.linspace(0.0, params.channel_width, 400)
    profiles = {
        pop.resolved_label(): primary_radiation_force_1d(
            x, p0=params.p0,
            volume=4 / 3 * np.pi * get_cell(pop.cell_type).r**3,
            kappa_f=sim.fluid.kappa, wavelength=sim.wavelength,
            phi=sim.phi_for(get_cell(pop.cell_type)), node_offset=sim.node_offset,
        )
        for pop in params.populations
    }

    alive = outcome.cells["alive"].to_numpy()
    figures = {
        "02_live_view": live_view_figure(
            outcome.tracks.trajectories, channel_width=params.channel_width,
            channel_length=params.channel_length, node_positions=nodes,
            alive=alive,
            collection_bounds=sim.collection_bounds,
        ),
        "02_cross_section": cross_section_figure(
            outcome.tracks.trajectories, channel_width=params.channel_width,
            channel_height=params.channel_height,
            channel_length=params.channel_length, alive=alive,
            node_positions=nodes,
        ),
        "02_live_count": cumulative_count_figure(
            outcome.tracks.trajectories, outcome.cells,
            channel_length=params.channel_length,
            collection_bounds=sim.collection_bounds,
        ),
        "02_trajectories": trajectory_figure(
            outcome.tracks.trajectories, node_positions=nodes,
            channel_width=params.channel_width,
            channel_length=params.channel_length,
            title="Cell trajectories through the acoustic field",
        ),
        "02_outlet_histogram": outlet_histogram_figure(
            outcome.cells, channel_width=params.channel_width,
            collection_bounds=sim.collection_bounds,
        ),
        "02_force_profile": force_profile_figure(x, profiles, node_positions=nodes),
        "02_size_distribution": size_distribution_figure(outcome.cells),
    }
    png_ok = True
    for name, fig in figures.items():
        html = OUT_DIR / f"{name}.html"
        fig.write_html(html, include_plotlyjs="cdn")
        print(f"  wrote {html}")
        if not png_ok:
            continue
        try:
            fig.write_image(OUT_DIR / f"{name}.png", width=1100, height=520, scale=2)
            print(f"  wrote {OUT_DIR / f'{name}.png'}")
        except Exception as exc:  # noqa: BLE001 - kaleido is an optional extra
            print(f"  (PNG export unavailable: {type(exc).__name__} — pip install kaleido; "
                  "the interactive HTML is written regardless)")
            png_ok = False

    result = InstrumentResult(
        fields=outcome.tracks.trajectories, metrics=m, table=outcome.cells,
        meta={"instrument": "saw_sorter", "example": "02_ctc_vs_rbc"},
    )
    written = save_result(result, OUT_DIR, "02_ctc_vs_rbc")
    for kind, path in written.items():
        print(f"  wrote {kind}: {path}")


if __name__ == "__main__":
    main()
