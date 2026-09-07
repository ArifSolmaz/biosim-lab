"""Example 4 — analytic vs FEM validation.

Solves the same operating point twice: once with the closed-form SSAW standing
wave, once with the Gmsh + scikit-fem Helmholtz solution post-processed through
the Gor'kov potential.  Reports the RMS difference in the force profile and in
the sorting metrics.

The two are *not* expected to agree perfectly, and the report quantifies why.
The dominant cause is that the closed-form model is **one-dimensional**: it
applies the substrate-plane pressure amplitude at every height, while the real
leaky-SAW field decays with distance from the transducer. Cells in the upper
half of a 50 um channel therefore feel substantially less force than the 1-D
model assumes, so the analytic mode is a *best-case* estimate of separation
performance. Two smaller effects also contribute:

* the analytic model assumes a perfect standing wave in the vertical direction
  (a rigid ceiling), while the FEM ceiling is a PDMS impedance boundary that
  lets part of the leaky wave out, adding a travelling component to <v^2>;
* the FEM field is renormalised to the requested ``p0``, since the absolute
  amplitude of a driven resonator depends on a quality factor that is a fitted
  device property rather than a first-principles output.

Run::

    python examples/04_validate_analytic_vs_fem.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from biosim_lab.core.materials import LINBO3_128YX, MCF7, RBC, WATER
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.core.viz.curves import field_heatmap_figure, force_profile_figure
from biosim_lab.instruments.saw_sorter.fem_model import SAWFieldModel
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    effective_contrast_factor,
    primary_radiation_force_1d,
)
from biosim_lab.instruments.saw_sorter.simulate import SAWSorterParams, SAWSorterSimulation

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"

FREQUENCY = 6.632e6
WIDTH, HEIGHT = 300e-6, 50e-6
P0 = 0.45e6
NODE = WIDTH / 2
TOLERANCE = 0.30  # RMS of the force profile, relative to its peak


def force_profiles(cell, xs: np.ndarray, field: SAWFieldModel) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(analytic, fem)`` force profiles [N] at the substrate plane."""
    k_x = 2 * np.pi / field.wavelength
    k_f = 2 * np.pi * FREQUENCY / WATER.c
    phi = effective_contrast_factor(
        cell.rho, WATER.rho, cell.kappa, WATER.kappa, k_transverse=k_x, k_fluid=k_f
    )
    volume = 4 / 3 * np.pi * cell.r**3
    analytic = primary_radiation_force_1d(
        xs, p0=P0, volume=volume, kappa_f=WATER.kappa, wavelength=field.wavelength,
        phi=phi, node_offset=NODE,
    )
    ff = field.force_field(radius=cell.r, density=cell.rho, compressibility=cell.kappa)
    fem = np.interp(xs, ff["x"].values, ff["F_x"].values[:, 0])
    return np.asarray(analytic), fem


def main() -> None:
    print("=" * 76)
    print("Analytic vs FEM validation — identical parameters, two routes")
    print("=" * 76)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RegimeWarning)
        field = SAWFieldModel(
            frequency=FREQUENCY, channel_width=WIDTH, channel_height=HEIGHT,
            fluid=WATER, substrate=LINBO3_128YX, pressure_amplitude=P0,
            resolution=64, element_order=2,
        )
        result = field.solve(nx=301, ny=41)

    d = result.diagnostics
    print(f"  wall model            {d['wall_model']}")
    print(f"  SAW wavelength        {d['saw_wavelength_m'] * 1e6:8.1f} um")
    print(f"  wavelength in water   {d['fluid_wavelength_m'] * 1e6:8.1f} um")
    print(f"  Rayleigh angle        {d['rayleigh_angle_deg']:8.1f} deg")
    print(f"  pressure nodes        {d['n_pressure_nodes']:8d}")
    print(f"  FEM->p0 renormalise   {d['normalisation_scale']:8.3g}")
    print()

    # ---- pressure profile ------------------------------------------------
    xs = result.grid["x"].values
    p_fem = np.abs(result.grid["p_real"].values + 1j * result.grid["p_imag"].values)[:, 0]
    p_analytic = P0 * np.abs(np.sin(2 * np.pi / field.wavelength * (xs - NODE)))
    rms_p = np.sqrt(np.mean((p_fem - p_analytic) ** 2)) / P0
    print(f"  pressure profile RMS difference   {rms_p * 100:6.2f} % of p0")

    # ---- force profiles --------------------------------------------------
    print()
    header = f"  {'cell':8s} {'|F| analytic':>14s} {'|F| FEM':>12s} {'ratio':>8s} {'RMS':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    profiles_analytic, profiles_fem = {}, {}
    worst_rms = 0.0
    for cell in (MCF7, RBC):
        analytic, fem = force_profiles(cell, xs, field)
        peak = np.abs(analytic).max()
        rms = np.sqrt(np.mean((fem - analytic) ** 2)) / peak
        worst_rms = max(worst_rms, rms)
        print(f"  {cell.key:8s} {peak * 1e12:12.2f} pN {np.abs(fem).max() * 1e12:10.2f} pN "
              f"{np.abs(fem).max() / peak:8.3f} {rms * 100:7.2f} %")
        profiles_analytic[f"{cell.key} (analytic)"] = analytic
        profiles_fem[f"{cell.key} (FEM)"] = fem

    # ---- how much force is lost with height ------------------------------
    ff = field.force_field(radius=MCF7.r, density=MCF7.rho, compressibility=MCF7.kappa)
    fx = ff["F_x"].values
    ys = ff["y"].values
    peak_by_height = np.abs(fx).max(axis=0)
    print()
    print("  vertical decay of the lateral force (the 1-D model ignores this):")
    for j in np.linspace(0, len(ys) - 1, 6).astype(int):
        bar = "#" * int(40 * peak_by_height[j] / peak_by_height[0])
        print(f"    y = {ys[j] * 1e6:5.1f} um   peak |F_x| = "
              f"{peak_by_height[j] * 1e12:7.2f} pN  |{bar}")
    height_avg = peak_by_height.mean() / peak_by_height[0]
    print(f"    height-averaged / substrate-plane force: {height_avg:.3f}")

    # ---- sorting metrics -------------------------------------------------
    print()
    base = dict(
        frequency=FREQUENCY, voltage_pp=None, pressure_amplitude=P0,
        channel_width=WIDTH, channel_height=HEIGHT, channel_length=2e-3,
        populations=[
            {"cell_type": "mcf7", "count": 200, "target": True},
            {"cell_type": "rbc", "count": 200, "target": False},
        ],
        seed=20260907, n_time_samples=61,
    )
    metrics = {}
    for mode in ("analytic", "fem"):
        params = SAWSorterParams.model_validate(
            {**base, "mode": mode, "fem_resolution": 48, "fem_grid": (241, 33)}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RegimeWarning)
            metrics[mode] = SAWSorterSimulation(params).run().metrics

    header = f"  {'metric':28s} {'analytic':>10s} {'FEM':>10s} {'delta':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ("efficiency_percent", "purity_percent", "enrichment_fold"):
        a, f = metrics["analytic"][key], metrics["fem"][key]
        print(f"  {key:28s} {a:10.2f} {f:10.2f} {f - a:+10.2f}")
    for label in ("mcf7", "rbc"):
        a = metrics["analytic"]["per_population"][label]["collected_fraction"] * 100
        f = metrics["fem"]["per_population"][label]["collected_fraction"] * 100
        print(f"  {'collected % ' + label:28s} {a:10.2f} {f:10.2f} {f - a:+10.2f}")

    print()
    verdict = "PASS" if worst_rms < TOLERANCE else "REVIEW"
    print(f"  worst force-profile RMS difference: {worst_rms * 100:.1f} % "
          f"(tolerance {TOLERANCE * 100:.0f} %) -> {verdict}")
    print("\n  Why the sorting metrics differ more than the force profile does:")
    print("   * the 1-D model applies the substrate-plane force at every height, but the")
    print("     real field decays with y — averaged over the channel the lateral force is")
    print(f"     only {height_avg * 100:.0f} % of the substrate-plane value. "
          "The analytic mode is a")
    print("     best-case estimate; the FEM mode is the realistic one.")
    print("   * the FEM ceiling is a PDMS impedance boundary rather than a rigid wall, so")
    print("     part of the leaky wave escapes and adds a travelling component to <v^2>;")
    print("   * the FEM amplitude is renormalised to p0, because the absolute response of")
    print("     a driven resonator depends on a fitted quality factor.")
    print("   * the vertical Gor'kov force is disabled by default (see")
    print("     SAWSorterSimulation._fem_arf): without a lift force to balance it, cells")
    print("     would pile up against a wall where the axial velocity is zero.")

    # ---- figures ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = force_profile_figure(
        xs, {**profiles_analytic, **profiles_fem},
        node_positions=[NODE],
        title="Acoustic radiation force: analytic vs FEM",
    )
    fig.write_html(OUT_DIR / "04_force_comparison.html", include_plotlyjs="cdn")
    print(f"\nwrote {OUT_DIR / '04_force_comparison.html'}")

    heat = field_heatmap_figure(
        result.grid, "p_abs", title="FEM pressure magnitude |p| (Pa)",
        colorbar_title="|p| (Pa)",
    )
    heat.write_html(OUT_DIR / "04_pressure_field.html", include_plotlyjs="cdn")
    print(f"wrote {OUT_DIR / '04_pressure_field.html'}")
    for name, f in (("04_force_comparison", fig), ("04_pressure_field", heat)):
        try:
            f.write_image(OUT_DIR / f"{name}.png", width=1100, height=520, scale=2)
            print(f"wrote {OUT_DIR / f'{name}.png'}")
        except Exception as exc:  # noqa: BLE001
            print(f"(PNG export unavailable: {type(exc).__name__})")
            break

    raise SystemExit(0 if worst_rms < TOLERANCE else 1)


if __name__ == "__main__":
    main()
