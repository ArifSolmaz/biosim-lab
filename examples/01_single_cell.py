"""Example 1 — one cell, one force balance.

The smallest useful thing the platform does: put a single MCF-7 cell in a
standing SAW field and watch the acoustic radiation force race the viscous
drag.  Everything larger is this, repeated.

Run::

    python examples/01_single_cell.py
"""

from __future__ import annotations

import numpy as np

from biosim_lab.core.materials import LINBO3_128YX, MCF7, WATER
from biosim_lab.core.particles import ForceRegistry, LagrangianTracker, make_state
from biosim_lab.instruments.saw_sorter.physics.acoustics import (
    contrast_factor,
    effective_contrast_factor,
    primary_radiation_force_1d,
    rayleigh_angle,
    saw_wavelength,
)
from biosim_lab.instruments.saw_sorter.physics.drag import mobility

FREQUENCY = 6.632e6      # Hz — chosen so one half SAW wavelength spans the channel
P0 = 0.45e6              # Pa — 15 Vpp through the documented calibration
CHANNEL_WIDTH = 300e-6   # m
NODE = CHANNEL_WIDTH / 2


def main() -> None:
    cell, fluid, substrate = MCF7, WATER, LINBO3_128YX

    lam = saw_wavelength(FREQUENCY, float(substrate.saw_velocity))
    k_x = 2 * np.pi / lam
    k_f = 2 * np.pi * FREQUENCY / fluid.c

    phi_classical = float(contrast_factor(cell.rho, fluid.rho, cell.kappa, fluid.kappa))
    phi_eff = float(
        effective_contrast_factor(
            cell.rho, fluid.rho, cell.kappa, fluid.kappa,
            k_transverse=k_x, k_fluid=k_f,
        )
    )
    volume = 4 / 3 * np.pi * cell.r**3

    print("=" * 70)
    print(f"Single-cell force balance — {cell.name}")
    print("=" * 70)
    print(f"  radius                {cell.r * 1e6:8.2f} um")
    print(f"  density               {cell.rho:8.1f} kg/m^3   (fluid {fluid.rho:.1f})")
    print(f"  compressibility       {cell.kappa:8.3e} 1/Pa   (fluid {fluid.kappa:.3e})")
    print(f"  contrast factor Phi   {phi_classical:8.4f}      (classical, 1-D wave)")
    print(f"  effective Phi in SSAW {phi_eff:8.4f}      (dipole term x (kx/kf)^2)")
    print()
    print(f"  SAW wavelength        {lam * 1e6:8.1f} um")
    print(f"  node spacing          {lam / 2 * 1e6:8.1f} um")
    print(f"  Rayleigh angle        "
          f"{np.degrees(rayleigh_angle(fluid.c, float(substrate.saw_velocity))):8.1f} deg")
    print()

    # Force at the point of steepest gradient, one eighth of a wavelength off the node.
    x_probe = NODE + lam / 8
    force = float(
        primary_radiation_force_1d(
            np.array([x_probe]), p0=P0, volume=volume, kappa_f=fluid.kappa,
            wavelength=lam, phi=phi_eff, node_offset=NODE,
        )[0]
    )
    speed = abs(force) * float(mobility(np.array([cell.r]), fluid.mu)[0])
    print(f"  peak radiation force  {abs(force) * 1e12:8.2f} pN")
    print(f"  terminal speed        {speed * 1e6:8.2f} um/s")
    print(f"  time to cross {CHANNEL_WIDTH / 2 * 1e6:.0f} um   "
          f"{CHANNEL_WIDTH / 2 / speed:8.2f} s (at peak force)")
    print()

    # Integrate the real trajectory: the force falls to zero at the node, so the
    # approach is asymptotic and slower than the constant-force estimate.
    registry = ForceRegistry()

    def arf(state):
        out = np.zeros_like(state.x)
        out[:, 0] = primary_radiation_force_1d(
            state.x[:, 0], p0=P0, volume=state.volume, kappa_f=fluid.kappa,
            wavelength=lam, phi=phi_eff, node_offset=NODE,
        )
        return out

    registry.register("acoustic_radiation", arf, description="Gor'kov primary force")

    state = make_state(
        np.array([[20e-6, 25e-6]]), cell.r, cell.rho, cell.kappa, ["mcf7"]
    )
    tracker = LagrangianTracker(
        registry, lambda t, x: np.zeros_like(x), fluid.mu,
        bounds=((0.0, CHANNEL_WIDTH), (0.0, 50e-6)),
    )
    track = tracker.run(state, (0.0, 2.0), n_samples=9)
    positions = track.trajectories["position"].values[0, :, 0]
    times = track.trajectories["time"].values

    print("  trajectory from x = 20 um:")
    for t, x in zip(times, positions):
        bar = "#" * int(60 * x / CHANNEL_WIDTH)
        print(f"    t = {t:4.2f} s   x = {x * 1e6:6.1f} um  |{bar}")
    print(f"\n  distance to the node after 2 s: "
          f"{abs(positions[-1] - NODE) * 1e6:.1f} um")


if __name__ == "__main__":
    main()
