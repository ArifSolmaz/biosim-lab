"""Acoustic radiation force: formulas, signs and cross-checks.

These are the tests that decide whether the physics is right, so they check
identities rather than remembered numbers wherever possible.
"""

import numpy as np
import pytest
import xarray as xr

from biosim_lab.core.materials import LIPID_DROPLET, MCF7, POLYSTYRENE_BEAD, RBC, WATER
from biosim_lab.core.plugin import RegimeWarning
from biosim_lab.instruments.saw_sorter.physics import acoustics as ac


def test_contrast_factor_is_positive_for_cells_and_negative_for_lipid():
    """Sign of Phi decides node vs antinode — the whole device depends on it."""
    for cell in (MCF7, RBC, POLYSTYRENE_BEAD):
        phi = float(ac.contrast_factor(cell.rho, WATER.rho, cell.kappa, WATER.kappa))
        assert phi > 0, f"{cell.key} should focus at the pressure node"
    phi_lipid = float(
        ac.contrast_factor(LIPID_DROPLET.rho, WATER.rho, LIPID_DROPLET.kappa, WATER.kappa)
    )
    assert phi_lipid < 0, "lipid droplets must move to the pressure antinode"


def test_classical_and_bruus_contrast_factors_differ_by_exactly_three():
    """The factor-3 trap: Phi_classical == 3 * Phi_Bruus, always."""
    for cell in (MCF7, RBC, LIPID_DROPLET):
        classical = ac.contrast_factor(cell.rho, WATER.rho, cell.kappa, WATER.kappa)
        bruus = ac.bruus_phi(cell.rho, WATER.rho, cell.kappa, WATER.kappa)
        assert float(classical) == pytest.approx(3.0 * float(bruus), rel=1e-12)


def test_contrast_factor_vanishes_for_a_neutral_particle():
    phi = ac.contrast_factor(WATER.rho, WATER.rho, WATER.kappa, WATER.kappa)
    assert float(phi) == pytest.approx(0.0, abs=1e-12)


def test_effective_contrast_reduces_to_classical_when_kx_equals_kf():
    k = 1234.5
    classical = ac.contrast_factor(MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa)
    effective = ac.effective_contrast_factor(
        MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa, k_transverse=k, k_fluid=k
    )
    assert float(effective) == pytest.approx(float(classical), rel=1e-12)


def test_effective_contrast_rejects_an_evanescent_pattern():
    with pytest.raises(ValueError, match="evanescent"):
        ac.effective_contrast_factor(
            MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa,
            k_transverse=2000.0, k_fluid=1000.0,
        )


def test_force_pushes_positive_contrast_particles_towards_the_node():
    lam = 600e-6
    node = 150e-6
    x = np.array([node - 30e-6, node + 30e-6])
    volume = 4 / 3 * np.pi * MCF7.r**3
    phi = ac.contrast_factor(MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa)
    f = ac.primary_radiation_force_1d(
        x, p0=4e5, volume=volume, kappa_f=WATER.kappa, wavelength=lam, phi=phi,
        node_offset=node,
    )
    assert f[0] > 0, "left of the node the force must point right"
    assert f[1] < 0, "right of the node the force must point left"


def test_force_vanishes_at_nodes_and_antinodes():
    lam = 600e-6
    volume = 4 / 3 * np.pi * 5e-6**3
    x = np.array([0.0, lam / 4, lam / 2])  # node, antinode, node
    f = ac.primary_radiation_force_1d(
        x, p0=4e5, volume=volume, kappa_f=WATER.kappa, wavelength=lam, phi=0.3
    )
    assert np.allclose(f, 0.0, atol=1e-20)


def test_force_scales_with_volume_and_inverse_wavelength():
    common = dict(p0=4e5, kappa_f=WATER.kappa, phi=0.3, node_offset=0.0)
    x = np.array([50e-6])
    base = ac.primary_radiation_force_1d(
        x, volume=1e-15, wavelength=600e-6, **common
    )
    doubled_volume = ac.primary_radiation_force_1d(
        x, volume=2e-15, wavelength=600e-6, **common
    )
    assert doubled_volume / base == pytest.approx(2.0)
    # Halving lambda doubles k, so the same *phase* must be compared.
    half_lambda = ac.primary_radiation_force_1d(
        np.array([25e-6]), volume=1e-15, wavelength=300e-6, **common
    )
    assert half_lambda / base == pytest.approx(2.0)


def test_force_scales_with_pressure_squared():
    common = dict(volume=1e-15, kappa_f=WATER.kappa, wavelength=600e-6, phi=0.3)
    x = np.array([50e-6])
    f1 = ac.primary_radiation_force_1d(x, p0=2e5, **common)
    f2 = ac.primary_radiation_force_1d(x, p0=4e5, **common)
    assert f2 / f1 == pytest.approx(4.0)


def test_gorkov_grid_force_reproduces_the_analytic_law_on_an_exact_field():
    """The general Gor'kov machinery must agree with the 1-D closed form.

    The synthetic field is an exact solution of the Helmholtz equation,
    ``p = p0 sin(kx x) cos(ky y)`` with ``kx^2 + ky^2 = (omega/c)^2``, so the
    comparison uses the SSAW-effective contrast factor.
    """
    p0, lam_saw = 4.5e5, 600e-6
    f = 3979.0 / lam_saw
    kx = 2 * np.pi / lam_saw
    kf = 2 * np.pi * f / WATER.c
    ky = np.sqrt(kf**2 - kx**2)

    xs = np.linspace(0.0, 300e-6, 601)
    ys = np.linspace(0.0, 50e-6, 121)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = p0 * np.sin(kx * (X - 150e-6)) * np.cos(ky * Y)
    grid = xr.Dataset(
        {"p_real": (("x", "y"), P), "p_imag": (("x", "y"), np.zeros_like(P))},
        coords={"x": xs, "y": ys},
    )

    volume = 4 / 3 * np.pi * MCF7.r**3
    numeric = ac.gorkov_force_on_grid(
        grid, volume=volume, rho_f=WATER.rho, c_f=WATER.c,
        rho_p=MCF7.rho, kappa_p=MCF7.kappa, frequency=f,
    )["F_x"].values[:, 0]

    phi_eff = ac.effective_contrast_factor(
        MCF7.rho, WATER.rho, MCF7.kappa, WATER.kappa, k_transverse=kx, k_fluid=kf
    )
    analytic = ac.primary_radiation_force_1d(
        xs, p0=p0, volume=volume, kappa_f=WATER.kappa, wavelength=lam_saw,
        phi=phi_eff, node_offset=150e-6,
    )
    rms = np.sqrt(np.mean((numeric - analytic) ** 2)) / np.abs(analytic).max()
    assert rms < 1e-3, f"Gor'kov grid force differs from the analytic law by {rms:.2%}"


def test_ssaw_node_spacing_is_half_the_saw_wavelength():
    lam = ac.saw_wavelength(20e6, 3979.0)
    assert lam == pytest.approx(3979.0 / 20e6)
    nodes = ac.node_positions(300e-6, lam, node_offset=150e-6)
    assert np.allclose(np.diff(nodes), lam / 2)
    assert np.isclose(nodes, 150e-6).any()


def test_rayleigh_angle_matches_snell():
    theta = ac.rayleigh_angle(WATER.c, 3979.0)
    assert np.degrees(theta) == pytest.approx(22.1, abs=0.2)
    with pytest.raises(ValueError, match="no leaky Rayleigh wave"):
        ac.rayleigh_angle(4000.0, 3979.0)


def test_large_particles_trigger_the_long_wavelength_warning():
    with pytest.warns(RegimeWarning, match=r"k\*a"):
        ac.check_gorkov_validity(20e-6, 75e-6)
    # A small particle in a long wave must not warn.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ac.check_gorkov_validity(1e-6, 600e-6)


def test_energy_density_definition():
    assert ac.acoustic_energy_density(4e5, WATER.kappa) == pytest.approx(
        0.25 * 4e5**2 * WATER.kappa
    )
