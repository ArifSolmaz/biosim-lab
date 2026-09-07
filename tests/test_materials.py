"""The material library's provenance contract."""

import numpy as np
import pytest

from biosim_lab.core import materials as m


def test_every_value_has_a_doi_or_an_explicit_assumption():
    """No number in the library may be unsourced and unflagged."""
    for group, key, prop, value in m.all_values():
        prov = value.prov
        assert (prov.doi is not None) ^ (prov.assumption is not None), (
            f"{group}/{key}.{prop} has neither a DOI nor an ASSUMPTION"
        )
        if prov.doi is not None:
            assert prov.citation, f"{group}/{key}.{prop} cites a DOI with no reference text"
        else:
            assert len(prov.assumption) > 20, (
                f"{group}/{key}.{prop} is flagged ASSUMPTION but does not say why"
            )


def test_provenance_rejects_both_or_neither():
    with pytest.raises(ValueError):
        m.Provenance()
    with pytest.raises(ValueError):
        m.Provenance(doi="10.1/x", assumption="also this")


def test_audit_reports_only_assumptions_by_default():
    rows = m.audit()
    assert rows, "the library should have some honest assumptions"
    assert all(r["provenance"].startswith("ASSUMPTION") for r in rows)
    assert len(rows) < len(m.all_values())


def test_water_compressibility_matches_the_textbook_value():
    # kappa = 1/(rho c^2); water at 25 C is 4.48e-10 1/Pa.
    assert m.WATER.kappa == pytest.approx(4.48e-10, rel=0.01)


def test_cell_radii_are_lognormal_with_the_requested_mean_and_cv():
    rng = np.random.default_rng(0)
    r = m.MCF7.sample_radii(200_000, rng)
    assert r.mean() == pytest.approx(m.MCF7.r, rel=0.01)
    assert (r.std() / r.mean()) == pytest.approx(float(m.MCF7.radius_cv), rel=0.02)
    assert (r > 0).all()


def test_zero_cv_gives_a_monodisperse_population():
    cell = m.CellType(
        key="t", name="t",
        radius_mean=m.V(5e-6, "m", m.Provenance(assumption="test fixture, not a real cell")),
        radius_cv=m.V(0.0, "dimensionless", m.Provenance(assumption="test fixture, exact")),
        density=m.V(1050.0, "kg/m**3", m.Provenance(assumption="test fixture value")),
        compressibility=m.V(4e-10, "1/Pa", m.Provenance(assumption="test fixture value")),
    )
    assert np.allclose(cell.sample_radii(10), 5e-6)


def test_lookup_errors_name_the_alternatives():
    with pytest.raises(KeyError, match="mcf7"):
        m.get_cell("nope")
    with pytest.raises(KeyError, match="water"):
        m.get_fluid("nope")
