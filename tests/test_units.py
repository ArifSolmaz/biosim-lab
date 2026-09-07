"""Unit handling at the API boundary."""

import pint
import pytest

from biosim_lab.core.units import Q_, to_si, ureg


def test_string_quantities_convert_to_si():
    assert to_si("20 MHz", "Hz") == pytest.approx(20e6)
    assert to_si("300 um", "m") == pytest.approx(300e-6)
    assert to_si("5 uL/min", "m**3/s") == pytest.approx(5e-9 / 60)


def test_bare_numbers_are_assumed_si():
    assert to_si(20e6, "Hz") == 20e6


def test_quantity_objects_convert():
    assert to_si(Q_(300, "um"), "m") == pytest.approx(300e-6)


def test_wrong_dimension_is_rejected():
    with pytest.raises(pint.DimensionalityError):
        to_si("20 um", "Hz")


def test_single_registry_instance():
    # Quantities from different registries cannot be combined; the platform
    # must therefore expose exactly one.
    assert Q_(1, "m").__class__ is ureg.Quantity
