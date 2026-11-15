import math

import pytest

from autocollimator.units import AngleUnit, OutputUnits, wrap_symmetric


@pytest.mark.unit
@pytest.mark.parametrize("unit", list(AngleUnit))
def test_round_trip_every_unit(unit: AngleUnit) -> None:
    for rad in (0.0, 1.0, -2.5e-6, 0.123456789):
        assert unit.to_rad(unit.from_rad(rad)) == pytest.approx(rad, rel=1e-15, abs=1e-300)


@pytest.mark.unit
def test_hand_computed_factors() -> None:
    assert AngleUnit.ARCSEC.from_rad(1.0) == pytest.approx(206264.806, abs=1e-3)
    assert AngleUnit.ARCSEC.from_rad(AngleUnit.DEG.to_rad(1.0)) == pytest.approx(3600.0, rel=1e-14)
    assert AngleUnit.ARCMIN.from_rad(AngleUnit.DEG.to_rad(1.0)) == pytest.approx(60.0, rel=1e-14)
    assert AngleUnit.DEG.from_rad(math.pi) == pytest.approx(180.0, rel=1e-15)
    assert AngleUnit.MRAD.from_rad(1.0) == 1000.0
    assert AngleUnit.URAD.from_rad(1.0) == 1_000_000.0
    assert AngleUnit.RAD.from_rad(0.25) == 0.25


@pytest.mark.unit
def test_symbols_and_parse() -> None:
    assert [u.symbol for u in AngleUnit] == ["rad", "°", "mrad", "µrad", "′", "″"]
    assert AngleUnit.parse("arcsec") is AngleUnit.ARCSEC
    assert AngleUnit.parse(AngleUnit.MRAD) is AngleUnit.MRAD
    with pytest.raises(ValueError, match="arcsec"):
        AngleUnit.parse("degrees")


@pytest.mark.unit
def test_output_units_default_and_independent() -> None:
    assert OutputUnits() == OutputUnits(AngleUnit.DEG, AngleUnit.DEG)
    u = OutputUnits(tilt_unit=AngleUnit.MRAD, rotation_unit=AngleUnit.DEG)
    assert (u.tilt_unit, u.rotation_unit) == (AngleUnit.MRAD, AngleUnit.DEG)
    with pytest.raises(TypeError):
        OutputUnits(tilt_unit="deg")


@pytest.mark.unit
def test_wrap_symmetric() -> None:
    assert wrap_symmetric(45.0, 90.0) == 45.0
    assert wrap_symmetric(-45.0, 90.0) == 45.0
    assert wrap_symmetric(45.0001, 90.0) == pytest.approx(-44.9999, abs=1e-12)
    assert wrap_symmetric(176.2, 90.0) == pytest.approx(-3.8, abs=1e-12)
    assert wrap_symmetric(-176.2, 90.0) == pytest.approx(3.8, abs=1e-12)
    assert wrap_symmetric(2.99, 90.0) == 2.99
    assert wrap_symmetric(-0.5, 90.0) == -0.5
