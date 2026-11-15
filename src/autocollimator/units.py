"""Angular output units and angle wrapping.

Pure helpers with no knowledge of images or of the instrument. All conversions go through
radians: ``unit.from_rad(x)`` converts a value in radians to ``unit``, ``unit.to_rad(x)`` back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AngleUnit(Enum):
    """Angular unit: value is the config string; ``per_rad`` is units per radian."""

    RAD = ("rad", 1.0, "rad")
    DEG = ("deg", 180.0 / math.pi, "°")
    MRAD = ("mrad", 1e3, "mrad")
    URAD = ("urad", 1e6, "µrad")
    ARCMIN = ("arcmin", 180.0 * 60.0 / math.pi, "′")
    ARCSEC = ("arcsec", 180.0 * 3600.0 / math.pi, "″")

    def __new__(cls, key: str, per_rad: float, symbol: str) -> "AngleUnit":
        obj = object.__new__(cls)
        obj._value_ = key
        obj.per_rad = per_rad
        obj.symbol = symbol
        return obj

    def from_rad(self, value: float) -> float:
        """Convert ``value`` in radians to this unit."""
        return value * self.per_rad

    def to_rad(self, value: float) -> float:
        """Convert ``value`` in this unit to radians."""
        return value / self.per_rad

    @classmethod
    def parse(cls, value: "str | AngleUnit", name: str = "angle unit") -> "AngleUnit":
        """Return the AngleUnit for a config string (or pass an AngleUnit through)."""
        if isinstance(value, AngleUnit):
            return value
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(u.value for u in cls)
            raise ValueError(f"{name} must be one of ({valid}), got {value!r}") from None


@dataclass(frozen=True)
class OutputUnits:
    """Units for reported values; tilt and rotation are chosen independently."""

    tilt_unit: AngleUnit = AngleUnit.DEG
    rotation_unit: AngleUnit = AngleUnit.DEG

    def __post_init__(self) -> None:
        if not isinstance(self.tilt_unit, AngleUnit):
            raise TypeError(f"tilt_unit must be AngleUnit, got {type(self.tilt_unit).__name__}")
        if not isinstance(self.rotation_unit, AngleUnit):
            raise TypeError(f"rotation_unit must be AngleUnit, got {type(self.rotation_unit).__name__}")


def wrap_symmetric(value: float, period: float) -> float:
    """Wrap ``value`` into (-period/2, period/2]. Values already in range are returned unchanged."""
    half = period / 2.0
    if -half < value <= half:
        return value
    return value - period * math.ceil((value - half) / period)
