"""Instrument measurement: image measurement + instrument knowledge -> angular values.

This layer never touches images. It takes the pixel-domain image measurement (MeasuredValues)
and what is known about the instrument (InstrumentModel) and reports tilt, rotation and
orthogonality in the requested angular units. All math is done in radians; units are applied
only when building the result.

Conventions
-----------
Pixel coordinates are measured from the image corner (image y points down). Instrument axes are
X right, Y up, so for a spot at image position (x, y) and zero point (x0, y0)::

    dx_mm =  (x - x0) * pitch_x
    dy_mm = -(y - y0) * pitch_y
    tilt_x = 1/2 * atan(dx_mm / collimator_efl_mm)   # reflection doubles the angle
    tilt_y = 1/2 * atan(dy_mm / collimator_efl_mm)

Positive tilt_x: spot moves right. Positive tilt_y: spot moves up.

Crosshair line angles (from find_cross_center): ``measured_angle_a0`` is the vertical line
(top/bottom ROIs) measured from vertical, ``measured_angle_a1`` is the horizontal line
(left/right ROIs) measured from horizontal. Both are counter-clockwise positive as seen in the
image with Y up. A crosshair is symmetric under 90 deg, so each line angle and the rotation are
wrapped to (-45 deg, 45 deg].

- rotation = mean of the two wrapped line angles; positive = counter-clockwise (Y up).
- orthogonality = a0 - a1 = (angle measured counter-clockwise from the horizontal line to the
  vertical line) - 90 deg. It is reported only; it is never zeroed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from autocollimator.units import AngleUnit, OutputUnits, wrap_symmetric

if TYPE_CHECKING:
    from autocollimator.config.models import MeasuredValues

LOGGER = logging.getLogger(__name__)

QUARTER_TURN_RAD = math.pi / 2.0
IMAGE_SIZE_MISMATCH_POLICIES = ("error", "warn")


class ImageSizeMismatchError(ValueError):
    """Image size does not match the capture mode's output size (policy "error")."""


class ReticleType(Enum):
    """Reticle imaged by the camera. Add new types here; to_angular dispatches on it."""

    CROSSHAIR = "crosshair"


class TiltZeroMode(Enum):
    IMAGE_CENTER = "image_center"
    CALIBRATED_POINT = "calibrated_point"
    MEASURED_ZERO = "measured_zero"


class RotationZeroMode(Enum):
    IMAGE_AXES = "image_axes"
    MEASURED_ZERO = "measured_zero"


def _parse_enum(enum_cls, value, name: str):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(m.value for m in enum_cls)
        raise ValueError(f"{name} must be one of ({valid}), got {value!r}") from None


def _opt_number(name: str, v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(v).__name__}")
    return float(v)


@dataclass(frozen=True)
class TiltZeroReference:
    """Tilt zero point in pixels. x_px/y_px are required unless mode is image_center."""

    mode: TiltZeroMode = TiltZeroMode.IMAGE_CENTER
    x_px: Optional[float] = None
    y_px: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _parse_enum(TiltZeroMode, self.mode, "tilt_zero_mode"))
        if self.mode is not TiltZeroMode.IMAGE_CENTER and (self.x_px is None or self.y_px is None):
            raise ValueError(f"tilt_zero_mode {self.mode.value!r} requires tilt_zero_x_px and tilt_zero_y_px")

    @staticmethod
    def from_dict(d: dict) -> "TiltZeroReference":
        return TiltZeroReference(
            mode=d.get("mode", TiltZeroMode.IMAGE_CENTER.value),
            x_px=_opt_number("tilt_zero.x_px", d.get("x_px")),
            y_px=_opt_number("tilt_zero.y_px", d.get("y_px")),
        )


@dataclass(frozen=True)
class RotationZeroReference:
    """Rotation zero in degrees, relative to the image axes. Required for measured_zero."""

    mode: RotationZeroMode = RotationZeroMode.IMAGE_AXES
    rotation_deg: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _parse_enum(RotationZeroMode, self.mode, "rotation_zero_mode"))
        if self.mode is RotationZeroMode.MEASURED_ZERO and self.rotation_deg is None:
            raise ValueError("rotation_zero_mode 'measured_zero' requires rotation_zero_deg")

    @staticmethod
    def from_dict(d: dict) -> "RotationZeroReference":
        return RotationZeroReference(
            mode=d.get("mode", RotationZeroMode.IMAGE_AXES.value),
            rotation_deg=_opt_number("rotation_zero.rotation_deg", d.get("rotation_deg")),
        )


@dataclass(frozen=True)
class InstrumentModel:
    """What is known about the instrument; everything to_angular needs besides the image measurement."""

    reticle_type: ReticleType
    collimator_efl_mm: float
    # Stored for the record; not used by the conversion yet.
    microscope_efl_mm: float
    capture_mode_id: str
    # Effective pixel pitch in the output image (sensor pitch scaled by the capture mode).
    pixel_pitch_x_um: float
    pixel_pitch_y_um: float
    expected_image_width: int
    expected_image_height: int
    image_size_mismatch: str = "error"
    tilt_zero: TiltZeroReference = TiltZeroReference()
    rotation_zero: RotationZeroReference = RotationZeroReference()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reticle_type", _parse_enum(ReticleType, self.reticle_type, "reticle_type"))
        for name in ("collimator_efl_mm", "microscope_efl_mm", "pixel_pitch_x_um", "pixel_pitch_y_um"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise ValueError(f"instrument_model.{name} must be a positive number, got {v!r}")
        for name in ("expected_image_width", "expected_image_height"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"instrument_model.{name} must be a positive int, got {v!r}")
        if self.image_size_mismatch not in IMAGE_SIZE_MISMATCH_POLICIES:
            raise ValueError(
                f"image_size_mismatch must be one of {IMAGE_SIZE_MISMATCH_POLICIES}, got {self.image_size_mismatch!r}"
            )

    @staticmethod
    def from_dict(d: dict) -> "InstrumentModel":
        return InstrumentModel(
            reticle_type=d["reticle_type"],
            collimator_efl_mm=d["collimator_efl_mm"],
            microscope_efl_mm=d["microscope_efl_mm"],
            capture_mode_id=d["capture_mode_id"],
            pixel_pitch_x_um=d["pixel_pitch_x_um"],
            pixel_pitch_y_um=d["pixel_pitch_y_um"],
            expected_image_width=d["expected_image_width"],
            expected_image_height=d["expected_image_height"],
            image_size_mismatch=d.get("image_size_mismatch", "error"),
            tilt_zero=TiltZeroReference.from_dict(d.get("tilt_zero") or {}),
            rotation_zero=RotationZeroReference.from_dict(d.get("rotation_zero") or {}),
        )


@dataclass(frozen=True)
class AngularMeasurement:
    """What the autocollimator reports. Angular values are None only when success is False."""

    success: bool
    tilt_x: Optional[float]
    tilt_y: Optional[float]
    rotation: Optional[float]
    orthogonality: Optional[float]
    tilt_unit: AngleUnit
    rotation_unit: AngleUnit
    # Zero references used: the resolved tilt zero point in pixels, rotation zero in rotation_unit.
    tilt_zero_mode: TiltZeroMode
    tilt_zero_x_px: float
    tilt_zero_y_px: float
    rotation_zero_mode: RotationZeroMode
    rotation_zero: float
    # Intermediate spot displacement from the tilt zero point, instrument axes (X right, Y up).
    spot_dx_mm: Optional[float]
    spot_dy_mm: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tilt_unit", AngleUnit.parse(self.tilt_unit, "tilt_unit"))
        object.__setattr__(self, "rotation_unit", AngleUnit.parse(self.rotation_unit, "rotation_unit"))
        object.__setattr__(self, "tilt_zero_mode", _parse_enum(TiltZeroMode, self.tilt_zero_mode, "tilt_zero_mode"))
        object.__setattr__(
            self, "rotation_zero_mode", _parse_enum(RotationZeroMode, self.rotation_zero_mode, "rotation_zero_mode")
        )
        if self.success:
            missing = [
                name
                for name in ("tilt_x", "tilt_y", "rotation", "orthogonality", "spot_dx_mm", "spot_dy_mm")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"angular_measurement: {', '.join(missing)} cannot be None when success is True")

    @staticmethod
    def from_dict(d: dict) -> "AngularMeasurement":
        return AngularMeasurement(
            success=d["success"],
            tilt_x=_opt_number("angular_measurement.tilt_x", d.get("tilt_x")),
            tilt_y=_opt_number("angular_measurement.tilt_y", d.get("tilt_y")),
            rotation=_opt_number("angular_measurement.rotation", d.get("rotation")),
            orthogonality=_opt_number("angular_measurement.orthogonality", d.get("orthogonality")),
            tilt_unit=d["tilt_unit"],
            rotation_unit=d["rotation_unit"],
            tilt_zero_mode=d["tilt_zero_mode"],
            tilt_zero_x_px=float(d["tilt_zero_x_px"]),
            tilt_zero_y_px=float(d["tilt_zero_y_px"]),
            rotation_zero_mode=d["rotation_zero_mode"],
            rotation_zero=float(d["rotation_zero"]),
            spot_dx_mm=_opt_number("angular_measurement.spot_dx_mm", d.get("spot_dx_mm")),
            spot_dy_mm=_opt_number("angular_measurement.spot_dy_mm", d.get("spot_dy_mm")),
        )


# ----------------------------
# Crosshair geometry (radians)
# ----------------------------
def crosshair_line_angles_rad(a0_deg: float, a1_deg: float) -> tuple[float, float]:
    """Wrap the two crosshair line angles (degrees, as measured) to (-pi/4, pi/4] radians."""
    return (
        wrap_symmetric(math.radians(a0_deg), QUARTER_TURN_RAD),
        wrap_symmetric(math.radians(a1_deg), QUARTER_TURN_RAD),
    )


def crosshair_orthogonality_rad(a0_deg: float, a1_deg: float) -> float:
    """a0 - a1, wrapped: deviation of the angle between the lines from 90 deg."""
    w0, w1 = crosshair_line_angles_rad(a0_deg, a1_deg)
    return wrap_symmetric(w0 - w1, QUARTER_TURN_RAD)


def crosshair_rotation_rad(a0_deg: float, a1_deg: float) -> float:
    """Mean of the wrapped line angles relative to the image axes, handling the +-45 deg wrap."""
    _w0, w1 = crosshair_line_angles_rad(a0_deg, a1_deg)
    return wrap_symmetric(w1 + crosshair_orthogonality_rad(a0_deg, a1_deg) / 2.0, QUARTER_TURN_RAD)


# ----------------------------
# Conversion
# ----------------------------
def check_image_size(model: InstrumentModel, width: int, height: int) -> None:
    """Apply the mismatch policy: raise for "error", log a warning for "warn"."""
    expected = (model.expected_image_width, model.expected_image_height)
    if (width, height) == expected:
        return
    msg = (
        f"Image size {width}x{height} does not match capture mode {model.capture_mode_id!r} "
        f"output size {expected[0]}x{expected[1]}"
    )
    if model.image_size_mismatch == "error":
        raise ImageSizeMismatchError(msg + " (image_size_mismatch = \"error\")")
    LOGGER.warning("%s; using the image's own size for the image center", msg)


def resolve_tilt_zero_px(model: InstrumentModel, width: int, height: int) -> tuple[float, float]:
    """Tilt zero point in pixels for an image of the given size."""
    tz = model.tilt_zero
    if tz.mode is TiltZeroMode.IMAGE_CENTER:
        return (width - 1) / 2.0, (height - 1) / 2.0
    assert tz.x_px is not None and tz.y_px is not None  # enforced by TiltZeroReference
    return tz.x_px, tz.y_px


def to_angular(
    image_measurement: "MeasuredValues",
    model: InstrumentModel,
    units: OutputUnits = OutputUnits(),
) -> AngularMeasurement:
    """Convert an image measurement to an angular measurement. Pure; no I/O."""
    if model.reticle_type is not ReticleType.CROSSHAIR:
        raise NotImplementedError(f"reticle_type {model.reticle_type.value!r} is not supported")

    width = image_measurement.image_width
    height = image_measurement.image_height
    if width is None or height is None:
        # Records written before the image size was stored.
        width, height = model.expected_image_width, model.expected_image_height
    x0, y0 = resolve_tilt_zero_px(model, width, height)

    rz = model.rotation_zero
    rotation_zero_rad = 0.0 if rz.mode is RotationZeroMode.IMAGE_AXES else math.radians(rz.rotation_deg)
    tu, ru = units.tilt_unit, units.rotation_unit
    common = dict(
        tilt_unit=tu,
        rotation_unit=ru,
        tilt_zero_mode=model.tilt_zero.mode,
        tilt_zero_x_px=x0,
        tilt_zero_y_px=y0,
        rotation_zero_mode=rz.mode,
        rotation_zero=ru.from_rad(rotation_zero_rad),
    )

    if not image_measurement.success:
        return AngularMeasurement(
            success=False, tilt_x=None, tilt_y=None, rotation=None, orthogonality=None,
            spot_dx_mm=None, spot_dy_mm=None, **common,
        )

    dx_mm = (image_measurement.center_position_x - x0) * model.pixel_pitch_x_um / 1000.0
    dy_mm = -(image_measurement.center_position_y - y0) * model.pixel_pitch_y_um / 1000.0
    tilt_x_rad = 0.5 * math.atan(dx_mm / model.collimator_efl_mm)
    tilt_y_rad = 0.5 * math.atan(dy_mm / model.collimator_efl_mm)

    a0, a1 = image_measurement.measured_angle_a0, image_measurement.measured_angle_a1
    rotation_rad = wrap_symmetric(crosshair_rotation_rad(a0, a1) - rotation_zero_rad, QUARTER_TURN_RAD)
    orthogonality_rad = crosshair_orthogonality_rad(a0, a1)

    return AngularMeasurement(
        success=True,
        tilt_x=tu.from_rad(tilt_x_rad),
        tilt_y=tu.from_rad(tilt_y_rad),
        rotation=ru.from_rad(rotation_rad),
        orthogonality=ru.from_rad(orthogonality_rad),
        spot_dx_mm=dx_mm,
        spot_dy_mm=dy_mm,
        **common,
    )
