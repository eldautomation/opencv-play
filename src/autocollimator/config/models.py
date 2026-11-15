from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from autocollimator.instrument import (
    IMAGE_SIZE_MISMATCH_POLICIES,
    AngularMeasurement,
    InstrumentModel,
    ReticleType,
    RotationZeroMode,
    RotationZeroReference,
    TiltZeroMode,
    TiltZeroReference,
)
from autocollimator.units import AngleUnit

LOGGER = logging.getLogger(__name__)


def _require_int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v).__name__}")
    return v


def _opt_int(name: str, v) -> Optional[int]:
    if v is None:
        return None
    return _require_int(name, v)


def _require_float(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    return float(v)


def _require_type_str(name: str, v) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    return v


def _require_str(name: str, v) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    if not v.strip():
        raise ValueError(f"{name} cannot be empty")
    return v


def _opt_str(name: str, v) -> Optional[str]:
    if v is None:
        return None
    return _require_str(name, v)


def _require_reticle_type(name: str, v) -> str:
    valid = tuple(r.value for r in ReticleType)
    if v not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {v!r}")
    return v


def _require_mismatch_policy(name: str, v) -> str:
    if v not in IMAGE_SIZE_MISMATCH_POLICIES:
        raise ValueError(f"{name} must be one of {IMAGE_SIZE_MISMATCH_POLICIES}, got {v!r}")
    return v


def _opt_float(name: str, v) -> Optional[float]:
    if v is None:
        return None
    return _require_float(name, v)


def _opt_float_from_key(d: dict, key: str, name: str) -> Optional[float]:
    # TOML has no null; treat missing key as None.
    if key not in d:
        return None
    return _require_float(name, d[key])


def _require_bool(name: str, v) -> bool:
    if not isinstance(v, bool):
        raise TypeError(f"{name} must be bool, got {type(v).__name__}")
    return v


def _require_subpixel_neighbors(name: str, v) -> int:
    # Same rule as target_utils.sdrm_2: an int in 1..3, ValueError otherwise.
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 3:
        raise ValueError(f"{name} must be an int in 1..3, got {v!r}")
    return v


# Profile modes for find_cross_center (also used by target_utils).
PROFILE_MODES = ("average", "deskew")
# How a BGR image is reduced to one channel (also used by target_utils). "luma" is cv2 BGR2GRAY.
GRAY_MODES = ("luma", "green", "max")
FINE_WINDOW_MODES = ("fixed", "adaptive")
# roi_size_x value meaning "ROI length = min(crop width, crop height)" (also used by target_utils).
ROI_LENGTH_AUTO = "auto"
# Default nominal (designed) reticle rotation relative to the sensor, in degrees (R24).
DEFAULT_NOMINAL_SLANT_DEG = 6.0
OVERSAMPLE_STEP_RANGE = (0.02, 0.5)
REFINE_SEARCH_RANGE = (0.5, 3.0)
MAX_REFINE_SHIFT_RANGE = (0.1, 3.0)
# Per-ROI profile method recorded in MeasuredValues. "not_run": pass 1 failed, so no profile
# method applies. Plain "fallback" was written before the fallback reasons were recorded, and
# failed measurements were recorded as "average" before "not_run"; both still load.
ROI_METHODS = (
    "average",
    "not_run",
    "deskew",
    "fallback:tilt",
    "fallback:bins",
    "fallback:edge",
    "fallback:shift",
    "fallback",
)


def _require_roi_length(name: str, v) -> int | str:
    # An int > 0 or "auto"; the upper limit (the crop) is checked by MeasurementParameters.
    if v == ROI_LENGTH_AUTO:
        return v
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"{name} must be an int > 0 or {ROI_LENGTH_AUTO!r}, got {v!r}")
    return v


def _require_finite_float(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def _require_profile_mode(name: str, v) -> str:
    if v not in PROFILE_MODES:
        raise ValueError(f"{name} must be one of {PROFILE_MODES}, got {v!r}")
    return v


def _require_gray_mode(name: str, v) -> str:
    if v not in GRAY_MODES:
        raise ValueError(f"{name} must be one of {GRAY_MODES}, got {v!r}")
    return v


def _require_fine_window_mode(name: str, v) -> str:
    if v not in FINE_WINDOW_MODES:
        raise ValueError(f"{name} must be one of {FINE_WINDOW_MODES}, got {v!r}")
    return v


def _require_oversample_step(name: str, v) -> float:
    lo, hi = OVERSAMPLE_STEP_RANGE
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a number in {lo}..{hi}, got {v!r}")
    return float(v)


def _require_fine_window(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 2:
        raise ValueError(f"{name} must be an int >= 2, got {v!r}")
    return v


def _require_refine_search(name: str, v) -> float:
    lo, hi = REFINE_SEARCH_RANGE
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a number in {lo}..{hi}, got {v!r}")
    return float(v)


def _require_max_refine_shift(name: str, v) -> float:
    lo, hi = MAX_REFINE_SHIFT_RANGE
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a number in {lo}..{hi}, got {v!r}")
    return float(v)


def _opt_fraction(name: str, v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be None or a number in 0..1, got {v!r}")
    return float(v)


def _opt_window(name: str, v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 2:
        raise ValueError(f"{name} must be None or a number >= 2, got {v!r}")
    return float(v)


def _require_roi_method(name: str, v) -> str:
    if v not in ROI_METHODS:
        raise ValueError(f"{name} must be one of {ROI_METHODS}, got {v!r}")
    return v


@dataclass(frozen=True)
class ImageSensor:
    id: str
    name: str
    manufacturer: str
    part_number: str
    pixel_size_x: float
    pixel_size_y: float
    pixels_x: int
    pixels_y: int

    @staticmethod
    def from_dict(d: dict) -> "ImageSensor":
        return ImageSensor(
            id=_require_str("image_sensor.id", d["id"]),
            name=_require_str("image_sensor.name", d["name"]),
            manufacturer=_require_str("image_sensor.manufacturer", d["manufacturer"]),
            part_number=_require_str("image_sensor.part_number", d["part_number"]),
            pixel_size_x=_require_float("image_sensor.pixel_size_x", d["pixel_size_x"]),
            pixel_size_y=_require_float("image_sensor.pixel_size_y", d["pixel_size_y"]),
            pixels_x=_require_int("image_sensor.pixels_x", d["pixels_x"]),
            pixels_y=_require_int("image_sensor.pixels_y", d["pixels_y"]),
        )


@dataclass(frozen=True)
class IOHardware:
    id: str
    name: str
    description: str

    @staticmethod
    def from_dict(d: dict) -> "IOHardware":
        return IOHardware(
            id=_require_str("io_hardware.id", d["id"]),
            name=_require_str("io_hardware.name", d["name"]),
            description=_require_str("io_hardware.description", d["description"]),
        )


@dataclass(frozen=True)
class SourceOptics:
    id: str
    name: str
    line_thickness: float
    feature_a_x: Optional[float]
    feature_a_y: Optional[float]
    efl: Optional[float]

    @staticmethod
    def from_dict(d: dict) -> "SourceOptics":
        return SourceOptics(
            id=_require_str("source_optics.id", d["id"]),
            name=_require_str("source_optics.name", d["name"]),
            line_thickness=_require_float("source_optics.line_thickness", d["line_thickness"]),
            feature_a_x=_opt_float_from_key(d, "feature_a_x", "source_optics.feature_a_x"),
            feature_a_y=_opt_float_from_key(d, "feature_a_y", "source_optics.feature_a_y"),
            efl=_opt_float_from_key(d, "efl", "source_optics.efl"),
        )


# Deprecated detect_optics keys and their replacements.
_DETECT_OPTICS_RENAMED = {"efl1": "collimator_efl_mm", "efl2": "microscope_efl_mm"}


@dataclass(frozen=True)
class DetectOptics:
    id: str
    name: str
    # Collimator objective focal length; sets the tilt scale (tilt = 1/2 * atan(d / f)).
    collimator_efl_mm: float
    # Microscope objective focal length; stored, not used by the conversion yet.
    microscope_efl_mm: float

    def __post_init__(self) -> None:
        for name in ("collimator_efl_mm", "microscope_efl_mm"):
            if not getattr(self, name) > 0:
                raise ValueError(f"detect_optics.{name} must be > 0, got {getattr(self, name)!r}")

    @staticmethod
    def from_dict(d: dict) -> "DetectOptics":
        d = dict(d)
        for old, new in _DETECT_OPTICS_RENAMED.items():
            if old in d:
                if new in d:
                    raise ValueError(f"detect_optics {d.get('id')!r}: both {old!r} and {new!r} are set; remove {old!r}")
                msg = f"detect_optics {d.get('id')!r}: key {old!r} is deprecated, use {new!r}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                LOGGER.warning(msg)
                d[new] = d.pop(old)
        return DetectOptics(
            id=_require_str("detect_optics.id", d["id"]),
            name=_require_str("detect_optics.name", d["name"]),
            collimator_efl_mm=_require_float("detect_optics.collimator_efl_mm", d["collimator_efl_mm"]),
            microscope_efl_mm=_require_float("detect_optics.microscope_efl_mm", d["microscope_efl_mm"]),
        )


@dataclass(frozen=True)
class CaptureMode:
    """
    Camera output mode: output image size and the scaler crop on the sensor (sensor pixels).

    The effective pixel pitch of the output image is the sensor pitch scaled by
    crop size / output size. The "hd_1080" entry's crop (0, 134, 1456, 819) is an ASSUMPTION
    (centered 16:9 crop of the IMX296), not yet verified with libcamera ScalerCrop metadata.
    """

    id: str
    name: str
    output_width: int
    output_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int

    def __post_init__(self) -> None:
        for name in ("output_width", "output_height", "crop_width", "crop_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"capture_mode.{name} must be > 0, got {getattr(self, name)!r}")
        for name in ("crop_x", "crop_y"):
            if getattr(self, name) < 0:
                raise ValueError(f"capture_mode.{name} must be >= 0, got {getattr(self, name)!r}")

    def fits_sensor(self, sensor: "ImageSensor") -> bool:
        return (
            self.crop_x + self.crop_width <= sensor.pixels_x
            and self.crop_y + self.crop_height <= sensor.pixels_y
        )

    def effective_pixel_pitch_um(self, sensor: "ImageSensor") -> tuple[float, float]:
        """(x, y) pitch of one output-image pixel in um, from the sensor pitch and the crop."""
        return (
            sensor.pixel_size_x * self.crop_width / self.output_width,
            sensor.pixel_size_y * self.crop_height / self.output_height,
        )

    @staticmethod
    def from_dict(d: dict) -> "CaptureMode":
        return CaptureMode(
            id=_require_str("capture_mode.id", d["id"]),
            name=_require_str("capture_mode.name", d["name"]),
            output_width=_require_int("capture_mode.output_width", d["output_width"]),
            output_height=_require_int("capture_mode.output_height", d["output_height"]),
            crop_x=_require_int("capture_mode.crop_x", d["crop_x"]),
            crop_y=_require_int("capture_mode.crop_y", d["crop_y"]),
            crop_width=_require_int("capture_mode.crop_width", d["crop_width"]),
            crop_height=_require_int("capture_mode.crop_height", d["crop_height"]),
        )


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    serial_number: str
    # Secret: kept out of repr, logs and saved measurement files (R6).
    encryption_key: str = field(repr=False)
    image_sensor_id: str
    source_optics_id: str
    io_hardware_id: str
    detect_optics_id: str
    measurement_parameters_id: str
    quality_limits_id:str
    # Optional here so measurement records written before capture modes existed still load;
    # load_main requires it for the configured device.
    capture_mode_id: Optional[str] = None
    reticle_type: str = ReticleType.CROSSHAIR.value
    # What to do when an image's size differs from the capture mode's output size.
    image_size_mismatch: str = "error"
    # Default output units (AngleUnit values); overridable per call.
    tilt_unit: str = AngleUnit.DEG.value
    rotation_unit: str = AngleUnit.DEG.value
    # Tilt zero: "image_center" | "calibrated_point" | "measured_zero"; x/y in pixels, required
    # unless image_center.
    tilt_zero_mode: str = TiltZeroMode.IMAGE_CENTER.value
    tilt_zero_x_px: Optional[float] = None
    tilt_zero_y_px: Optional[float] = None
    # Rotation zero: "image_axes" | "measured_zero"; value in degrees relative to the image axes.
    rotation_zero_mode: str = RotationZeroMode.IMAGE_AXES.value
    rotation_zero_deg: Optional[float] = None

    def __post_init__(self) -> None:
        _require_reticle_type("device.reticle_type", self.reticle_type)
        _require_mismatch_policy("device.image_size_mismatch", self.image_size_mismatch)
        AngleUnit.parse(self.tilt_unit, "device.tilt_unit")
        AngleUnit.parse(self.rotation_unit, "device.rotation_unit")
        # Validates the modes and that each mode has the values it needs.
        self.tilt_zero
        self.rotation_zero

    @property
    def tilt_zero(self) -> TiltZeroReference:
        return TiltZeroReference(self.tilt_zero_mode, self.tilt_zero_x_px, self.tilt_zero_y_px)

    @property
    def rotation_zero(self) -> RotationZeroReference:
        return RotationZeroReference(self.rotation_zero_mode, self.rotation_zero_deg)

    @staticmethod
    def from_dict(d: dict) -> "Device":
        return Device(
            id=_require_str("device.id", d["id"]),
            name=_require_str("device.name", d["name"]),
            serial_number=_require_str("device.serial_number", d["serial_number"]),
            # Measurement records are saved without the key, so it may be absent here;
            # load_main enforces that a configured device has a real key.
            encryption_key=_require_type_str("device.encryption_key", d.get("encryption_key", "")),
            image_sensor_id=_require_str("device.image_sensor_id", d["image_sensor_id"]),
            source_optics_id=_require_str("device.source_optics_id", d["source_optics_id"]),
            io_hardware_id=_require_str("device.io_hardware_id", d["io_hardware_id"]),
            detect_optics_id=_require_str("device.detect_optics_id", d["detect_optics_id"]),
            measurement_parameters_id=_require_str("device.measurement_parameters_id", d["measurement_parameters_id"]),
            quality_limits_id=_require_str("device.quality_limits_id",d["quality_limits_id"]),
            capture_mode_id=_opt_str("device.capture_mode_id", d.get("capture_mode_id")),
            reticle_type=_require_str("device.reticle_type", d.get("reticle_type", ReticleType.CROSSHAIR.value)),
            image_size_mismatch=_require_str("device.image_size_mismatch", d.get("image_size_mismatch", "error")),
            tilt_unit=_require_str("device.tilt_unit", d.get("tilt_unit", AngleUnit.DEG.value)),
            rotation_unit=_require_str("device.rotation_unit", d.get("rotation_unit", AngleUnit.DEG.value)),
            tilt_zero_mode=_require_str(
                "device.tilt_zero_mode", d.get("tilt_zero_mode", TiltZeroMode.IMAGE_CENTER.value)
            ),
            tilt_zero_x_px=_opt_float("device.tilt_zero_x_px", d.get("tilt_zero_x_px")),
            tilt_zero_y_px=_opt_float("device.tilt_zero_y_px", d.get("tilt_zero_y_px")),
            rotation_zero_mode=_require_str(
                "device.rotation_zero_mode", d.get("rotation_zero_mode", RotationZeroMode.IMAGE_AXES.value)
            ),
            rotation_zero_deg=_opt_float("device.rotation_zero_deg", d.get("rotation_zero_deg")),
        )



# Pre-search (R4): when the measurement runs it, what it does when it fails, how it reports.
PRESEARCH_MODES = ("off", "every_image", "track")
PRESEARCH_FAILURE_ACTIONS = ("use_configured_crop", "fail")
PRESEARCH_METHODS = ("sdrm_full_frame", "projection")
# A candidate's line: "vertical" (position is x) or "horizontal" (position is y).
PRESEARCH_LINES = ("vertical", "horizontal")
# Where the crop used for a measurement came from.
CROP_SOURCES = ("configured", "presearch", "tracked")
PRESEARCH_CROP_FRACTION_RANGE = (0.1, 1.0)


def _require_choice(name: str, v, choices: tuple[str, ...]) -> str:
    if v not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {v!r}")
    return v


def _require_table(name: str, v) -> dict:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be a table, got {type(v).__name__}")
    return v


def _require_positive_int(name: str, v, minimum: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {v!r}")
    return v


def _require_number_in(name: str, v, lo: float, hi: float, *, lo_open: bool = False) -> float:
    ok = not isinstance(v, bool) and isinstance(v, (int, float)) and (lo < v if lo_open else lo <= v) and v <= hi
    if not ok:
        bracket = "(" if lo_open else "["
        raise ValueError(f"{name} must be a number in {bracket}{lo}, {hi}], got {v!r}")
    return float(v)


@dataclass(frozen=True)
class PresearchParameters:
    """
    Pre-search settings (R4), optional in TOML as [measurement_parameters.presearch].

    ``presearch_mode``: "off" (default; the configured crop is used and the pre-search is not
    run), "every_image" (pre-search each image, measure with the recommended crop) or "track"
    (measure around the previous successful center; pre-search when there is none or that
    measurement fails). ``presearch_failure_action``: "use_configured_crop" (default) or
    "fail" when the pre-search fails inside a measurement.
    """

    presearch_mode: str = "off"
    presearch_failure_action: str = "use_configured_crop"
    # The image is downsampled by this factor before searching.
    presearch_downsample: int = 2
    # Recommended region size = fraction x min(image width, image height) ...
    presearch_crop_fraction: float = 0.5
    # ... but never smaller than this (and never smaller than 4 x the line FWHM).
    min_crop_size_px: int = 240
    # RSS-ratio limit for the pre-search; separate from quality_limits.rss_ratio.
    presearch_q_limit: float = 0.7
    # The validation pass must agree with the pre-search center within this distance (px).
    presearch_agreement_px: float = 5.0
    # A second candidate line at least this fraction of the strongest's strength is warned about.
    presearch_ambiguity_ratio: float = 0.8

    def __post_init__(self) -> None:
        p = "measurement_parameters.presearch"
        _require_choice(f"{p}.presearch_mode", self.presearch_mode, PRESEARCH_MODES)
        _require_choice(f"{p}.presearch_failure_action", self.presearch_failure_action, PRESEARCH_FAILURE_ACTIONS)
        _require_positive_int(f"{p}.presearch_downsample", self.presearch_downsample)
        _require_number_in(f"{p}.presearch_crop_fraction", self.presearch_crop_fraction, *PRESEARCH_CROP_FRACTION_RANGE)
        _require_positive_int(f"{p}.min_crop_size_px", self.min_crop_size_px)
        _require_number_in(f"{p}.presearch_q_limit", self.presearch_q_limit, 0.0, 1.0, lo_open=True)
        if not self.presearch_q_limit < 1.0:
            raise ValueError(f"{p}.presearch_q_limit must be < 1, got {self.presearch_q_limit!r}")
        _require_number_in(f"{p}.presearch_agreement_px", self.presearch_agreement_px, 0.0, float("inf"), lo_open=True)
        _require_number_in(f"{p}.presearch_ambiguity_ratio", self.presearch_ambiguity_ratio, 0.0, 1.0, lo_open=True)

    @staticmethod
    def from_dict(d: dict) -> "PresearchParameters":
        unknown = sorted(set(d) - set(PresearchParameters.__dataclass_fields__))
        if unknown:
            raise ValueError(f"measurement_parameters.presearch: unknown key(s) {unknown}")
        return PresearchParameters(**d)


@dataclass(frozen=True)
class MeasurementParameters:
    id:str
    name:str
    crop_center_x: int
    crop_center_y: int
    crop_size_x: int
    crop_size_y: int
    # ROI length along the line's normal (px): an int, or "auto" = min(crop width, crop height).
    # An int must not exceed the crop (and, for the device's active parameters with the
    # pre-search on, min_crop_size_px: checked by store.validate_device_references).
    roi_size_x: int | str
    roi_size_y: int
    # Optional in TOML; defaults keep the original integer SDRM centering.
    subpixel: bool = False
    subpixel_neighbors: int = 2
    # Optional in TOML; "average" keeps the original single-pass profile.
    profile_mode: str = "average"
    oversample_step: float = 0.1
    fine_window: int = 40
    # Optional in TOML; "fixed" uses fine_window as is, "adaptive" grows it to 1.5 x line FWHM.
    fine_window_mode: str = "fixed"
    refine_search: float = 1.5
    max_refine_shift: float = 0.75
    # Optional in TOML; "luma" keeps the original cv2 grayscale conversion.
    gray_mode: str = "luma"
    # Optional in TOML ([measurement_parameters.presearch]); the defaults keep the pre-search off.
    presearch: PresearchParameters = field(default_factory=PresearchParameters)
    # Optional in TOML. Nominal (designed) reticle rotation relative to the sensor, degrees.
    # Not used by the measurement yet; to be compared with the measured rotation (R24).
    nominal_slant: float = DEFAULT_NOMINAL_SLANT_DEG

    def __post_init__(self) -> None:
        if not isinstance(self.presearch, PresearchParameters):
            raise TypeError(
                f"measurement_parameters.presearch must be PresearchParameters, got {type(self.presearch).__name__}"
            )
        _require_roi_length("measurement_parameters.roi_size_x", self.roi_size_x)
        if self.roi_size_x != ROI_LENGTH_AUTO:
            crop_min = min(self.crop_size_x, self.crop_size_y)
            if self.roi_size_x > crop_min:
                raise ValueError(
                    f"measurement_parameters.roi_size_x ({self.roi_size_x}) must not be larger than the crop "
                    f"(min(crop_size_x, crop_size_y) = {crop_min})"
                )
        _require_finite_float("measurement_parameters.nominal_slant", self.nominal_slant)
        _require_bool("measurement_parameters.subpixel", self.subpixel)
        _require_subpixel_neighbors("measurement_parameters.subpixel_neighbors", self.subpixel_neighbors)
        _require_profile_mode("measurement_parameters.profile_mode", self.profile_mode)
        _require_oversample_step("measurement_parameters.oversample_step", self.oversample_step)
        _require_fine_window("measurement_parameters.fine_window", self.fine_window)
        _require_fine_window_mode("measurement_parameters.fine_window_mode", self.fine_window_mode)
        _require_refine_search("measurement_parameters.refine_search", self.refine_search)
        _require_max_refine_shift("measurement_parameters.max_refine_shift", self.max_refine_shift)
        _require_gray_mode("measurement_parameters.gray_mode", self.gray_mode)
        if not self.refine_search < self.fine_window:
            raise ValueError(
                f"measurement_parameters.refine_search ({self.refine_search}) must be smaller than "
                f"measurement_parameters.fine_window ({self.fine_window})"
            )

    @staticmethod
    def from_dict(d: dict) -> "MeasurementParameters":
        return MeasurementParameters(
            id=_require_str("measurement_parameters.id",d["id"]),
            name=_require_str("measurement_parameters.name",d["name"]),
            crop_center_x=_require_int("measurement_parameters.crop_center_x", d["crop_center_x"]),
            crop_center_y=_require_int("measurement_parameters.crop_center_y", d["crop_center_y"]),
            crop_size_x=_require_int("measurement_parameters.crop_size_x", d["crop_size_x"]),
            crop_size_y=_require_int("measurement_parameters.crop_size_y", d["crop_size_y"]),
            roi_size_x=_require_roi_length("measurement_parameters.roi_size_x", d.get("roi_size_x", ROI_LENGTH_AUTO)),
            roi_size_y=_require_int("measurement_parameters.roi_size_y", d["roi_size_y"]),
            subpixel=_require_bool("measurement_parameters.subpixel", d.get("subpixel", False)),
            subpixel_neighbors=_require_subpixel_neighbors(
                "measurement_parameters.subpixel_neighbors", d.get("subpixel_neighbors", 2)
            ),
            profile_mode=_require_profile_mode("measurement_parameters.profile_mode", d.get("profile_mode", "average")),
            oversample_step=_require_oversample_step(
                "measurement_parameters.oversample_step", d.get("oversample_step", 0.1)
            ),
            fine_window=_require_fine_window("measurement_parameters.fine_window", d.get("fine_window", 40)),
            fine_window_mode=_require_fine_window_mode(
                "measurement_parameters.fine_window_mode", d.get("fine_window_mode", "fixed")
            ),
            refine_search=_require_refine_search(
                "measurement_parameters.refine_search", d.get("refine_search", 1.5)
            ),
            max_refine_shift=_require_max_refine_shift(
                "measurement_parameters.max_refine_shift", d.get("max_refine_shift", 0.75)
            ),
            gray_mode=_require_gray_mode("measurement_parameters.gray_mode", d.get("gray_mode", "luma")),
            presearch=PresearchParameters.from_dict(
                _require_table("measurement_parameters.presearch", d.get("presearch", {}))
            ),
            nominal_slant=_require_finite_float(
                "measurement_parameters.nominal_slant", d.get("nominal_slant", DEFAULT_NOMINAL_SLANT_DEG)
            ),
        )


@dataclass(frozen=True)
class QualityLimits:
    id:str
    name:str
    rss_ratio: float
    brightness_ratio: float
    max_brightness: float
    min_brightness: float
    linewidth: float
    metric_tolerance: float

    @staticmethod
    def from_dict(d: dict) -> "QualityLimits":
        return QualityLimits(
            id=_require_str("quality_limits.id",d["id"]),
            name=_require_str("quality_limits.name",d["name"]),
            rss_ratio=_require_float("quality_limits.rss_ratio", d["rss_ratio"]),
            brightness_ratio=_require_float("quality_limits.brightness_ratio", d["brightness_ratio"]),
            max_brightness=_require_float("quality_limits.max_brightness", d["max_brightness"]),
            min_brightness=_require_float("quality_limits.min_brightness", d["min_brightness"]),
            linewidth=_require_float("quality_limits.linewidth", d["linewidth"]),
            metric_tolerance=_require_float("quality_limits.metric_tolerance", d["metric_tolerance"]),
        )


@dataclass(frozen=True)
class PresearchCandidate:
    """A line found by the pre-search; ``position`` is x for "vertical", y for "horizontal" (px)."""

    line: str
    position: float
    # Peak height of the line's projection profile above its background (gray levels).
    strength: float

    def __post_init__(self) -> None:
        _require_choice("presearch_candidate.line", self.line, PRESEARCH_LINES)

    @staticmethod
    def from_dict(d: dict) -> "PresearchCandidate":
        return PresearchCandidate(
            line=_require_str("presearch_candidate.line", d["line"]),
            position=_require_float("presearch_candidate.position", d["position"]),
            strength=_require_float("presearch_candidate.strength", d["strength"]),
        )


@dataclass(frozen=True)
class PresearchResult:
    """
    Result of the pre-search (``autocollimator.presearch.presearch``). All positions and sizes
    are in full-resolution image pixels.

    On success the recommended square region (center, size) is set and ``failure_reason`` is
    None. On failure ``failure_reason`` says why and no region is recommended; ``position``
    may still hold the estimate that failed a check.
    """

    success: bool
    failure_reason: Optional[str]
    # "sdrm_full_frame" | "projection"; None when no method produced a position.
    method_used: Optional[str]
    # Estimated crosshair center.
    position_x: Optional[float]
    position_y: Optional[float]
    recommended_center_x: Optional[int]
    recommended_center_y: Optional[int]
    recommended_size: Optional[int]
    # sdrm_full_frame: one RSS ratio per strip (top, bottom, right, left);
    # projection: one per line (vertical, horizontal).
    q_values: tuple[float, ...] = ()
    line_fwhm_px: Optional[float] = None
    warnings: tuple[str, ...] = ()
    candidates: tuple[PresearchCandidate, ...] = ()
    region_shifted: bool = False
    region_shrunk: bool = False
    downsample: int = 1

    def __post_init__(self) -> None:
        _require_bool("presearch_result.success", self.success)
        if self.method_used is not None:
            _require_choice("presearch_result.method_used", self.method_used, PRESEARCH_METHODS)
        if self.success:
            missing = [
                name
                for name in ("method_used", "position_x", "position_y", "recommended_center_x",
                             "recommended_center_y", "recommended_size")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"presearch_result: {', '.join(missing)} cannot be None when success is True")
            if self.failure_reason is not None:
                raise ValueError("presearch_result: failure_reason must be None when success is True")
        elif not (isinstance(self.failure_reason, str) and self.failure_reason.strip()):
            raise ValueError("presearch_result: a failed result needs a failure_reason")

    @staticmethod
    def from_dict(d: dict) -> "PresearchResult":
        return PresearchResult(
            success=_require_bool("presearch_result.success", d["success"]),
            failure_reason=_opt_str("presearch_result.failure_reason", d.get("failure_reason")),
            method_used=_opt_str("presearch_result.method_used", d.get("method_used")),
            position_x=_opt_float("presearch_result.position_x", d.get("position_x")),
            position_y=_opt_float("presearch_result.position_y", d.get("position_y")),
            recommended_center_x=_opt_int("presearch_result.recommended_center_x", d.get("recommended_center_x")),
            recommended_center_y=_opt_int("presearch_result.recommended_center_y", d.get("recommended_center_y")),
            recommended_size=_opt_int("presearch_result.recommended_size", d.get("recommended_size")),
            q_values=tuple(_require_float("presearch_result.q_values", q) for q in d.get("q_values", ())),
            line_fwhm_px=_opt_float("presearch_result.line_fwhm_px", d.get("line_fwhm_px")),
            warnings=tuple(_require_str("presearch_result.warnings", w) for w in d.get("warnings", ())),
            candidates=tuple(PresearchCandidate.from_dict(c) for c in d.get("candidates", ())),
            region_shifted=_require_bool("presearch_result.region_shifted", d.get("region_shifted", False)),
            region_shrunk=_require_bool("presearch_result.region_shrunk", d.get("region_shrunk", False)),
            downsample=_require_int("presearch_result.downsample", d.get("downsample", 1)),
        )


@dataclass(frozen=True)
class PresearchRecord:
    """
    How the crop of one measurement was chosen: the pre-search mode, whether the pre-search
    ran and its full result, and the crop actually used (source "configured", "presearch"
    or "tracked"). The crop and source are None when nothing was measured (pre-search failed
    with presearch_failure_action "fail").
    """

    mode: str
    ran: bool
    result: Optional[PresearchResult]
    crop_center_x: Optional[int]
    crop_center_y: Optional[int]
    crop_size_x: Optional[int]
    crop_size_y: Optional[int]
    crop_source: Optional[str]

    def __post_init__(self) -> None:
        _require_choice("presearch.mode", self.mode, PRESEARCH_MODES)
        _require_bool("presearch.ran", self.ran)
        if self.ran != (self.result is not None):
            raise ValueError("presearch: result must be set exactly when the pre-search ran")
        if self.crop_source is not None:
            _require_choice("presearch.crop_source", self.crop_source, CROP_SOURCES)

    @staticmethod
    def from_dict(d: dict) -> "PresearchRecord":
        return PresearchRecord(
            mode=_require_str("presearch.mode", d["mode"]),
            ran=_require_bool("presearch.ran", d["ran"]),
            result=PresearchResult.from_dict(d["result"]) if d.get("result") else None,
            crop_center_x=_opt_int("presearch.crop_center_x", d.get("crop_center_x")),
            crop_center_y=_opt_int("presearch.crop_center_y", d.get("crop_center_y")),
            crop_size_x=_opt_int("presearch.crop_size_x", d.get("crop_size_x")),
            crop_size_y=_opt_int("presearch.crop_size_y", d.get("crop_size_y")),
            crop_source=_opt_str("presearch.crop_source", d.get("crop_source")),
        )


@dataclass(frozen=True)
class MeasuredValues:
    # Position and angles are None only when success is False.
    center_position_x: Optional[float]
    center_position_y: Optional[float]
    measured_angle_a0: Optional[float]
    measured_angle_a1: Optional[float]
    rss_ratio_r0: float
    rss_ratio_r1: float
    rss_ratio_r2: float
    rss_ratio_r3: float
    brightness_ratio_b0: float
    brightness_ratio_b1: float
    brightness_ratio_b2: float
    brightness_ratio_b3: float
    linewidth_w0: float
    linewidth_w1: float
    linewidth_w2: float
    linewidth_w3: float
    overlay_hash:str
    success: bool
    message: str
    # Profile method per ROI (top, bottom, right, left): one of ROI_METHODS.
    roi_method_m0: str = "average"
    roi_method_m1: str = "average"
    roi_method_m2: str = "average"
    roi_method_m3: str = "average"
    # Fraction of pixels per ROI with any channel >= 250 (R18); None in records written before
    # the saturation check existed.
    saturation_s0: Optional[float] = None
    saturation_s1: Optional[float] = None
    saturation_s2: Optional[float] = None
    saturation_s3: Optional[float] = None
    # Deskew half-width in px used per ROI (fine_window, or >= fine_window when adaptive); None
    # when pass 2 did not run (average mode, failed pass 1) or in older records.
    fine_window_used_f0: Optional[float] = None
    fine_window_used_f1: Optional[float] = None
    fine_window_used_f2: Optional[float] = None
    fine_window_used_f3: Optional[float] = None
    # Size of the measured image in pixels; None in records written before it was stored.
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("roi_method_m0", "roi_method_m1", "roi_method_m2", "roi_method_m3"):
            _require_roi_method(f"measured_values.{name}", getattr(self, name))
        for name in ("saturation_s0", "saturation_s1", "saturation_s2", "saturation_s3"):
            _opt_fraction(f"measured_values.{name}", getattr(self, name))
        for name in ("fine_window_used_f0", "fine_window_used_f1", "fine_window_used_f2", "fine_window_used_f3"):
            _opt_window(f"measured_values.{name}", getattr(self, name))
        if self.success:
            missing = [
                name
                for name in ("center_position_x", "center_position_y", "measured_angle_a0", "measured_angle_a1")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"measured_values: {', '.join(missing)} cannot be None when success is True")

    @staticmethod
    def from_dict(d: dict) -> "MeasuredValues":
        return MeasuredValues(
            center_position_x=_opt_float("measured_values.center_position_x", d["center_position_x"]),
            center_position_y=_opt_float("measured_values.center_position_y", d["center_position_y"]),
            measured_angle_a0=_opt_float("measured_values.measured_angle_a0", d["measured_angle_a0"]),
            measured_angle_a1=_opt_float("measured_values.measured_angle_a1", d["measured_angle_a1"]),
            rss_ratio_r0=_require_float("measured_values.rss_ratio_r0", d["rss_ratio_r0"]),
            rss_ratio_r1=_require_float("measured_values.rss_ratio_r1", d["rss_ratio_r1"]),
            rss_ratio_r2=_require_float("measured_values.rss_ratio_r2", d["rss_ratio_r2"]),
            rss_ratio_r3=_require_float("measured_values.rss_ratio_r3", d["rss_ratio_r3"]),
            brightness_ratio_b0=_require_float("measured_values.brightness_ratio_b0", d["brightness_ratio_b0"]),
            brightness_ratio_b1=_require_float("measured_values.brightness_ratio_b1", d["brightness_ratio_b1"]),
            brightness_ratio_b2=_require_float("measured_values.brightness_ratio_b2", d["brightness_ratio_b2"]),
            brightness_ratio_b3=_require_float("measured_values.brightness_ratio_b3", d["brightness_ratio_b3"]),
            linewidth_w0=_require_float("measured_values.linewidth_w0", d["linewidth_w0"]),
            linewidth_w1=_require_float("measured_values.linewidth_w1", d["linewidth_w1"]),
            linewidth_w2=_require_float("measured_values.linewidth_w2", d["linewidth_w2"]),
            linewidth_w3=_require_float("measured_values.linewidth_w3", d["linewidth_w3"]),
            overlay_hash=_require_str("measured_values.overlay_hash", d["overlay_hash"]),
            success=_require_bool("success", d["success"]),
            message=_require_str("message", d["message"]),
            roi_method_m0=_require_roi_method("measured_values.roi_method_m0", d.get("roi_method_m0", "average")),
            roi_method_m1=_require_roi_method("measured_values.roi_method_m1", d.get("roi_method_m1", "average")),
            roi_method_m2=_require_roi_method("measured_values.roi_method_m2", d.get("roi_method_m2", "average")),
            roi_method_m3=_require_roi_method("measured_values.roi_method_m3", d.get("roi_method_m3", "average")),
            saturation_s0=_opt_fraction("measured_values.saturation_s0", d.get("saturation_s0")),
            saturation_s1=_opt_fraction("measured_values.saturation_s1", d.get("saturation_s1")),
            saturation_s2=_opt_fraction("measured_values.saturation_s2", d.get("saturation_s2")),
            saturation_s3=_opt_fraction("measured_values.saturation_s3", d.get("saturation_s3")),
            fine_window_used_f0=_opt_window("measured_values.fine_window_used_f0", d.get("fine_window_used_f0")),
            fine_window_used_f1=_opt_window("measured_values.fine_window_used_f1", d.get("fine_window_used_f1")),
            fine_window_used_f2=_opt_window("measured_values.fine_window_used_f2", d.get("fine_window_used_f2")),
            fine_window_used_f3=_opt_window("measured_values.fine_window_used_f3", d.get("fine_window_used_f3")),
            image_width=_opt_int("measured_values.image_width", d.get("image_width")),
            image_height=_opt_int("measured_values.image_height", d.get("image_height")),
        )


@dataclass(frozen=True)
class Recommendations:
    updated_brightness: float
    updated_crop_center_x: int
    updated_crop_center_y: int
    updated_crop_size_x: int
    updated_crop_size_y: int
    updated_roi_size_x: int
    updated_roi_size_y: int

    @staticmethod
    def from_dict(d: dict) -> "Recommendations":
        return Recommendations(
            updated_brightness=_require_float("recommendations.updated_brightness", d["updated_brightness"]),
            updated_crop_center_x=_require_int("recommendations.updated_crop_center_x", d["updated_crop_center_x"]),
            updated_crop_center_y=_require_int("recommendations.updated_crop_center_y", d["updated_crop_center_y"]),
            updated_crop_size_x=_require_int("recommendations.updated_crop_size_x", d["updated_crop_size_x"]),
            updated_crop_size_y=_require_int("recommendations.updated_crop_size_y", d["updated_crop_size_y"]),
            updated_roi_size_x=_require_int("recommendations.updated_roi_size_x", d["updated_roi_size_x"]),
            updated_roi_size_y=_require_int("recommendations.updated_roi_size_y", d["updated_roi_size_y"]),
        )

@dataclass(frozen=True)
class MeasurementHardware:
    device: Device
    image_sensor: ImageSensor
    source_optics: SourceOptics
    detect_optics: DetectOptics
    io_hardware: IOHardware
    # None in records written before capture modes existed.
    capture_mode: Optional[CaptureMode] = None

    @staticmethod
    def from_dict(d: dict) -> "MeasurementHardware":
        return MeasurementHardware(
            device=Device.from_dict(d["device"]),
            image_sensor=ImageSensor.from_dict(d["image_sensor"]),
            source_optics=SourceOptics.from_dict(d["source_optics"]),
            detect_optics=DetectOptics.from_dict(d["detect_optics"]),
            io_hardware=IOHardware.from_dict(d["io_hardware"]),
            capture_mode=CaptureMode.from_dict(d["capture_mode"]) if d.get("capture_mode") else None,
        )


@dataclass(frozen=True)
class MeasurementOutput:
    """
    One measurement: the image measurement (``measured_values``, pixel domain), the
    instrument measurement derived from it (``angular_measurement``) and a snapshot of the
    ``instrument_model`` used for the conversion.

    Records written before Phase F have no angular measurement or instrument model (None);
    their ``output_units`` / ``pixel_to_unit_scale_factor`` keys are ignored on load. Records
    written before Phase G have no ``presearch`` section (None).
    """

    hardware: MeasurementHardware
    measurement_parameters: MeasurementParameters
    quality_limits: QualityLimits
    measured_values: MeasuredValues
    recommendations: Recommendations
    measurement_name: str
    image_name: str
    image_hash: str
    angular_measurement: Optional[AngularMeasurement] = None
    instrument_model: Optional[InstrumentModel] = None
    presearch: Optional[PresearchRecord] = None

    @staticmethod
    def from_dict(d: dict) -> "MeasurementOutput":
        return MeasurementOutput(
            hardware=MeasurementHardware.from_dict(d["hardware"]),
            measurement_parameters=MeasurementParameters.from_dict(d["measurement_parameters"]),
            quality_limits=QualityLimits.from_dict(d["quality_limits"]),
            measured_values=MeasuredValues.from_dict(d["measured_values"]),
            recommendations=Recommendations.from_dict(d["recommendations"]),
            measurement_name=_require_str("measurement_output.measurement_name",d["measurement_name"]),
            image_name=_require_str("measurement_output.image_name", d["image_name"]),
            image_hash=_require_str("measurement_output.image_hash", d["image_hash"]),
            angular_measurement=(
                AngularMeasurement.from_dict(d["angular_measurement"]) if d.get("angular_measurement") else None
            ),
            instrument_model=(
                InstrumentModel.from_dict(d["instrument_model"]) if d.get("instrument_model") else None
            ),
            presearch=PresearchRecord.from_dict(d["presearch"]) if d.get("presearch") else None,
        )



