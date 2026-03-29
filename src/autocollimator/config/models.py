from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _require_int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v).__name__}")
    return v


def _require_float(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    return float(v)


def _require_str(name: str, v) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    if not v.strip():
        raise ValueError(f"{name} cannot be empty")
    return v


def _opt_float_from_key(d: dict, key: str, name: str) -> Optional[float]:
    # TOML has no null; treat missing key as None.
    if key not in d:
        return None
    return _require_float(name, d[key])


@dataclass(frozen=True)
class ImageSensor:
    id: int
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
            id=_require_int("image_sensor.id", d["id"]),
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
    id: int
    name: str
    description: str

    @staticmethod
    def from_dict(d: dict) -> "IOHardware":
        return IOHardware(
            id=_require_int("io_hardware.id", d["id"]),
            name=_require_str("io_hardware.name", d["name"]),
            description=_require_str("io_hardware.description", d["description"]),
        )


@dataclass(frozen=True)
class SourceOptics:
    id: int
    name: str
    line_thickness: float
    feature_a_x: Optional[float]
    feature_a_y: Optional[float]
    efl: Optional[float]

    @staticmethod
    def from_dict(d: dict) -> "SourceOptics":
        return SourceOptics(
            id=_require_int("source_optics.id", d["id"]),
            name=_require_str("source_optics.name", d["name"]),
            line_thickness=_require_float("source_optics.line_thickness", d["line_thickness"]),
            feature_a_x=_opt_float_from_key(d, "feature_a_x", "source_optics.feature_a_x"),
            feature_a_y=_opt_float_from_key(d, "feature_a_y", "source_optics.feature_a_y"),
            efl=_opt_float_from_key(d, "efl", "source_optics.efl"),
        )


@dataclass(frozen=True)
class DetectOptics:
    optic_id: int
    name: str
    efl1: float
    efl2: float

    @staticmethod
    def from_dict(d: dict) -> "DetectOptics":
        return DetectOptics(
            optic_id=_require_int("detect_optics.optic_id", d["optic_id"]),
            name=_require_str("detect_optics.name", d["name"]),
            efl1=_require_float("detect_optics.efl1", d["efl1"]),
            efl2=_require_float("detect_optics.efl2", d["efl2"]),
        )


@dataclass(frozen=True)
class Device:
    id: int
    name: str
    serial_number: str
    encryption_key: str
    image_sensor_id: int
    source_optics_id: int
    io_hardware_id: int
    detect_optics_id: int

    @staticmethod
    def from_dict(d: dict) -> "Device":
        return Device(
            id=_require_int("device.id", d["id"]),
            name=_require_str("device.name", d["name"]),
            serial_number=_require_str("device.serial_number", d["serial_number"]),
            encryption_key=_require_str("device.encryption_key", d["encryption_key"]),
            image_sensor_id=_require_int("device.image_sensor_id", d["image_sensor_id"]),
            source_optics_id=_require_int("device.source_optics_id", d["source_optics_id"]),
            io_hardware_id=_require_int("device.io_hardware_id", d["io_hardware_id"]),
            detect_optics_id=_require_int("device.detect_optics_id", d["detect_optics_id"]),
        )
