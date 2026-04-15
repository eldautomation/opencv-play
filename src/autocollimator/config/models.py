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


def _require_bool(name: str, v) -> bool:
    if not isinstance(v, bool):
        raise TypeError(f"{name} must be bool, got {type(v).__name__}")
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


@dataclass(frozen=True)
class DetectOptics:
    id: str
    name: str
    efl1: float
    efl2: float

    @staticmethod
    def from_dict(d: dict) -> "DetectOptics":
        return DetectOptics(
            id=_require_str("detect_optics.id", d["id"]),
            name=_require_str("detect_optics.name", d["name"]),
            efl1=_require_float("detect_optics.efl1", d["efl1"]),
            efl2=_require_float("detect_optics.efl2", d["efl2"]),
        )


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    serial_number: str
    encryption_key: str
    image_sensor_id: str
    source_optics_id: str
    io_hardware_id: str
    detect_optics_id: str
    measurement_parameters_id: str
    quality_limits_id:str

    @staticmethod
    def from_dict(d: dict) -> "Device":
        return Device(
            id=_require_str("device.id", d["id"]),
            name=_require_str("device.name", d["name"]),
            serial_number=_require_str("device.serial_number", d["serial_number"]),
            encryption_key=_require_str("device.encryption_key", d["encryption_key"]),
            image_sensor_id=_require_str("device.image_sensor_id", d["image_sensor_id"]),
            source_optics_id=_require_str("device.source_optics_id", d["source_optics_id"]),
            io_hardware_id=_require_str("device.io_hardware_id", d["io_hardware_id"]),
            detect_optics_id=_require_str("device.detect_optics_id", d["detect_optics_id"]),
            measurement_parameters_id=_require_str("device.measurement_parameters_id", d["measurement_parameters_id"]),
            quality_limits_id=_require_str("device.quality_limits_id",d["quality_limits_id"]),
        )



@dataclass(frozen=True)
class MeasurementParameters:
    id:str
    name:str
    crop_center_x: int
    crop_center_y: int
    crop_size_x: int
    crop_size_y: int
    roi_size_x: int
    roi_size_y: int
    
    @staticmethod
    def from_dict(d: dict) -> "MeasurementParameters":
        return MeasurementParameters(
            id=_require_str("measurement_parameters.id",d["id"]),
            name=_require_str("measurement_parameters.name",d["name"]),
            crop_center_x=_require_int("measurement_parameters.crop_center_x", d["crop_center_x"]),
            crop_center_y=_require_int("measurement_parameters.crop_center_y", d["crop_center_y"]),
            crop_size_x=_require_int("measurement_parameters.crop_size_x", d["crop_size_x"]),
            crop_size_y=_require_int("measurement_parameters.crop_size_y", d["crop_size_y"]),
            roi_size_x=_require_int("measurement_parameters.roi_size_x", d["roi_size_x"]),
            roi_size_y=_require_int("measurement_parameters.roi_size_y", d["roi_size_y"]),
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
class MeasuredValues:
    center_position_x: float
    center_position_y: float
    measured_angle_a0: float
    measured_angle_a1: float
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

    @staticmethod
    def from_dict(d: dict) -> "MeasuredValues":
        return MeasuredValues(
            center_position_x=_require_float("measured_values.center_position_x", d["center_position_x"]),
            center_position_y=_require_float("measured_values.center_position_y", d["center_position_y"]),
            measured_angle_a0=_require_float("measured_values.measured_angle_a0", d["measured_angle_a0"]),
            measured_angle_a1=_require_float("measured_values.measured_angle_a1", d["measured_angle_a1"]),
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

    @staticmethod
    def from_dict(d: dict) -> "MeasurementHardware":
        return MeasurementHardware(
            device=Device.from_dict(d["device"]),
            image_sensor=ImageSensor.from_dict(d["image_sensor"]),
            source_optics=SourceOptics.from_dict(d["source_optics"]),
            detect_optics=DetectOptics.from_dict(d["detect_optics"]),
            io_hardware=IOHardware.from_dict(d["io_hardware"]),
        )


@dataclass(frozen=True)
class MeasurementOutput:
    hardware: MeasurementHardware
    measurement_parameters: MeasurementParameters
    quality_limits: QualityLimits
    measured_values: MeasuredValues
    recommendations: Recommendations
    measurement_name: str
    image_name: str
    image_hash: str
    output_units: str
    pixel_to_unit_scale_factor: float

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
            output_units=_require_str("measurement_output.output_units", d["output_units"]),
            pixel_to_unit_scale_factor=_require_float(
                "measurement_output.pixel_to_unit_scale_factor",
                d["pixel_to_unit_scale_factor"],
            ),
        )



