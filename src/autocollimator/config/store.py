from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import os
import yaml

import tomli
import tomli_w

from .models import (
    Device,
    DetectOptics,
    ImageSensor,
    IOHardware,
    SourceOptics,
    MeasurementParameters,
    QualityLimits,
    MeasurementOutput,
)


class ConfigError(RuntimeError):
    pass


def _load_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("rb") as f:
        data = tomli.load(f)
    return data or {}


def _save_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(tomli_w.dumps(data).encode("utf-8"))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _index_by_id(items: list, id_attr: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k = getattr(item, id_attr)
        if k in out:
            raise ConfigError(f"Duplicate {id_attr}={k}")
        out[k] = item
    return out


def resolve_env_vars(s: str) -> str:
    return os.path.expandvars(s)


def load_library(config_dir: Path) -> dict[str, Any]:
    lib = config_dir / "library"

    sensors_raw = _load_toml(lib / "image_sensor.toml").get("image_sensor", [])
    io_raw = _load_toml(lib / "io_hardware.toml").get("io_hardware", [])
    src_raw = _load_toml(lib / "source_optics.toml").get("source_optics", [])
    det_raw = _load_toml(lib / "detect_optics.toml").get("detect_optics", [])
    mp_raw = _load_toml(lib / "measurement_parameters.toml").get("measurement_parameters", [])
    ql_raw = _load_toml(lib / "quality_limits.toml").get("quality_limits", [])

    sensors = [ImageSensor.from_dict(d) for d in sensors_raw]
    io_hw = [IOHardware.from_dict(d) for d in io_raw]
    source_optics = [SourceOptics.from_dict(d) for d in src_raw]
    detect_optics = [DetectOptics.from_dict(d) for d in det_raw]
    measurement_parameters = [MeasurementParameters.from_dict(d) for d in mp_raw]
    quality_limits = [QualityLimits.from_dict(d) for d in ql_raw]

    return {
        "sensors": sensors,
        "io_hardware": io_hw,
        "source_optics": source_optics,
        "detect_optics": detect_optics,
        "measurement_parameters": measurement_parameters,
        "quality_limits": quality_limits,
        "sensors_by_id": _index_by_id(sensors, "id"),
        "io_by_id": _index_by_id(io_hw, "id"),
        "source_by_id": _index_by_id(source_optics, "id"),
        "detect_by_id": _index_by_id(detect_optics, "id"),
        "measurement_parameters_by_id": _index_by_id(measurement_parameters, "id"),
        "quality_limits_by_id": _index_by_id(quality_limits, "id"),
    }


def load_main(config_dir: Path) -> Device:
    raw = _load_toml(config_dir / "main.toml")
    if "device" not in raw:
        raise ConfigError("main.toml must contain [device] table")

    d = dict(raw["device"])
    d["encryption_key"] = resolve_env_vars(d.get("encryption_key", ""))
    return Device.from_dict(d)


def validate_device_references(device: Device, lib: dict[str, Any]) -> None:
    if device.image_sensor_id not in lib["sensors_by_id"]:
        raise ConfigError(
            f"device.image_sensor_id={device.image_sensor_id} not found in library/image_sensor.toml"
        )
    if device.source_optics_id not in lib["source_by_id"]:
        raise ConfigError(
            f"device.source_optics_id={device.source_optics_id} not found in library/source_optics.toml"
        )
    if device.io_hardware_id not in lib["io_by_id"]:
        raise ConfigError(
            f"device.io_hardware_id={device.io_hardware_id} not found in library/io_hardware.toml"
        )
    if device.detect_optics_id not in lib["detect_by_id"]:
        raise ConfigError(
            f"device.detect_optics_id={device.detect_optics_id} not found in library/detect_optics.toml"
        )
    if device.measurement_parameters_id not in lib["measurement_parameters_by_id"]:
        raise ConfigError(
            f"device.measurement_parameters_id={device.measurement_parameters_id} "
            "not found in library/measurement_parameters.toml"
        )
    if device.quality_limits_id not in lib["quality_limits_by_id"]:
        raise ConfigError(
            f"device.quality_limits_id={device.quality_limits_id} "
            "not found in library/quality_limits.toml"
        )


# ----------------------------
# Generic CRUD helpers
# ----------------------------
def _add_entry(config_path: Path, list_key: str, entry_dict: dict, id_key: str) -> None:
    raw = _load_toml(config_path)
    items = list(raw.get(list_key, []))

    new_id = entry_dict[id_key]
    if not isinstance(new_id, str):
        raise TypeError(f"{list_key} {id_key} must be str, got {type(new_id).__name__}")

    existing_ids = {d.get(id_key) for d in items if id_key in d}
    if new_id in existing_ids:
        raise ConfigError(f"Duplicate {list_key} {id_key}={new_id}")

    items.append(entry_dict)
    raw[list_key] = items
    _save_toml(config_path, raw)


def _update_entry(config_path: Path, list_key: str, entry_dict: dict, id_key: str) -> None:
    raw = _load_toml(config_path)
    items = list(raw.get(list_key, []))

    target_id = entry_dict[id_key]
    if not isinstance(target_id, str):
        raise TypeError(f"{list_key} {id_key} must be str, got {type(target_id).__name__}")

    for i, d in enumerate(items):
        if d.get(id_key) == target_id:
            items[i] = entry_dict
            raw[list_key] = items
            _save_toml(config_path, raw)
            return

    raise ConfigError(f"{list_key} {id_key}={target_id} not found (cannot update)")


def _delete_entry(config_path: Path, list_key: str, id_key: str, entry_id: str) -> None:
    raw = _load_toml(config_path)
    items = list(raw.get(list_key, []))

    if not isinstance(entry_id, str):
        raise TypeError(f"entry_id must be str, got {type(entry_id).__name__}")

    new_items = [d for d in items if d.get(id_key) != entry_id]

    if len(new_items) == len(items):
        raise ConfigError(f"{list_key} {id_key}={entry_id} not found (cannot delete)")

    raw[list_key] = new_items
    _save_toml(config_path, raw)


def _ensure_not_referenced_by_main(config_dir: Path, *, field_name: str, entry_id: str) -> None:
    try:
        device = load_main(config_dir)
    except FileNotFoundError:
        return

    current_value = getattr(device, field_name)
    if current_value == entry_id:
        raise ConfigError(
            f"Cannot delete entry {entry_id}: it is currently referenced by "
            f"device.{field_name} in main.toml"
        )


# ----------------------------
# Specific CRUD functions
# ----------------------------
def add_image_sensor(config_dir: Path, sensor: ImageSensor) -> None:
    _add_entry(
        config_dir / "library" / "image_sensor.toml",
        "image_sensor",
        asdict(sensor),
        "id",
    )


def update_image_sensor(config_dir: Path, sensor: ImageSensor) -> None:
    _update_entry(
        config_dir / "library" / "image_sensor.toml",
        "image_sensor",
        asdict(sensor),
        "id",
    )


def delete_image_sensor(config_dir: Path, sensor_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="image_sensor_id",
        entry_id=sensor_id,
    )
    _delete_entry(
        config_dir / "library" / "image_sensor.toml",
        "image_sensor",
        "id",
        sensor_id,
    )


def add_io_hardware(config_dir: Path, hw: IOHardware) -> None:
    _add_entry(
        config_dir / "library" / "io_hardware.toml",
        "io_hardware",
        asdict(hw),
        "id",
    )


def update_io_hardware(config_dir: Path, hw: IOHardware) -> None:
    _update_entry(
        config_dir / "library" / "io_hardware.toml",
        "io_hardware",
        asdict(hw),
        "id",
    )


def delete_io_hardware(config_dir: Path, io_hardware_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="io_hardware_id",
        entry_id=io_hardware_id,
    )
    _delete_entry(
        config_dir / "library" / "io_hardware.toml",
        "io_hardware",
        "id",
        io_hardware_id,
    )


def add_source_optics(config_dir: Path, src: SourceOptics) -> None:
    entry = {k: v for k, v in asdict(src).items() if v is not None}
    _add_entry(
        config_dir / "library" / "source_optics.toml",
        "source_optics",
        entry,
        "id",
    )


def update_source_optics(config_dir: Path, src: SourceOptics) -> None:
    entry = {k: v for k, v in asdict(src).items() if v is not None}
    _update_entry(
        config_dir / "library" / "source_optics.toml",
        "source_optics",
        entry,
        "id",
    )


def delete_source_optics(config_dir: Path, source_optics_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="source_optics_id",
        entry_id=source_optics_id,
    )
    _delete_entry(
        config_dir / "library" / "source_optics.toml",
        "source_optics",
        "id",
        source_optics_id,
    )


def add_detect_optics(config_dir: Path, opt: DetectOptics) -> None:
    _add_entry(
        config_dir / "library" / "detect_optics.toml",
        "detect_optics",
        asdict(opt),
        "id",
    )


def update_detect_optics(config_dir: Path, opt: DetectOptics) -> None:
    _update_entry(
        config_dir / "library" / "detect_optics.toml",
        "detect_optics",
        asdict(opt),
        "id",
    )


def delete_detect_optics(config_dir: Path, detect_optics_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="detect_optics_id",
        entry_id=detect_optics_id,
    )
    _delete_entry(
        config_dir / "library" / "detect_optics.toml",
        "detect_optics",
        "id",
        detect_optics_id,
    )


def add_measurement_parameters(config_dir: Path, params: MeasurementParameters) -> None:
    _add_entry(
        config_dir / "library" / "measurement_parameters.toml",
        "measurement_parameters",
        asdict(params),
        "id",
    )


def update_measurement_parameters(config_dir: Path, params: MeasurementParameters) -> None:
    _update_entry(
        config_dir / "library" / "measurement_parameters.toml",
        "measurement_parameters",
        asdict(params),
        "id",
    )


def delete_measurement_parameters(config_dir: Path, measurement_parameters_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="measurement_parameters_id",
        entry_id=measurement_parameters_id,
    )
    _delete_entry(
        config_dir / "library" / "measurement_parameters.toml",
        "measurement_parameters",
        "id",
        measurement_parameters_id,
    )


def add_quality_limits(config_dir: Path, ql: QualityLimits) -> None:
    _add_entry(
        config_dir / "library" / "quality_limits.toml",
        "quality_limits",
        asdict(ql),
        "id",
    )


def update_quality_limits(config_dir: Path, ql: QualityLimits) -> None:
    _update_entry(
        config_dir / "library" / "quality_limits.toml",
        "quality_limits",
        asdict(ql),
        "id",
    )


def delete_quality_limits(config_dir: Path, ql_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="quality_limits_id",
        entry_id=ql_id,
    )
    _delete_entry(
        config_dir / "library" / "quality_limits.toml",
        "quality_limits",
        "id",
        ql_id,
    )


def load_measurement_output(path: Path) -> MeasurementOutput:
    raw = _load_yaml(path)
    return MeasurementOutput.from_dict(raw)


def save_measurement_output(path: Path, measurement: MeasurementOutput) -> None:
    _save_yaml(path, asdict(measurement))




    