from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import os

import tomli
import tomli_w

from .models import Device, DetectOptics, ImageSensor, IOHardware, SourceOptics


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


def _index_by_id(items: list, id_attr: str) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for item in items:
        k = getattr(item, id_attr)
        if k in out:
            raise ConfigError(f"Duplicate {id_attr}={k}")
        out[k] = item
    return out


def resolve_env_vars(s: str) -> str:
    # Supports "${VAR}" expansion
    return os.path.expandvars(s)


def load_library(config_dir: Path) -> dict[str, Any]:
    lib = config_dir / "library"

    sensors_raw = _load_toml(lib / "image_sensor.toml").get("image_sensor", [])
    io_raw = _load_toml(lib / "io_hardware.toml").get("io_hardware", [])
    src_raw = _load_toml(lib / "source_optics.toml").get("source_optics", [])
    det_raw = _load_toml(lib / "detect_optics.toml").get("detect_optics", [])

    sensors = [ImageSensor.from_dict(d) for d in sensors_raw]
    io_hw = [IOHardware.from_dict(d) for d in io_raw]
    source_optics = [SourceOptics.from_dict(d) for d in src_raw]
    detect_optics = [DetectOptics.from_dict(d) for d in det_raw]

    return {
        "sensors": sensors,
        "io_hardware": io_hw,
        "source_optics": source_optics,
        "detect_optics": detect_optics,
        "sensors_by_id": _index_by_id(sensors, "id"),
        "io_by_id": _index_by_id(io_hw, "id"),
        "source_by_id": _index_by_id(source_optics, "id"),
        "detect_by_id": _index_by_id(detect_optics, "optic_id"),
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
        raise ConfigError(f"device.image_sensor_id={device.image_sensor_id} not found in library/image_sensor.toml")
    if device.source_optics_id not in lib["source_by_id"]:
        raise ConfigError(f"device.source_optics_id={device.source_optics_id} not found in library/source_optics.toml")
    if device.io_hardware_id not in lib["io_by_id"]:
        raise ConfigError(f"device.io_hardware_id={device.io_hardware_id} not found in library/io_hardware.toml")
    if device.detect_optics_id not in lib["detect_by_id"]:
        raise ConfigError(f"device.detect_optics_id={device.detect_optics_id} not found in library/detect_optics.toml")


# ----------------------------
# Generic CRUD helpers
# ----------------------------
def _add_entry(config_path: Path, list_key: str, entry_dict: dict, id_key: str) -> None:
    raw = _load_toml(config_path)
    items = list(raw.get(list_key, []))

    new_id = entry_dict[id_key]
    existing_ids = {int(d.get(id_key)) for d in items if id_key in d}
    if int(new_id) in existing_ids:
        raise ConfigError(f"Duplicate {list_key} {id_key}={new_id}")

    items.append(entry_dict)
    raw[list_key] = items
    _save_toml(config_path, raw)


def _update_entry(config_path: Path, list_key: str, entry_dict: dict, id_key: str) -> None:
    raw = _load_toml(config_path)
    items = list(raw.get(list_key, []))

    target_id = int(entry_dict[id_key])
    for i, d in enumerate(items):
        if int(d.get(id_key)) == target_id:
            items[i] = entry_dict
            raw[list_key] = items
            _save_toml(config_path, raw)
            return

    raise ConfigError(f"{list_key} {id_key}={target_id} not found (cannot update)")


# ----------------------------
# Specific CRUD functions
# ----------------------------
def add_image_sensor(config_dir: Path, sensor: ImageSensor) -> None:
    _add_entry(config_dir / "library" / "image_sensor.toml", "image_sensor", asdict(sensor), "id")


def update_image_sensor(config_dir: Path, sensor: ImageSensor) -> None:
    _update_entry(config_dir / "library" / "image_sensor.toml", "image_sensor", asdict(sensor), "id")


def add_io_hardware(config_dir: Path, hw: IOHardware) -> None:
    _add_entry(config_dir / "library" / "io_hardware.toml", "io_hardware", asdict(hw), "id")


def update_io_hardware(config_dir: Path, hw: IOHardware) -> None:
    _update_entry(config_dir / "library" / "io_hardware.toml", "io_hardware", asdict(hw), "id")


def add_source_optics(config_dir: Path, src: SourceOptics) -> None:
    # Drop None fields when writing (TOML has no null)
    d = {k: v for k, v in asdict(src).items() if v is not None}
    _add_entry(config_dir / "library" / "source_optics.toml", "source_optics", d, "id")


def update_source_optics(config_dir: Path, src: SourceOptics) -> None:
    d = {k: v for k, v in asdict(src).items() if v is not None}
    _update_entry(config_dir / "library" / "source_optics.toml", "source_optics", d, "id")


def add_detect_optics(config_dir: Path, opt: DetectOptics) -> None:
    _add_entry(config_dir / "library" / "detect_optics.toml", "detect_optics", asdict(opt), "optic_id")


def update_detect_optics(config_dir: Path, opt: DetectOptics) -> None:
    _update_entry(config_dir / "library" / "detect_optics.toml", "detect_optics", asdict(opt), "optic_id")