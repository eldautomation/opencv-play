from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any
import os
import re
import sys
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
import tomli_w
import cv2

from dataclasses import replace

from .models import (
    CaptureMode,
    Device,
    DetectOptics,
    ImageSensor,
    IOHardware,
    SourceOptics,
    MeasurementParameters,
    QualityLimits,
    MeasurementOutput,
    ROI_LENGTH_AUTO,
)


class ConfigError(RuntimeError):
    pass


def _load_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
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


def _yaml_safe_dict(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """dict_factory for asdict: store Enums (units, modes, reticle type) by their config value."""
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in items}


def _index_by_id(items: list, id_attr: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k = getattr(item, id_attr)
        if k in out:
            raise ConfigError(f"Duplicate {id_attr}={k}")
        out[k] = item
    return out

def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_image(path: Path, image) -> None:
    """
    Save an image array to disk.

    Raises
    ------
    ValueError
        If OpenCV fails to write the file.
    """
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")
    

def resolve_env_vars(s: str) -> str:
    return os.path.expandvars(s)


_ENV_REF = re.compile(r"\$\{([^}]*)\}")
DEFAULT_KEY_ENV_VAR = "DEVICE_ENCRYPTION_KEY"


def _resolve_encryption_key(raw_value, config_path: Path) -> str:
    """
    Expand environment variables in device.encryption_key and require a real key.

    Raises ConfigError naming the environment variable(s) to set if the result is empty
    or still contains an unexpanded ${...}. The key value itself is never included.
    """
    if not isinstance(raw_value, str):
        raise ConfigError(f"device.encryption_key in {config_path} must be a string")

    key = resolve_env_vars(raw_value)
    unresolved = _ENV_REF.findall(key)
    if key.strip() and not unresolved:
        return key

    names = unresolved or _ENV_REF.findall(raw_value) or [DEFAULT_KEY_ENV_VAR]
    var_list = ", ".join(names)
    raise ConfigError(
        f"device.encryption_key in {config_path} is not set: set the environment variable "
        f"{var_list} before starting (e.g. export {names[0]}=...)."
    )


def load_library(config_dir: Path) -> dict[str, Any]:
    lib = config_dir / "library"

    sensors_raw = _load_toml(lib / "image_sensor.toml").get("image_sensor", [])
    io_raw = _load_toml(lib / "io_hardware.toml").get("io_hardware", [])
    src_raw = _load_toml(lib / "source_optics.toml").get("source_optics", [])
    det_raw = _load_toml(lib / "detect_optics.toml").get("detect_optics", [])
    mp_raw = _load_toml(lib / "measurement_parameters.toml").get("measurement_parameters", [])
    ql_raw = _load_toml(lib / "quality_limits.toml").get("quality_limits", [])
    cm_raw = _load_toml(lib / "capture_mode.toml").get("capture_mode", [])

    sensors = [ImageSensor.from_dict(d) for d in sensors_raw]
    io_hw = [IOHardware.from_dict(d) for d in io_raw]
    source_optics = [SourceOptics.from_dict(d) for d in src_raw]
    detect_optics = [DetectOptics.from_dict(d) for d in det_raw]
    measurement_parameters = [MeasurementParameters.from_dict(d) for d in mp_raw]
    quality_limits = [QualityLimits.from_dict(d) for d in ql_raw]
    capture_modes = [CaptureMode.from_dict(d) for d in cm_raw]

    return {
        "sensors": sensors,
        "io_hardware": io_hw,
        "source_optics": source_optics,
        "detect_optics": detect_optics,
        "measurement_parameters": measurement_parameters,
        "quality_limits": quality_limits,
        "capture_modes": capture_modes,
        "sensors_by_id": _index_by_id(sensors, "id"),
        "io_by_id": _index_by_id(io_hw, "id"),
        "source_by_id": _index_by_id(source_optics, "id"),
        "detect_by_id": _index_by_id(detect_optics, "id"),
        "measurement_parameters_by_id": _index_by_id(measurement_parameters, "id"),
        "quality_limits_by_id": _index_by_id(quality_limits, "id"),
        "capture_modes_by_id": _index_by_id(capture_modes, "id"),
    }


def load_main(config_dir: Path) -> Device:
    raw = _load_toml(config_dir / "main.toml")
    if "device" not in raw:
        raise ConfigError("main.toml must contain [device] table")

    d = dict(raw["device"])
    d["encryption_key"] = _resolve_encryption_key(d.get("encryption_key", ""), config_dir / "main.toml")
    if "capture_mode_id" not in d:
        raise ConfigError(f"device.capture_mode_id is missing in {config_dir / 'main.toml'}")
    return Device.from_dict(d)


def update_main_device_fields(config_dir: Path, fields: dict[str, Any]) -> None:
    """
    Set (or, with value None, remove) keys of the [device] table in main.toml.

    Edits the raw file line by line, so other keys, comments and the unexpanded
    ``encryption_key = "${...}"`` reference are kept; the resolved key is never written.
    The result is parsed and validated before the file is replaced.
    """
    path = config_dir / "main.toml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    if "encryption_key" in fields:
        raise ConfigError("device.encryption_key cannot be changed through update_main_device_fields")
    with path.open("r", encoding="utf-8", newline="") as f:
        text = f.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)

    table = re.compile(r"^\s*\[")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "[device]")
    except StopIteration:
        raise ConfigError(f"{path} must contain [device] table") from None

    for key, value in fields.items():
        end = next((i for i in range(start + 1, len(lines)) if table.match(lines[i])), len(lines))
        key_line = re.compile(rf"^\s*{re.escape(key)}\s*=")
        idx = next((i for i in range(start + 1, end) if key_line.match(lines[i])), None)
        if value is None:
            if idx is not None:
                del lines[idx]
            continue
        rendered = tomli_w.dumps({key: value}).strip() + newline
        if idx is not None:
            lines[idx] = rendered
        else:
            # Append after the last non-blank line of the [device] table.
            last = max((i for i in range(start, end) if lines[i].strip()), default=start)
            if not lines[last].endswith(("\n", "\r")):
                lines[last] += newline
            lines.insert(last + 1, rendered)

    new_text = "".join(lines)
    device = dict(tomllib.loads(new_text).get("device", {}))
    device["encryption_key"] = device.get("encryption_key") or "unchecked"
    Device.from_dict(device)  # raises before anything is written if the result is invalid

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(new_text)


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
    mp = lib["measurement_parameters_by_id"][device.measurement_parameters_id]
    if (
        mp.roi_size_x != ROI_LENGTH_AUTO
        and mp.presearch.presearch_mode != "off"
        and mp.roi_size_x > mp.presearch.min_crop_size_px
    ):
        # A pre-search crop is at least min_crop_size_px; a longer explicit ROI would not fit.
        raise ConfigError(
            f"measurement_parameters {mp.id}: roi_size_x ({mp.roi_size_x}) must not be larger than "
            f"presearch.min_crop_size_px ({mp.presearch.min_crop_size_px}) when the pre-search is on "
            f"(use roi_size_x = \"{ROI_LENGTH_AUTO}\")"
        )
    if device.quality_limits_id not in lib["quality_limits_by_id"]:
        raise ConfigError(
            f"device.quality_limits_id={device.quality_limits_id} "
            "not found in library/quality_limits.toml"
        )
    if device.capture_mode_id not in lib["capture_modes_by_id"]:
        raise ConfigError(
            f"device.capture_mode_id={device.capture_mode_id} not found in library/capture_mode.toml"
        )
    capture_mode = lib["capture_modes_by_id"][device.capture_mode_id]
    sensor = lib["sensors_by_id"][device.image_sensor_id]
    if not capture_mode.fits_sensor(sensor):
        raise ConfigError(
            f"capture_mode {capture_mode.id}: crop ({capture_mode.crop_x}, {capture_mode.crop_y}, "
            f"{capture_mode.crop_width}, {capture_mode.crop_height}) does not fit sensor "
            f"{sensor.id} ({sensor.pixels_x}x{sensor.pixels_y})"
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


def add_capture_mode(config_dir: Path, mode: CaptureMode) -> None:
    _add_entry(
        config_dir / "library" / "capture_mode.toml",
        "capture_mode",
        asdict(mode),
        "id",
    )


def update_capture_mode(config_dir: Path, mode: CaptureMode) -> None:
    _update_entry(
        config_dir / "library" / "capture_mode.toml",
        "capture_mode",
        asdict(mode),
        "id",
    )


def delete_capture_mode(config_dir: Path, capture_mode_id: str) -> None:
    _ensure_not_referenced_by_main(
        config_dir,
        field_name="capture_mode_id",
        entry_id=capture_mode_id,
    )
    _delete_entry(
        config_dir / "library" / "capture_mode.toml",
        "capture_mode",
        "id",
        capture_mode_id,
    )


def load_measurement_output(path: Path) -> MeasurementOutput:
    raw = _load_yaml(path)
    return MeasurementOutput.from_dict(raw)


def save_measurement_output(
    output_dir: Path,
    prefix: str,
    measurement: MeasurementOutput,
    input_image,
    overlay_image,
) -> tuple[Path, Path, Path]:
    """
    Save a measurement bundle using a shared prefix.

    Files written:
    - {prefix}_input.png
    - {prefix}_overlay.png
    - {prefix}_measurement.yaml

    Parameters
    ----------
    output_dir:
        Folder where files should be written.
    prefix:
        Shared filename prefix.
    measurement:
        MeasurementOutput dataclass to persist.
    input_image:
        Raw source image as a NumPy array.
    overlay_image:
        Overlay/debug image as a NumPy array.

    Returns
    -------
    tuple[Path, Path, Path]
        Paths to (input_image_path, overlay_image_path, measurement_yaml_path)
    """
    _ensure_output_dir(output_dir)

    input_path = output_dir / f"{prefix}_input.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    measurement_path = output_dir / f"{prefix}_measurement.yaml"

    _write_image(input_path, input_image)
    _write_image(overlay_path, overlay_image)

    measurement = replace(
            measurement,
            measurement_name=prefix,
        )

    # Persist a YAML-safe measurement object.
    # Current schema does not store image paths, only hashes and metadata,
    # so we keep the measurement as-is and write it after images are saved.
    # The device encryption key is a secret and is never written (R6).
    data = asdict(measurement, dict_factory=_yaml_safe_dict)
    data["hardware"]["device"].pop("encryption_key", None)
    _save_yaml(measurement_path, data)

    return input_path, overlay_path, measurement_path
