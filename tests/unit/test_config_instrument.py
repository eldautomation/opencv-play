"""Phase F config (R5): detect optics rename, product optics, capture modes, device options."""
import dataclasses
import shutil
from pathlib import Path

import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.config.models import CaptureMode, DetectOptics, Device, ImageSensor
from autocollimator.config.store import ConfigError, load_library, load_main, validate_device_references
from autocollimator.instrument import ReticleType
from autocollimator.units import AngleUnit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

IMX296 = ImageSensor("101", "IMX296", "Sony", "IMX296LQR-C", 3.45, 3.45, 1456, 1088)


def _copy_configs(tmp_path: Path) -> Path:
    dst = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dst)
    return dst


def _set_main(config_dir: Path, key: str, value: str | None) -> None:
    """Replace (or with value=None remove) one key line in main.toml."""
    path = config_dir / "main.toml"
    lines = [ln for ln in path.read_text().splitlines() if not ln.replace(" ", "").startswith(f"{key}=")]
    if value is not None:
        lines.insert(1, f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_old_efl_keys_load_with_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning) as record:
        opt = DetectOptics.from_dict({"id": "x", "name": "old", "efl1": 200.0, "efl2": 45.0})
    messages = [str(w.message) for w in record]
    assert any("'efl1' is deprecated, use 'collimator_efl_mm'" in m for m in messages)
    assert any("'efl2' is deprecated, use 'microscope_efl_mm'" in m for m in messages)
    assert (opt.collimator_efl_mm, opt.microscope_efl_mm) == (200.0, 45.0)


@pytest.mark.unit
def test_old_and_new_efl_keys_together_rejected() -> None:
    with pytest.raises(ValueError, match="efl1"):
        DetectOptics.from_dict({"id": "x", "name": "n", "efl1": 1.0, "collimator_efl_mm": 2.0, "efl2": 3.0})


@pytest.mark.unit
def test_both_product_configurations_load() -> None:
    lib = load_library(CONFIG_DIR)
    by_efl = {o.collimator_efl_mm: o for o in lib["detect_optics"]}
    assert by_efl[200.0].microscope_efl_mm == 45.0
    assert by_efl[400.0].microscope_efl_mm == 90.0


@pytest.mark.unit
def test_default_device_uses_200mm_and_hd_1080() -> None:
    device = load_main(CONFIG_DIR)
    lib = load_library(CONFIG_DIR)
    assert lib["detect_by_id"][device.detect_optics_id].collimator_efl_mm == 200.0
    assert device.capture_mode_id == "hd_1080"
    assert device.reticle_type == "crosshair"
    assert device.image_size_mismatch == "error"
    assert (device.tilt_unit, device.rotation_unit) == ("deg", "deg")


@pytest.mark.unit
def test_effective_pitch_both_capture_modes() -> None:
    modes = load_library(CONFIG_DIR)["capture_modes_by_id"]
    sensor = load_library(CONFIG_DIR)["sensors_by_id"]["101"]
    assert (sensor.pixel_size_x, sensor.pixels_x, sensor.pixels_y) == (3.45, 1456, 1088)
    assert modes["native"].effective_pixel_pitch_um(sensor) == pytest.approx((3.45, 3.45), rel=1e-15)
    # 3.45 * 1456 / 1920 = 2.61625 and 3.45 * 819 / 1080 = 2.61625
    assert modes["hd_1080"].effective_pixel_pitch_um(sensor) == pytest.approx((2.61625, 2.61625), rel=1e-12)
    assert (modes["hd_1080"].crop_x, modes["hd_1080"].crop_y) == (0, 134)


@pytest.mark.unit
def test_instrument_model_from_config(tmp_path) -> None:
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        model = app.get_instrument_model()
        units = app.get_output_units()
        override = app.get_output_units(tilt_unit="mrad")
    assert model.reticle_type is ReticleType.CROSSHAIR
    assert (model.collimator_efl_mm, model.microscope_efl_mm) == (200.0, 45.0)
    assert (model.pixel_pitch_x_um, model.pixel_pitch_y_um) == pytest.approx((2.61625, 2.61625), rel=1e-12)
    assert (model.expected_image_width, model.expected_image_height) == (1920, 1080)
    assert model.capture_mode_id == "hd_1080"
    assert (units.tilt_unit, units.rotation_unit) == (AngleUnit.DEG, AngleUnit.DEG)
    assert (override.tilt_unit, override.rotation_unit) == (AngleUnit.MRAD, AngleUnit.DEG)


@pytest.mark.unit
@pytest.mark.parametrize(
    "key, value, match",
    [
        ("reticle_type", '"ring"', "reticle_type"),
        ("tilt_unit", '"degrees"', "tilt_unit"),
        ("rotation_unit", '"gon"', "rotation_unit"),
        ("image_size_mismatch", '"ignore"', "image_size_mismatch"),
    ],
)
def test_invalid_device_options_rejected(tmp_path, key, value, match) -> None:
    config_dir = _copy_configs(tmp_path)
    _set_main(config_dir, key, value)
    with pytest.raises(ValueError, match=match):
        load_main(config_dir)


@pytest.mark.unit
def test_missing_capture_mode_rejected(tmp_path) -> None:
    config_dir = _copy_configs(tmp_path)
    _set_main(config_dir, "capture_mode_id", None)
    with pytest.raises(ConfigError, match="capture_mode_id"):
        load_main(config_dir)


@pytest.mark.unit
def test_unknown_capture_mode_rejected(tmp_path) -> None:
    config_dir = _copy_configs(tmp_path)
    _set_main(config_dir, "capture_mode_id", '"4k"')
    with pytest.raises(ConfigError, match="capture_mode_id=4k"):
        validate_device_references(load_main(config_dir), load_library(config_dir))


@pytest.mark.unit
def test_crop_outside_sensor_rejected(tmp_path) -> None:
    config_dir = _copy_configs(tmp_path)
    path = config_dir / "library" / "capture_mode.toml"
    path.write_text(path.read_text().replace("crop_y = 134", "crop_y = 300"))
    with pytest.raises(ConfigError, match="does not fit sensor"):
        validate_device_references(load_main(config_dir), load_library(config_dir))


@pytest.mark.unit
def test_capture_mode_validation() -> None:
    mode = CaptureMode("m", "m", 1920, 1080, 0, 134, 1456, 819)
    assert mode.fits_sensor(IMX296)
    assert not dataclasses.replace(mode, crop_y=270).fits_sensor(IMX296)
    with pytest.raises(ValueError, match="output_width"):
        dataclasses.replace(mode, output_width=0)


@pytest.mark.unit
def test_old_device_record_loads_with_defaults() -> None:
    d = {
        "id": "001", "name": "n", "serial_number": "s", "image_sensor_id": "101", "source_optics_id": "201",
        "io_hardware_id": "401", "detect_optics_id": "301", "measurement_parameters_id": "901",
        "quality_limits_id": "801",
    }
    device = Device.from_dict(d)
    assert device.capture_mode_id is None
    assert (device.reticle_type, device.tilt_unit, device.image_size_mismatch) == ("crosshair", "deg", "error")
