"""Phase F: workflow composes image -> angular measurement; mismatch policy; YAML output (R3, R5)."""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import cv2
import pytest
import yaml

from autocollimator.app import AutocollimatorApp
from autocollimator.config.store import load_measurement_output
from autocollimator.instrument import ImageSizeMismatchError, ReticleType, TiltZeroMode
from autocollimator.units import AngleUnit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


def _image(name: str):
    img = cv2.imread(str(IMAGE_DIR / name))
    if img is None:
        pytest.fail(f"Cannot read image: {name}")
    return img


def _configs_with(tmp_path: Path, **device_keys: str) -> Path:
    """Copy configs/ and override [device] keys in main.toml (values are TOML literals)."""
    dst = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dst)
    path = dst / "main.toml"
    lines = path.read_text().splitlines()
    for key, value in device_keys.items():
        lines = [ln for ln in lines if not ln.replace(" ", "").startswith(f"{key}=")]
        lines.insert(1, f"{key} = {value}")
    path.write_text("\n".join(lines) + "\n")
    return dst


@pytest.mark.integration
def test_size_mismatch_error_policy(tmp_path) -> None:
    config_dir = _configs_with(tmp_path, capture_mode_id='"native"', image_size_mismatch='"error"')
    with AutocollimatorApp(config_dir=config_dir, output_dir=tmp_path / "out") as app:
        with pytest.raises(ImageSizeMismatchError, match="1920x1080.*native.*1456x1088"):
            app.run_center_finding_on_image(_image("blob-05.jpg"))


@pytest.mark.integration
def test_size_mismatch_warn_policy_uses_image_center(tmp_path, caplog) -> None:
    config_dir = _configs_with(tmp_path, capture_mode_id='"native"', image_size_mismatch='"warn"')
    with caplog.at_level(logging.WARNING, logger="autocollimator.instrument"):
        with AutocollimatorApp(config_dir=config_dir, output_dir=tmp_path / "out") as app:
            result, _overlay = app.run_center_finding_on_image(_image("blob-05.jpg"), tilt_unit="rad")
    assert "does not match capture mode 'native'" in caplog.text

    mv, am = result.measured_values, result.angular_measurement
    assert mv.success and am.success
    # Center of the 1920x1080 image itself, not of the 1456x1088 capture mode.
    assert (am.tilt_zero_x_px, am.tilt_zero_y_px) == (959.5, 539.5)
    # Native pitch 3.45 um, f = 200 mm.
    dx_mm = (mv.center_position_x - 959.5) * 3.45e-3
    dy_mm = -(mv.center_position_y - 539.5) * 3.45e-3
    assert am.tilt_x == pytest.approx(0.5 * math.atan(dx_mm / 200.0), rel=1e-12)
    assert am.tilt_y == pytest.approx(0.5 * math.atan(dy_mm / 200.0), rel=1e-12)
    assert result.instrument_model.pixel_pitch_x_um == pytest.approx(3.45, rel=1e-15)


@pytest.mark.integration
def test_default_config_hd_1080_angular_measurement(tmp_path) -> None:
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        result, _overlay = app.run_center_finding_on_image(
            _image("blob-06.jpg"), tilt_unit="arcsec", rotation_unit="deg"
        )
    mv, am, model = result.measured_values, result.angular_measurement, result.instrument_model
    assert (mv.image_width, mv.image_height) == (1920, 1080)
    assert (am.tilt_unit, am.rotation_unit) == (AngleUnit.ARCSEC, AngleUnit.DEG)
    assert am.tilt_zero_mode is TiltZeroMode.IMAGE_CENTER
    assert model.capture_mode_id == "hd_1080" and model.collimator_efl_mm == 200.0
    # R3: blob-06's vertical line used to be reported as about 178.5 deg.
    assert -45.0 < mv.measured_angle_a0 <= 45.0 and -45.0 < mv.measured_angle_a1 <= 45.0
    assert mv.measured_angle_a0 == pytest.approx(-1.45, abs=0.05)
    dx_mm = (mv.center_position_x - 959.5) * 2.61625e-3
    assert am.tilt_x == pytest.approx(math.degrees(0.5 * math.atan(dx_mm / 200.0)) * 3600.0, rel=1e-12)
    assert am.spot_dx_mm == pytest.approx(dx_mm, rel=1e-12)


@pytest.mark.integration
def test_failed_image_measurement_gives_failed_angular_measurement(tmp_path) -> None:
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        result, _overlay = app.run_center_finding_on_image(_image("blob-09.jpg"))
    am = result.angular_measurement
    assert result.measured_values.success is False and am.success is False
    assert (am.tilt_x, am.tilt_y, am.rotation, am.orthogonality) == (None, None, None, None)


@pytest.mark.integration
def test_measurement_yaml_round_trip(tmp_path) -> None:
    img = _image("blob-05.jpg")
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        result, overlay = app.run_center_finding_on_image(img, tilt_unit="mrad")
        _in, _ov, yaml_path = app.save_measurement_output(result, img, overlay)

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert raw["angular_measurement"]["tilt_unit"] == "mrad"
    assert raw["angular_measurement"]["rotation_unit"] == "deg"
    assert raw["instrument_model"]["reticle_type"] == "crosshair"
    assert raw["instrument_model"]["collimator_efl_mm"] == 200.0
    assert raw["instrument_model"]["capture_mode_id"] == "hd_1080"
    assert raw["instrument_model"]["tilt_zero"]["mode"] == "image_center"
    assert raw["hardware"]["capture_mode"]["crop_y"] == 134
    assert "output_units" not in raw

    reloaded = load_measurement_output(yaml_path)
    assert reloaded.measured_values == result.measured_values
    assert reloaded.angular_measurement == result.angular_measurement
    assert reloaded.instrument_model == result.instrument_model
    assert reloaded.instrument_model.reticle_type is ReticleType.CROSSHAIR


@pytest.mark.integration
def test_old_measurement_file_still_loads(tmp_path) -> None:
    img = _image("blob-05.jpg")
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        result, overlay = app.run_center_finding_on_image(img)
        _in, _ov, yaml_path = app.save_measurement_output(result, img, overlay)

    # Rewrite as a pre-Phase-F record: efl1/efl2, output_units, no angular/instrument data.
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for key in ("angular_measurement", "instrument_model"):
        raw.pop(key)
    raw["hardware"].pop("capture_mode")
    for key in ("capture_mode_id", "reticle_type", "image_size_mismatch", "tilt_unit", "rotation_unit",
                "tilt_zero_mode", "tilt_zero_x_px", "tilt_zero_y_px", "rotation_zero_mode", "rotation_zero_deg"):
        raw["hardware"]["device"].pop(key)
    optics = raw["hardware"]["detect_optics"]
    optics["efl1"] = optics.pop("collimator_efl_mm")
    optics["efl2"] = optics.pop("microscope_efl_mm")
    raw["measured_values"].pop("image_width")
    raw["measured_values"].pop("image_height")
    raw["output_units"] = "Pixels"
    raw["pixel_to_unit_scale_factor"] = 1.0
    old_path = tmp_path / "old_measurement.yaml"
    old_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="efl"):
        old = load_measurement_output(old_path)
    assert old.angular_measurement is None and old.instrument_model is None
    assert old.hardware.detect_optics.collimator_efl_mm == 200.0
    assert old.measured_values.center_position_x == result.measured_values.center_position_x
    assert old.measured_values.image_width is None
