"""Phase F: tilt and rotation zero references stored in the device config, set independently."""
import shutil
from pathlib import Path

import pytest

from autocollimator.app import AutocollimatorApp, MeasurementFailedError
from autocollimator.config.models import MeasuredValues, MeasurementOutput, Recommendations
from autocollimator.config.store import load_main, update_main_device_fields
from autocollimator.instrument import RotationZeroMode, TiltZeroMode, to_angular
from autocollimator.units import OutputUnits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.fixture
def config_dir(tmp_path) -> Path:
    dst = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dst)
    return dst


@pytest.fixture
def app(config_dir, tmp_path):
    with AutocollimatorApp(config_dir=config_dir, output_dir=tmp_path / "out") as a:
        yield a


def _output(app: AutocollimatorApp, x=1000.25, y=500.5, a0=1.43, a1=1.43, success=True) -> MeasurementOutput:
    mv = MeasuredValues(
        center_position_x=x if success else None,
        center_position_y=y if success else None,
        measured_angle_a0=a0 if success else None,
        measured_angle_a1=a1 if success else None,
        rss_ratio_r0=0.1, rss_ratio_r1=0.1, rss_ratio_r2=0.1, rss_ratio_r3=0.1,
        brightness_ratio_b0=0, brightness_ratio_b1=0, brightness_ratio_b2=0, brightness_ratio_b3=0,
        linewidth_w0=0, linewidth_w1=0, linewidth_w2=0, linewidth_w3=0,
        overlay_hash="0" * 32,
        success=success,
        message="Center found" if success else "Center could not be determined: quality check failed",
        image_width=1920,
        image_height=1080,
    )
    return MeasurementOutput(
        hardware=app.get_measurement_hardware(),
        measurement_parameters=app.get_measurement_parameters(),
        quality_limits=app.get_quality_limits(),
        measured_values=mv,
        recommendations=Recommendations(0, 0, 0, 0, 0, 0, 0),
        measurement_name="m",
        image_name="i",
        image_hash="0" * 32,
        output_units="Pixels",
        pixel_to_unit_scale_factor=1,
    )


@pytest.mark.unit
def test_defaults_are_image_center_and_image_axes(app) -> None:
    model = app.get_instrument_model()
    assert model.tilt_zero.mode is TiltZeroMode.IMAGE_CENTER
    assert model.rotation_zero.mode is RotationZeroMode.IMAGE_AXES


@pytest.mark.unit
def test_set_tilt_zero_persists_and_keeps_rotation_zero(app, config_dir) -> None:
    ref = app.set_tilt_zero_from_measurement(_output(app))
    assert (ref.mode, ref.x_px, ref.y_px) == (TiltZeroMode.MEASURED_ZERO, 1000.25, 500.5)
    reloaded = load_main(config_dir)
    assert (reloaded.tilt_zero_mode, reloaded.tilt_zero_x_px, reloaded.tilt_zero_y_px) == (
        "measured_zero", 1000.25, 500.5,
    )
    assert reloaded.rotation_zero_mode == "image_axes" and reloaded.rotation_zero_deg is None
    assert app.get_instrument_model().tilt_zero == ref


@pytest.mark.unit
def test_set_rotation_zero_persists_and_keeps_tilt_zero(app, config_dir) -> None:
    ref = app.set_rotation_zero_from_measurement(_output(app, a0=1.43, a1=1.43))
    assert ref.mode is RotationZeroMode.MEASURED_ZERO
    assert ref.rotation_deg == pytest.approx(1.43, abs=1e-12)
    reloaded = load_main(config_dir)
    assert reloaded.rotation_zero_mode == "measured_zero"
    assert reloaded.rotation_zero_deg == pytest.approx(1.43, abs=1e-12)
    assert reloaded.tilt_zero_mode == "image_center" and reloaded.tilt_zero_x_px is None

    # Zeroed at 1.43 deg, now 1.57 deg -> reads 0.14 deg.
    now = _output(app, a0=1.57, a1=1.57).measured_values
    assert to_angular(now, app.get_instrument_model(), OutputUnits()).rotation == pytest.approx(0.14, abs=1e-12)


@pytest.mark.unit
def test_rotation_zero_from_wrapped_angles(app) -> None:
    ref = app.set_rotation_zero_from_measurement(_output(app, a0=176.2, a1=176.3))
    assert ref.rotation_deg == pytest.approx(-3.75, abs=1e-9)


@pytest.mark.unit
def test_both_zeros_set_independently(app, config_dir) -> None:
    app.set_tilt_zero_from_measurement(_output(app, x=900.0, y=400.0))
    app.set_rotation_zero_from_measurement(_output(app, a0=2.0, a1=2.0))
    app.set_tilt_zero_from_measurement(_output(app, x=901.0, y=401.0))
    reloaded = load_main(config_dir)
    assert (reloaded.tilt_zero_x_px, reloaded.tilt_zero_y_px) == (901.0, 401.0)
    assert reloaded.rotation_zero_deg == pytest.approx(2.0, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["set_tilt_zero_from_measurement", "set_rotation_zero_from_measurement"])
def test_failed_measurement_raises_and_changes_nothing(app, config_dir, method) -> None:
    before = (config_dir / "main.toml").read_bytes()
    with pytest.raises(MeasurementFailedError, match="failed measurement"):
        getattr(app, method)(_output(app, success=False))
    assert (config_dir / "main.toml").read_bytes() == before


@pytest.mark.unit
def test_encryption_key_reference_and_comments_kept(app, config_dir) -> None:
    app.set_tilt_zero_from_measurement(_output(app))
    text = (config_dir / "main.toml").read_text()
    assert 'encryption_key = "${DEVICE_ENCRYPTION_KEY}"' in text
    assert "test-dummy-key" not in text
    assert "# rad | deg | mrad | urad | arcmin | arcsec" in text


@pytest.mark.unit
def test_calibrated_point_loads_from_toml(config_dir) -> None:
    update_main_device_fields(
        config_dir, {"tilt_zero_mode": "calibrated_point", "tilt_zero_x_px": 960.0, "tilt_zero_y_px": 540.0}
    )
    device = load_main(config_dir)
    assert device.tilt_zero.mode is TiltZeroMode.CALIBRATED_POINT
    assert (device.tilt_zero.x_px, device.tilt_zero.y_px) == (960.0, 540.0)


@pytest.mark.unit
def test_invalid_zero_config_rejected_before_writing(config_dir) -> None:
    before = (config_dir / "main.toml").read_bytes()
    with pytest.raises(ValueError, match="tilt_zero_x_px"):
        update_main_device_fields(config_dir, {"tilt_zero_mode": "calibrated_point"})
    with pytest.raises(ValueError, match="rotation_zero_mode"):
        update_main_device_fields(config_dir, {"rotation_zero_mode": "fixed"})
    assert (config_dir / "main.toml").read_bytes() == before
