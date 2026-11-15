"""Instrument layer (R3): image measurement -> tilt / rotation / orthogonality.

All expected values are computed by hand from tilt = 1/2 * atan(d / f), not taken from the code.
"""
import dataclasses
import math

import pytest

from autocollimator.config.models import MeasuredValues
from autocollimator.instrument import (
    AngularMeasurement,
    ImageSizeMismatchError,
    InstrumentModel,
    ReticleType,
    RotationZeroMode,
    RotationZeroReference,
    TiltZeroMode,
    TiltZeroReference,
    check_image_size,
    to_angular,
)
from autocollimator.units import AngleUnit, OutputUnits

W, H = 1920, 1080
CX, CY = (W - 1) / 2, (H - 1) / 2  # 959.5, 539.5
RAD = OutputUnits(AngleUnit.RAD, AngleUnit.RAD)
DEG = OutputUnits(AngleUnit.DEG, AngleUnit.DEG)


def _model(**overrides) -> InstrumentModel:
    d = dict(
        reticle_type=ReticleType.CROSSHAIR,
        collimator_efl_mm=200.0,
        microscope_efl_mm=45.0,
        capture_mode_id="test",
        pixel_pitch_x_um=3.45,
        pixel_pitch_y_um=3.45,
        expected_image_width=W,
        expected_image_height=H,
    )
    d.update(overrides)
    return InstrumentModel(**d)


def _mv(x=CX, y=CY, a0=0.0, a1=0.0, success=True, width=W, height=H) -> MeasuredValues:
    return MeasuredValues(
        center_position_x=x if success else None,
        center_position_y=y if success else None,
        measured_angle_a0=a0 if success else None,
        measured_angle_a1=a1 if success else None,
        rss_ratio_r0=0.1, rss_ratio_r1=0.1, rss_ratio_r2=0.1, rss_ratio_r3=0.1,
        brightness_ratio_b0=0, brightness_ratio_b1=0, brightness_ratio_b2=0, brightness_ratio_b3=0,
        linewidth_w0=0, linewidth_w1=0, linewidth_w2=0, linewidth_w3=0,
        overlay_hash="0" * 32,
        success=success,
        message="Center found" if success else "Center could not be determined",
        image_width=width,
        image_height=height,
    )


# ---- tilt magnitudes --------------------------------------------------------------------------

@pytest.mark.unit
def test_one_pixel_200mm_native_pitch() -> None:
    r = to_angular(_mv(x=CX + 1), _model(), RAD)
    assert r.tilt_x == pytest.approx(8.625e-6, rel=1e-6)
    assert to_angular(_mv(x=CX + 1), _model(), DEG).tilt_x == pytest.approx(4.9418e-4, rel=1e-4)
    arcsec = OutputUnits(AngleUnit.ARCSEC, AngleUnit.DEG)
    assert to_angular(_mv(x=CX + 1), _model(), arcsec).tilt_x == pytest.approx(1.7790, rel=1e-4)
    assert r.spot_dx_mm == pytest.approx(3.45e-3, rel=1e-12)


@pytest.mark.unit
def test_one_pixel_400mm_is_half() -> None:
    r = to_angular(_mv(x=CX + 1), _model(collimator_efl_mm=400.0), RAD)
    assert r.tilt_x == pytest.approx(8.625e-6 / 2, rel=1e-6)


@pytest.mark.unit
def test_one_pixel_hd_1080_pitch() -> None:
    model = _model(pixel_pitch_x_um=2.61625, pixel_pitch_y_um=2.61625)
    assert to_angular(_mv(x=CX + 1), model, RAD).tilt_x == pytest.approx(6.5406e-6, rel=1e-4)
    arcsec = OutputUnits(AngleUnit.ARCSEC, AngleUnit.DEG)
    assert to_angular(_mv(x=CX + 1), model, arcsec).tilt_x == pytest.approx(1.3491, rel=1e-4)


@pytest.mark.unit
def test_large_displacement_uses_exact_atan() -> None:
    # 500 px * 3.45 um = 1.725 mm; 1.725/200 = 0.008625.
    # atan(x) = x - x^3/3 + x^5/5 - ... with x = 0.008625:
    #   0.008625 - 2.13873046875e-7 + 9.546e-12 = 0.0086247861365 -> half = 0.00431239306825
    r = to_angular(_mv(x=CX + 500), _model(), RAD)
    assert r.tilt_x == pytest.approx(0.00431239306825, abs=1e-13)
    small_angle = 1.725 / (2 * 200)  # 0.0043125
    assert abs(r.tilt_x - small_angle) > 1e-7


# ---- signs and zero point --------------------------------------------------------------------

@pytest.mark.unit
def test_signs() -> None:
    right = to_angular(_mv(x=CX + 10), _model(), RAD)
    assert right.tilt_x > 0 and right.tilt_y == 0
    left = to_angular(_mv(x=CX - 10), _model(), RAD)
    assert left.tilt_x < 0
    up = to_angular(_mv(y=CY - 10), _model(), RAD)  # smaller image y = above center
    assert up.tilt_y > 0 and up.spot_dy_mm > 0
    down = to_angular(_mv(y=CY + 10), _model(), RAD)
    assert down.tilt_y < 0


@pytest.mark.unit
def test_spot_at_image_center_is_zero() -> None:
    r = to_angular(_mv(x=959.5, y=539.5), _model(), RAD)
    assert (r.tilt_x, r.tilt_y) == (0.0, 0.0)
    assert (r.tilt_zero_x_px, r.tilt_zero_y_px) == (959.5, 539.5)
    assert r.tilt_zero_mode is TiltZeroMode.IMAGE_CENTER


@pytest.mark.unit
def test_old_record_without_image_size_uses_model_size() -> None:
    r = to_angular(_mv(width=None, height=None), _model(), RAD)
    assert (r.tilt_zero_x_px, r.tilt_zero_y_px) == (959.5, 539.5)


@pytest.mark.unit
@pytest.mark.parametrize("mode", [TiltZeroMode.CALIBRATED_POINT, TiltZeroMode.MEASURED_ZERO])
def test_tilt_zero_point_modes(mode) -> None:
    model = _model(tilt_zero=TiltZeroReference(mode, 1000.0, 500.0))
    r = to_angular(_mv(x=1001.0, y=499.0), model, RAD)
    assert r.tilt_zero_mode is mode
    assert (r.tilt_zero_x_px, r.tilt_zero_y_px) == (1000.0, 500.0)
    assert r.tilt_x == pytest.approx(8.625e-6, rel=1e-6)
    assert r.tilt_y == pytest.approx(8.625e-6, rel=1e-6)
    at_zero = to_angular(_mv(x=1000.0, y=500.0), model, RAD)
    assert (at_zero.tilt_x, at_zero.tilt_y) == (0.0, 0.0)


@pytest.mark.unit
def test_tilt_zero_point_mode_requires_values() -> None:
    with pytest.raises(ValueError, match="tilt_zero_x_px"):
        TiltZeroReference(TiltZeroMode.CALIBRATED_POINT, None, 500.0)
    with pytest.raises(ValueError, match="tilt_zero_mode"):
        TiltZeroReference("center")


# ---- rotation and orthogonality ---------------------------------------------------------------

@pytest.mark.unit
def test_rotation_wrap_negative_angles() -> None:
    r = to_angular(_mv(a0=176.2, a1=176.3), _model(), DEG)
    assert r.rotation == pytest.approx(-3.75, abs=1e-9)
    already_wrapped = to_angular(_mv(a0=-3.8, a1=-3.7), _model(), DEG)
    assert already_wrapped.rotation == pytest.approx(-3.75, abs=1e-9)


@pytest.mark.unit
def test_rotation_across_45_degrees() -> None:
    # Lines at 44.9 deg and 45.1 deg (= -44.9 deg): mean 45 deg, not 0.
    r = to_angular(_mv(a0=44.9, a1=-44.9), _model(), DEG)
    assert r.rotation == pytest.approx(45.0, abs=1e-9)
    assert r.orthogonality == pytest.approx(-0.2, abs=1e-9)
    r2 = to_angular(_mv(a0=-44.9, a1=44.9), _model(), DEG)
    assert r2.rotation == pytest.approx(45.0, abs=1e-9)
    assert r2.orthogonality == pytest.approx(0.2, abs=1e-9)


@pytest.mark.unit
def test_rotation_sign_and_orthogonality() -> None:
    r = to_angular(_mv(a0=2.0, a1=1.0), _model(), DEG)
    assert r.rotation == pytest.approx(1.5, abs=1e-12)
    # Vertical line at 92 deg, horizontal at 1 deg: angle between = 91 deg -> +1 deg.
    assert r.orthogonality == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_rotation_zero_measured() -> None:
    model = _model(rotation_zero=RotationZeroReference(RotationZeroMode.MEASURED_ZERO, 1.43))
    r = to_angular(_mv(a0=1.57, a1=1.57), model, DEG)
    assert r.rotation == pytest.approx(0.14, abs=1e-12)
    assert r.rotation_zero_mode is RotationZeroMode.MEASURED_ZERO
    assert r.rotation_zero == pytest.approx(1.43, abs=1e-12)


@pytest.mark.unit
def test_rotation_zero_wraps_result() -> None:
    model = _model(rotation_zero=RotationZeroReference(RotationZeroMode.MEASURED_ZERO, 44.0))
    assert to_angular(_mv(a0=-44.0, a1=-44.0), model, DEG).rotation == pytest.approx(2.0, abs=1e-9)


@pytest.mark.unit
def test_rotation_image_axes_default() -> None:
    r = to_angular(_mv(a0=1.57, a1=1.57), _model(), DEG)
    assert r.rotation == pytest.approx(1.57, abs=1e-12)
    assert r.rotation_zero_mode is RotationZeroMode.IMAGE_AXES and r.rotation_zero == 0.0


@pytest.mark.unit
def test_orthogonality_unaffected_by_rotation_zero() -> None:
    mv = _mv(a0=1.6, a1=1.5)
    plain = to_angular(mv, _model(), DEG)
    zeroed = to_angular(mv, _model(rotation_zero=RotationZeroReference("measured_zero", 1.43)), DEG)
    assert plain.orthogonality == pytest.approx(0.1, abs=1e-12)
    assert zeroed.orthogonality == plain.orthogonality
    assert zeroed.rotation != plain.rotation


@pytest.mark.unit
def test_zeros_set_independently() -> None:
    mv = _mv(x=1001.0, y=500.0, a0=1.57, a1=1.57)
    tilt_only = _model(tilt_zero=TiltZeroReference("measured_zero", 1000.0, 500.0))
    rot_only = _model(rotation_zero=RotationZeroReference("measured_zero", 1.43))
    t = to_angular(mv, tilt_only, DEG)
    r = to_angular(mv, rot_only, DEG)
    assert t.rotation == pytest.approx(1.57, abs=1e-12)
    assert r.rotation == pytest.approx(0.14, abs=1e-12)
    assert t.tilt_zero_x_px == 1000.0 and r.tilt_zero_x_px == CX


# ---- units, failure, mismatch ------------------------------------------------------------------

@pytest.mark.unit
def test_units_chosen_independently() -> None:
    units = OutputUnits(tilt_unit=AngleUnit.MRAD, rotation_unit=AngleUnit.DEG)
    r = to_angular(_mv(x=CX + 1, a0=176.2, a1=176.3), _model(), units)
    assert r.tilt_x == pytest.approx(8.625e-3, rel=1e-6)
    assert r.rotation == pytest.approx(-3.75, abs=1e-9)
    assert (r.tilt_unit, r.rotation_unit) == (AngleUnit.MRAD, AngleUnit.DEG)


@pytest.mark.unit
def test_failed_image_measurement() -> None:
    r = to_angular(_mv(success=False), _model(), DEG)
    assert r.success is False
    assert (r.tilt_x, r.tilt_y, r.rotation, r.orthogonality) == (None, None, None, None)
    assert (r.spot_dx_mm, r.spot_dy_mm) == (None, None)


@pytest.mark.unit
def test_success_record_requires_values() -> None:
    good = to_angular(_mv(), _model(), DEG)
    with pytest.raises(ValueError, match="tilt_x"):
        dataclasses.replace(good, tilt_x=None)


@pytest.mark.unit
def test_check_image_size_policy(caplog) -> None:
    check_image_size(_model(), W, H)  # match: no error, no warning
    with pytest.raises(ImageSizeMismatchError, match="1456x1088"):
        check_image_size(_model(expected_image_width=1456, expected_image_height=1088), W, H)
    with caplog.at_level("WARNING", logger="autocollimator.instrument"):
        check_image_size(_model(expected_image_width=1456, expected_image_height=1088,
                                image_size_mismatch="warn"), W, H)
    assert "does not match" in caplog.text


@pytest.mark.unit
def test_model_validation() -> None:
    with pytest.raises(ValueError, match="reticle_type"):
        _model(reticle_type="ring")
    with pytest.raises(ValueError, match="collimator_efl_mm"):
        _model(collimator_efl_mm=0.0)
    with pytest.raises(ValueError, match="image_size_mismatch"):
        _model(image_size_mismatch="ignore")
    assert _model(reticle_type="crosshair").reticle_type is ReticleType.CROSSHAIR


@pytest.mark.unit
def test_angular_measurement_from_dict_round_trip() -> None:
    r = to_angular(_mv(x=CX + 3, a0=1.0, a1=2.0), _model(), OutputUnits(AngleUnit.ARCSEC, AngleUnit.DEG))
    d = {k: (v.value if hasattr(v, "value") else v) for k, v in dataclasses.asdict(r).items()}
    assert AngularMeasurement.from_dict(d) == r
    m = _model(tilt_zero=TiltZeroReference("calibrated_point", 1.0, 2.0))
    md = dataclasses.asdict(m, dict_factory=lambda kv: {k: (v.value if hasattr(v, "value") else v) for k, v in kv})
    assert InstrumentModel.from_dict(md) == m
    assert math.isfinite(r.tilt_x)
