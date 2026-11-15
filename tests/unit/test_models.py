import dataclasses

import pytest

from autocollimator.config.models import MeasuredValues

OPTIONAL_FIELDS = ("center_position_x", "center_position_y", "measured_angle_a0", "measured_angle_a1")


def _values(**overrides) -> dict:
    d = {
        "center_position_x": 883.72,
        "center_position_y": 566.83,
        "measured_angle_a0": 2.99,
        "measured_angle_a1": 3.16,
        "rss_ratio_r0": 0.076,
        "rss_ratio_r1": 0.156,
        "rss_ratio_r2": 0.156,
        "rss_ratio_r3": 0.28,
        "brightness_ratio_b0": 0.0,
        "brightness_ratio_b1": 0.0,
        "brightness_ratio_b2": 0.0,
        "brightness_ratio_b3": 0.0,
        "linewidth_w0": 0.0,
        "linewidth_w1": 0.0,
        "linewidth_w2": 0.0,
        "linewidth_w3": 0.0,
        "overlay_hash": "0" * 32,
        "success": True,
        "message": "Center found",
    }
    d.update(overrides)
    return d


def _failed(**overrides) -> dict:
    return _values(
        **{name: None for name in OPTIONAL_FIELDS},
        success=False,
        message="Center could not be determined",
        **overrides,
    )


@pytest.mark.unit
def test_measured_values_failure_allows_none():
    mv = MeasuredValues(**_failed())
    assert mv.success is False
    assert all(getattr(mv, name) is None for name in OPTIONAL_FIELDS)


@pytest.mark.unit
@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_measured_values_success_rejects_none(field):
    with pytest.raises(ValueError, match=f"{field} cannot be None when success is True"):
        MeasuredValues(**_values(**{field: None}))


@pytest.mark.unit
def test_measured_values_from_dict_failure_with_nulls():
    mv = MeasuredValues.from_dict(_failed())
    assert mv == MeasuredValues(**_failed())


@pytest.mark.unit
@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_measured_values_from_dict_success_rejects_null(field):
    with pytest.raises(ValueError, match=f"{field} cannot be None when success is True"):
        MeasuredValues.from_dict(_values(**{field: None}))


@pytest.mark.unit
def test_measured_values_from_dict_round_trips_success():
    mv = MeasuredValues.from_dict(_values())
    assert MeasuredValues.from_dict(dataclasses.asdict(mv)) == mv


ROI_METHOD_FIELDS = ("roi_method_m0", "roi_method_m1", "roi_method_m2", "roi_method_m3")


@pytest.mark.unit
def test_measured_values_roi_methods_default_to_average():
    mv = MeasuredValues(**_values())
    assert all(getattr(mv, name) == "average" for name in ROI_METHOD_FIELDS)


@pytest.mark.unit
def test_measured_values_from_dict_without_roi_methods_defaults_to_average():
    # Measurement files written before the deskew mode have no roi_method_* keys.
    mv = MeasuredValues.from_dict(_values())
    assert all(getattr(mv, name) == "average" for name in ROI_METHOD_FIELDS)


@pytest.mark.unit
def test_measured_values_roi_methods_round_trip():
    methods = dict(zip(ROI_METHOD_FIELDS, ("deskew", "fallback", "deskew", "deskew")))
    mv = MeasuredValues(**_values(**methods))
    assert MeasuredValues.from_dict(dataclasses.asdict(mv)) == mv


@pytest.mark.unit
@pytest.mark.parametrize("field", ROI_METHOD_FIELDS)
@pytest.mark.parametrize("value", ["", "Deskew", None, 1])
def test_measured_values_invalid_roi_method_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} must be one of"):
        MeasuredValues(**_values(**{field: value}))
    with pytest.raises(ValueError, match=f"{field} must be one of"):
        MeasuredValues.from_dict(_values(**{field: value}))



@pytest.mark.unit
@pytest.mark.parametrize("value", ["fallback:tilt", "fallback:bins", "fallback:edge", "fallback:shift"])
def test_measured_values_accepts_fallback_reasons(value):
    mv = MeasuredValues.from_dict(_values(roi_method_m2=value))
    assert mv.roi_method_m2 == value


@pytest.mark.unit
def test_measured_values_legacy_plain_fallback_still_loads():
    # Measurement files written before fallback reasons were recorded contain "fallback".
    mv = MeasuredValues.from_dict(_values(roi_method_m0="fallback"))
    assert mv.roi_method_m0 == "fallback"


@pytest.mark.unit
def test_measured_values_not_run_round_trip():
    failed = dict(_values(success=False, message="quality check failed"), **dict.fromkeys(OPTIONAL_FIELDS))
    mv = MeasuredValues(**failed, **dict.fromkeys(ROI_METHOD_FIELDS, "not_run"))
    assert MeasuredValues.from_dict(dataclasses.asdict(mv)) == mv


@pytest.mark.unit
@pytest.mark.parametrize("legacy", [{}, dict.fromkeys(ROI_METHOD_FIELDS, "average")])
def test_failed_measurement_with_legacy_roi_methods_still_loads(legacy):
    # Failed measurements were recorded as "average" (or without roi_method_*) before "not_run".
    failed = dict(_values(success=False, message="quality check failed"), **dict.fromkeys(OPTIONAL_FIELDS))
    mv = MeasuredValues.from_dict(dict(failed, **legacy))
    assert mv.success is False
    assert all(getattr(mv, name) == "average" for name in ROI_METHOD_FIELDS)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["fallback:", "fallback:other"])
def test_measured_values_unknown_fallback_reason_raises(value):
    with pytest.raises(ValueError, match="roi_method_m1 must be one of"):
        MeasuredValues(**_values(roi_method_m1=value))


SATURATION_FIELDS = ("saturation_s0", "saturation_s1", "saturation_s2", "saturation_s3")


@pytest.mark.unit
def test_measured_values_from_dict_without_saturation_loads_as_none():
    mv = MeasuredValues.from_dict(_values())
    assert all(getattr(mv, name) is None for name in SATURATION_FIELDS)


@pytest.mark.unit
def test_measured_values_saturation_round_trip():
    sat = dict(zip(SATURATION_FIELDS, (0.0, 0.0153, 1.0, 1)))
    mv = MeasuredValues.from_dict(_values(**sat))
    assert MeasuredValues.from_dict(dataclasses.asdict(mv)) == mv
    assert (mv.saturation_s0, mv.saturation_s1, mv.saturation_s2, mv.saturation_s3) == (0.0, 0.0153, 1.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("field", SATURATION_FIELDS)
@pytest.mark.parametrize("value", [1.5, -0.1, True, "0.1"])
def test_measured_values_invalid_saturation_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} must be None or a number in 0..1"):
        MeasuredValues.from_dict(_values(**{field: value}))
    with pytest.raises(ValueError, match=f"{field} must be None or a number in 0..1"):
        MeasuredValues(**_values(**{field: value}))
