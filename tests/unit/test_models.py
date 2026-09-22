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
