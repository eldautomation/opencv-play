"""R18: gray_mode channel reduction (and, below, per-ROI saturation)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from autocollimator.target_utils import (
    ROI_NAMES,
    SATURATION_LEVEL,
    ensure_gray_f64,
    find_cross_center_detailed,
    roi_saturation_fraction,
)

IMAGE_SHAPE = (1088, 1456)  # (h, w): configured sensor size
CENTER = (872.0, 544.0)  # default crop center
CROSS_KW = dict(crop_center=(872, 544), crop_size=(1018, 652), roi_size=(500, 20), q_limit=0.5)


def _cross(fwhm: float = 6.0, *, center=CENTER, background: float = 30.0, peak: float = 210.0) -> np.ndarray:
    """Untilted Gaussian crosshair as uint8 (H, W)."""
    h, w = IMAGE_SHAPE
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    gx = np.exp(-0.5 * ((xs - center[0]) / sigma) ** 2)[None, :]
    gy = np.exp(-0.5 * ((ys - center[1]) / sigma) ** 2)[:, None]
    img = background + (peak - background) * np.maximum(gx, gy)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def random_bgr() -> np.ndarray:
    return np.random.default_rng(18).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)


@pytest.mark.unit
def test_gray_mode_luma_is_cv2_conversion(random_bgr):
    expected = cv2.cvtColor(random_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    assert np.array_equal(ensure_gray_f64(random_bgr), expected)
    assert np.array_equal(ensure_gray_f64(random_bgr, "luma"), expected)


@pytest.mark.unit
def test_gray_mode_green_is_g_channel_exactly(random_bgr):
    gray = ensure_gray_f64(random_bgr, "green")
    assert gray.dtype == np.float64
    assert np.array_equal(gray, random_bgr[:, :, 1].astype(np.float64))


@pytest.mark.unit
def test_gray_mode_max_is_channel_max(random_bgr):
    assert np.array_equal(ensure_gray_f64(random_bgr, "max"), random_bgr.max(axis=2).astype(np.float64))


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["luma", "green", "max"])
def test_gray_mode_leaves_2d_input_unchanged(mode, random_bgr):
    gray = random_bgr[:, :, 0]
    assert np.array_equal(ensure_gray_f64(gray, mode), gray.astype(np.float64))


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["red", "Luma", ""])
def test_gray_mode_unknown_raises(mode, random_bgr):
    with pytest.raises(ValueError, match="gray_mode must be one of"):
        ensure_gray_f64(random_bgr, mode)
    with pytest.raises(ValueError, match="gray_mode must be one of"):
        find_cross_center_detailed(random_bgr, gray_mode=mode, **CROSS_KW)


@pytest.mark.unit
def test_find_cross_center_green_uses_only_g_channel():
    # G holds the cross at the crop center; B and R hold crosses elsewhere plus noise, so any
    # leak of B or R into the profile would move the center.
    green = _cross()
    blue = _cross(center=(CENTER[0] + 7.0, CENTER[1] - 5.0))
    noise = np.random.default_rng(19).integers(0, 60, size=IMAGE_SHAPE, dtype=np.uint8)
    red = np.maximum(_cross(fwhm=12.0, center=(CENTER[0] - 9.0, CENTER[1] + 4.0)), noise)
    bgr = np.dstack([blue, green, red])

    from_green = find_cross_center_detailed(bgr, gray_mode="green", **CROSS_KW)
    from_g_channel = find_cross_center_detailed(green, **CROSS_KW)
    assert from_green.center is not None
    assert from_green.center == from_g_channel.center
    assert from_green.angles == from_g_channel.angles
    assert from_green.q_ratio_list == from_g_channel.q_ratio_list
    assert find_cross_center_detailed(bgr, **CROSS_KW).center != from_green.center


def _roi_bounds(trace) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a RoiTrace in full-image coordinates (left/right ROIs are transposed)."""
    h, w = trace.roi.shape if trace.name in ("top", "bottom") else trace.roi.shape[::-1]
    return trace.x0, trace.y0, trace.x0 + w, trace.y0 + h


def _bgr_cross_with_hot_pixels(n_red: int, n_green: int, n_249: int):
    """BGR cross with pixels set in the top ROI; returns (image, top ROI area)."""
    bgr = np.dstack([_cross()] * 3)
    x0, y0, x1, y1 = _roi_bounds(find_cross_center_detailed(bgr, **CROSS_KW).rois[0])
    area = (x1 - x0) * (y1 - y0)
    flat = np.random.default_rng(20).permutation(area)[: n_red + n_green + n_249]
    rows, cols = np.unravel_index(flat, (y1 - y0, x1 - x0))
    rows, cols = rows + y0, cols + x0
    red, green, near = slice(0, n_red), slice(n_red, n_red + n_green), slice(n_red + n_green, None)
    bgr[rows[red], cols[red], 2] = 255  # only R clipped: counted (any channel)
    bgr[rows[green], cols[green], 1] = SATURATION_LEVEL  # exactly at the level: counted
    bgr[rows[near], cols[near]] = SATURATION_LEVEL - 1  # all channels 249: not counted
    return bgr, area


@pytest.mark.unit
def test_roi_saturation_fraction_counts_any_channel():
    roi = np.zeros((4, 5, 3), dtype=np.uint8)
    roi[0, 0, 0] = 250
    roi[1, 1, 2] = 255
    roi[2, 2] = 249
    assert roi_saturation_fraction(roi) == 2 / 20
    assert roi_saturation_fraction(roi[:, :, 0]) == 1 / 20
    with pytest.raises(ValueError):
        roi_saturation_fraction(np.zeros((0, 3), dtype=np.uint8))


@pytest.mark.unit
def test_saturation_fraction_per_roi_on_synthetic_image():
    bgr, area = _bgr_cross_with_hot_pixels(n_red=37, n_green=11, n_249=25)
    result = find_cross_center_detailed(bgr, **CROSS_KW)
    assert result.saturation == ((37 + 11) / area, 0.0, 0.0, 0.0)
    assert [t.saturation for t in result.rois] == list(result.saturation)


@pytest.mark.unit
def test_saturation_on_grayscale_input():
    gray = _cross()
    trace = find_cross_center_detailed(gray, **CROSS_KW).rois[2]  # right
    x0, y0, x1, y1 = _roi_bounds(trace)
    gray[y0 : y0 + 3, x0] = 255
    result = find_cross_center_detailed(gray, **CROSS_KW)
    assert result.saturation == (0.0, 0.0, 3 / ((x1 - x0) * (y1 - y0)), 0.0)


@pytest.mark.unit
def test_saturation_is_reported_on_failed_measurement():
    noise = np.random.default_rng(21).integers(0, 200, size=IMAGE_SHAPE + (3,), dtype=np.uint8)
    noise[:, :, 2] = 255
    result = find_cross_center_detailed(noise, **CROSS_KW)
    assert result.center is None
    assert result.saturation == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.unit
def test_saturation_warning_above_one_percent(caplog):
    bgr, area = _bgr_cross_with_hot_pixels(n_red=150, n_green=50, n_249=0)
    assert 200 / area > 0.01
    with caplog.at_level("WARNING", logger="autocollimator.target_utils"):
        result = find_cross_center_detailed(bgr, **CROSS_KW)
    assert result.center is not None  # saturation never fails the measurement
    warnings = [r.getMessage() for r in caplog.records if "saturated" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].startswith("ROI top:")


@pytest.mark.unit
def test_no_saturation_warning_below_one_percent(caplog):
    bgr, area = _bgr_cross_with_hot_pixels(n_red=10, n_green=0, n_249=500)
    with caplog.at_level("WARNING", logger="autocollimator.target_utils"):
        find_cross_center_detailed(bgr, **CROSS_KW)
    assert not [r for r in caplog.records if "saturated" in r.getMessage()]
