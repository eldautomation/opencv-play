"""R18: gray_mode channel reduction (and, below, per-ROI saturation)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from autocollimator.target_utils import ensure_gray_f64, find_cross_center_detailed

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
