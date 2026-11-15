"""C2: pass 2 of the deskew profile mode on a single ROI, and option validation."""
import numpy as np
import pytest

from autocollimator.target_utils import (
    _deskew_position,
    _deskew_profile,
    _sdrm_core,
    find_cross_center_detailed,
    sdrm_ssd_curve,
)

ROW_REF_DEPTH = 20  # default roi_size_y
TAN_3DEG = float(np.tan(np.radians(3.0)))


def _tilted_line_roi(center: float, slope: float, *, depth: int = ROW_REF_DEPTH, width: int = 200,
                     fwhm: float = 3.0) -> np.ndarray:
    """ROI with a line along the rows: col = center + slope * (row - depth / 2), 5x supersampled."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sub = (np.arange(5) + 0.5) / 5 - 0.5
    rows = (np.arange(depth)[:, None] + sub[None, :]).ravel()[:, None]
    cols = (np.arange(width)[:, None] + sub[None, :]).ravel()[None, :]
    d = cols - center - slope * (rows - depth / 2.0)
    g = np.exp(-0.5 * (d / sigma) ** 2).reshape(depth, 5, width, 5).mean(axis=(1, 3))
    return 30.0 + 180.0 * g


def _deskew(roi, pos1, slope, **overrides):
    kwargs = dict(oversample_step=0.1, fine_window=40, subpixel=True, subpixel_neighbors=2, name="test")
    kwargs.update(overrides)
    return _deskew_position(roi, pos1, slope, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("center", [100.0, 100.23, 100.5, 100.77])
def test_deskew_position_recovers_center(center):
    roi = _tilted_line_roi(center, TAN_3DEG)
    pos = _deskew(roi, float(round(center)), TAN_3DEG)
    assert pos is not None
    assert abs(pos - center) < 0.02


@pytest.mark.unit
def test_deskew_position_without_subpixel_is_on_the_fine_grid():
    roi = _tilted_line_roi(100.37, TAN_3DEG)
    pos = _deskew(roi, 100.0, TAN_3DEG, subpixel=False)
    assert pos is not None
    assert abs(pos - 100.37) <= 0.05 + 1e-9
    steps = (pos - 100.0) / 0.1
    assert steps == pytest.approx(round(steps), abs=1e-9)


@pytest.mark.unit
def test_deskew_position_falls_back_when_line_shift_below_one_pixel():
    slope = 0.9 / ROW_REF_DEPTH  # line shifts 0.9 px over the ROI depth
    assert _deskew(_tilted_line_roi(100.0, slope), 100.0, slope) is None


@pytest.mark.unit
@pytest.mark.parametrize("slope", [float("nan"), float("inf")])
def test_deskew_position_falls_back_on_undefined_slope(slope):
    assert _deskew(_tilted_line_roi(100.0, TAN_3DEG), 100.0, slope) is None


@pytest.mark.unit
def test_deskew_position_falls_back_when_too_many_bins_empty():
    # A 60 px wide ROI cannot fill a +/-40 px window: about a quarter of the bins stay empty.
    roi = _tilted_line_roi(30.0, TAN_3DEG, width=60)
    assert _deskew(roi, 30.0, TAN_3DEG) is None
    assert _deskew(roi, 30.0, TAN_3DEG, fine_window=20) is not None


@pytest.mark.unit
def test_deskew_position_interpolates_sparse_bins():
    # With 20 rows and 0.05 px bins, slope 0.06 leaves 10 % of the bins empty (filled by
    # interpolation); slope 0.1 leaves 50 % empty, which is above the fallback limit.
    roi = _tilted_line_roi(100.3, 0.06)
    pos = _deskew(roi, 100.0, 0.06, oversample_step=0.05)
    assert pos is not None
    assert abs(pos - 100.3) < 0.02
    assert _deskew(_tilted_line_roi(100.3, 0.1), 100.0, 0.1, oversample_step=0.05) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"profile_mode": "tilted"}, "profile_mode must be one of"),
        ({"oversample_step": 0.01}, "oversample_step must be a number in 0.02..0.5"),
        ({"oversample_step": 0.6}, "oversample_step must be a number in 0.02..0.5"),
        ({"fine_window": 1}, "fine_window must be an int >= 2"),
        ({"fine_window": 40.0}, "fine_window must be an int >= 2"),
        ({"refine_search": 0.4}, "refine_search must be a number in 0.5..3.0"),
        ({"refine_search": 3.5}, "refine_search must be a number in 0.5..3.0"),
        ({"refine_search": True}, "refine_search must be a number in 0.5..3.0"),
        ({"refine_search": 2.5, "fine_window": 2}, "refine_search .* must be smaller than fine_window"),
    ],
)
def test_find_cross_center_rejects_invalid_profile_options(overrides, message):
    image = np.full((100, 100), 30.0)
    with pytest.raises(ValueError, match=message):
        find_cross_center_detailed(image, (50, 50), (60, 60), (60, 10), 0.5, **overrides)


@pytest.mark.unit
def test_deskew_position_default_search_is_1_5_px():
    roi = _tilted_line_roi(100.3, TAN_3DEG)
    assert _deskew(roi, 100.0, TAN_3DEG) == _deskew(roi, 100.0, TAN_3DEG, refine_search=1.5)


@pytest.mark.unit
@pytest.mark.parametrize("refine_search", [0.5, 1.5, 2.5])
def test_deskew_position_stays_within_refine_search(refine_search):
    # Pass-1 estimate 2 px off: the result is limited to +/-refine_search around it.
    roi = _tilted_line_roi(102.0, TAN_3DEG)
    pos = _deskew(roi, 100.0, TAN_3DEG, subpixel=False, refine_search=refine_search)
    assert pos is not None
    assert abs(pos - 100.0) <= refine_search + 1e-9
    if refine_search > 2.0:
        assert abs(pos - 102.0) <= 0.05 + 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("search", [5, 15, 30])
def test_sdrm_ssd_curve_matches_core_minimum(search):
    roi = _tilted_line_roi(100.37, TAN_3DEG)
    profile, _empty = _deskew_profile(roi, 100.0, TAN_3DEG, oversample_step=0.1, fine_window=40, name="t")
    ssd = sdrm_ssd_curve(profile, search)
    assert ssd.shape == (2 * search + 1,)
    k_core, r1 = _sdrm_core(profile, search)
    assert int(np.argmin(ssd)) - search == k_core
    rss = np.sqrt(ssd)
    assert r1 == pytest.approx(rss.min() / rss.mean(), rel=1e-12)


@pytest.mark.unit
def test_sdrm_ssd_curve_rejects_short_profile():
    with pytest.raises(ValueError, match="more than 2\\*search_size"):
        sdrm_ssd_curve(np.ones(10), 5)
