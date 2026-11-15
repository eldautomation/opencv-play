"""C2: the deskew pass never turns a passing measurement into a failure on the test images."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import _deskew_position, find_cross_center_detailed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"
IMAGES = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))


@pytest.fixture(scope="module")
def default_params(tmp_path_factory):
    """Crop/ROI/q_limit from the default config, as the workflow uses them."""
    with AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs",
        output_dir=tmp_path_factory.mktemp("deskew"),
        input_dir=IMAGE_DIR,
    ) as app:
        mps = app.get_measurement_parameters()
        qls = app.get_quality_limits()
    return {
        "crop_center": (int(mps.crop_center_x), int(mps.crop_center_y)),
        "crop_size": (int(mps.crop_size_x), int(mps.crop_size_y)),
        "roi_size": (mps.roi_size_x, int(mps.roi_size_y)),
        "q_limit": qls.rss_ratio,
    }


@pytest.mark.integration
@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("name", IMAGES)
def test_deskew_succeeds_where_average_succeeds(name, subpixel, default_params):
    img = cv2.imread(str(IMAGE_DIR / name))
    if img is None:
        pytest.fail(f"Cannot read image: {name}")

    average = find_cross_center_detailed(img, subpixel=subpixel, **default_params)
    deskew = find_cross_center_detailed(img, subpixel=subpixel, profile_mode="deskew", **default_params)

    assert deskew.q_ratio_list == average.q_ratio_list
    assert (deskew.center is None) == (average.center is None), (
        f"{name}: average success={average.center is not None}, deskew success={deskew.center is not None}"
    )
    if deskew.center is not None:
        assert deskew.angles is not None
        assert set(deskew.roi_methods) <= {"deskew", "fallback:tilt", "fallback:bins", "fallback:edge", "fallback:shift"}


@pytest.mark.integration
@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("name", ["blob-13.jpg", "blob-22.jpg"])
def test_explicit_default_refine_search_is_identical(name, subpixel, default_params):
    img = cv2.imread(str(IMAGE_DIR / name))
    if img is None:
        pytest.fail(f"Cannot read image: {name}")
    default = find_cross_center_detailed(img, subpixel=subpixel, profile_mode="deskew", **default_params)
    explicit = find_cross_center_detailed(
        img, subpixel=subpixel, profile_mode="deskew", refine_search=1.5, **default_params
    )
    assert explicit.center == default.center
    assert explicit.angles == default.angles
    assert explicit.roi_methods == default.roi_methods
    assert [t.pos2 for t in explicit.rois] == [t.pos2 for t in default.rois]


@pytest.mark.integration
@pytest.mark.parametrize("fine_window_mode", ["fixed", "adaptive"])
def test_blob_13_deskew_subpixel_matches_average_subpixel(fine_window_mode, default_params):
    # Regression (C2-fix): the right ROI's pass-2 minimum sat on the +/-1.5 px search edge,
    # moving y by ~1 px. The edge guard must fall back to pass 1 for that ROI.
    img = cv2.imread(str(IMAGE_DIR / "blob-13.jpg"))
    if img is None:
        pytest.fail("Cannot read image: blob-13.jpg")
    average = find_cross_center_detailed(img, subpixel=True, **default_params)
    deskew = find_cross_center_detailed(
        img, subpixel=True, profile_mode="deskew", fine_window_mode=fine_window_mode, **default_params
    )
    assert average.center is not None and deskew.center is not None
    dx = deskew.center[0] - average.center[0]
    dy = deskew.center[1] - average.center[1]
    assert abs(dx) <= 0.1 and abs(dy) <= 0.1, f"deskew - average = ({dx:+.3f}, {dy:+.3f}) px"
    right = deskew.rois[2]
    # Both modes fall back for the right ROI. Its line FWHM is ~17.6 px (diagnose_refine.py:
    # 17.3), so the adaptive window is max(40, 1.5 x 17.6) = 40, the same as fixed. (395a071
    # passed here in adaptive mode only because the estimator measured the ~61 px pedestal
    # under the line; the ROI needs a window >= 50 px to have a real pass-2 minimum.)
    assert deskew.roi_methods[2] == "fallback:edge"
    assert deskew.fine_windows[2] == 40
    if fine_window_mode == "adaptive":
        assert right.line_width == pytest.approx(17.6, abs=1.0)
    else:
        assert right.line_width is None
    # The edge guard itself, on the 40 px window that triggers it.
    method, pos2 = _deskew_position(
        right.roi, right.pos1, right.slope, oversample_step=0.1, fine_window=40,
        subpixel=True, subpixel_neighbors=2, name="right",
    )
    assert method == "fallback:edge"
    assert abs(pos2 - right.pos1) == pytest.approx(1.5, abs=0.01)


@pytest.mark.integration
def test_blob_22_fixed_window_matches_pre_adaptive_result(default_params):
    # R17: the fixed window (default) must reproduce the deskew result from before the
    # adaptive window was added.
    img = cv2.imread(str(IMAGE_DIR / "blob-22.jpg"))
    if img is None:
        pytest.fail("Cannot read image: blob-22.jpg")
    deskew = find_cross_center_detailed(img, subpixel=True, profile_mode="deskew", **default_params)
    # Re-measured unrounded when line_intersection stopped rounding (Phase E; was (883.73, 566.49)).
    assert deskew.center == pytest.approx((883.7314602524013, 566.4864480670376), abs=0.01)
    assert deskew.fine_windows == (40.0,) * 4
