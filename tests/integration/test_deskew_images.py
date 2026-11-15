"""C2: the deskew pass never turns a passing measurement into a failure on the test images."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center_detailed

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
        "roi_size": (int(mps.roi_size_x), int(mps.roi_size_y)),
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
def test_blob_13_deskew_subpixel_matches_average_subpixel(default_params):
    # Regression (C2-fix): the right ROI's pass-2 minimum sat on the +/-1.5 px search edge,
    # moving y by ~1 px. The edge guard must fall back to pass 1 for that ROI.
    img = cv2.imread(str(IMAGE_DIR / "blob-13.jpg"))
    if img is None:
        pytest.fail("Cannot read image: blob-13.jpg")
    average = find_cross_center_detailed(img, subpixel=True, **default_params)
    deskew = find_cross_center_detailed(img, subpixel=True, profile_mode="deskew", **default_params)
    assert average.center is not None and deskew.center is not None
    dx = deskew.center[0] - average.center[0]
    dy = deskew.center[1] - average.center[1]
    assert abs(dx) <= 0.1 and abs(dy) <= 0.1, f"deskew - average = ({dx:+.3f}, {dy:+.3f}) px"
    assert deskew.roi_methods[2] == "fallback:edge"  # right ROI
