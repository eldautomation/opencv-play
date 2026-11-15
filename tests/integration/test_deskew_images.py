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
        assert set(deskew.roi_methods) <= {"deskew", "fallback"}
