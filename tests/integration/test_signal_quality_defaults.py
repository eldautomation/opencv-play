"""Signal-quality phase: the default configuration still gives today's blob-22 result."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center_detailed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


def _blob_22():
    img = cv2.imread(str(IMAGE_DIR / "blob-22.jpg"))
    if img is None:
        pytest.fail("Cannot read image: blob-22.jpg")
    return img


@pytest.mark.integration
def test_blob_22_default_workflow_unchanged(tmp_path: Path) -> None:
    with AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs", output_dir=tmp_path, input_dir=IMAGE_DIR
    ) as app:
        assert app.get_measurement_parameters().gray_mode == "luma"
        output, _overlay = app.run_center_finding_on_image(_blob_22())
    mv = output.measured_values
    # Values measured before the signal-quality phase (gray_mode, saturation, adaptive window).
    assert mv.success is True
    # Center re-measured unrounded when line_intersection stopped rounding (Phase E; was (883.72, 566.83)).
    assert (mv.center_position_x, mv.center_position_y) == pytest.approx((883.7198956447866, 566.825883389706), abs=0.01)
    assert (mv.measured_angle_a0, mv.measured_angle_a1) == (2.993723861728256, 3.1575496667574487)
    assert (mv.rss_ratio_r0, mv.rss_ratio_r1, mv.rss_ratio_r2, mv.rss_ratio_r3) == (
        0.07555122678593008,
        0.15635433472537652,
        0.15601477983851808,
        0.28049225611597295,
    )
    assert (mv.roi_method_m0, mv.roi_method_m1, mv.roi_method_m2, mv.roi_method_m3) == ("average",) * 4
    assert mv.overlay_hash == "018ef863baad9951a7874a3d2eb6b898"


@pytest.mark.integration
def test_blob_22_explicit_luma_is_identical_to_default() -> None:
    img = _blob_22()
    kw = dict(crop_center=(872, 544), crop_size=(1018, 652), roi_size=(500, 20), q_limit=0.5)
    default = find_cross_center_detailed(img, **kw)
    luma = find_cross_center_detailed(img, gray_mode="luma", **kw)
    assert luma.center == default.center
    assert luma.angles == default.angles
    assert luma.q_ratio_list == default.q_ratio_list
    assert luma.roi_methods == default.roi_methods
