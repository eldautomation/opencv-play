"""R2: find_cross_center must track known sub-pixel shifts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"

SHIFTS_PX = [round(i * 0.1, 1) for i in range(11)]
TOLERANCE_PX = 0.1


@pytest.fixture(scope="module")
def default_params(tmp_path_factory):
    """Crop/ROI/q_limit from the default config, as the workflow uses them."""
    with AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs",
        output_dir=tmp_path_factory.mktemp("subpixel"),
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


@pytest.fixture(scope="module")
def reference(default_params):
    """blob-22.jpg and its unshifted measured center."""
    img_path = IMAGE_DIR / "blob-22.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.fail(f"Cannot read image: {img_path}")
    position, _angles, _overlay, q_ratios = find_cross_center(image=img, subpixel=True, **default_params)
    if position is None or position[0] is None:
        pytest.fail(f"Reference measurement on blob-22.jpg failed (q_ratios={q_ratios})")
    return img, position


@pytest.mark.integration
@pytest.mark.parametrize("shift", SHIFTS_PX)
def test_subpixel_shift_is_tracked(shift, default_params, reference) -> None:
    img, (x0, y0) = reference
    h, w = img.shape[:2]
    m = np.float32([[1, 0, shift], [0, 1, shift]])
    shifted = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC)

    position, _angles, _overlay, q_ratios = find_cross_center(image=shifted, subpixel=True, **default_params)
    assert position is not None and position[0] is not None, (
        f"Measurement failed at shift {shift} px (q_ratios={q_ratios})"
    )

    dx = position[0] - x0
    dy = position[1] - y0
    assert abs(dx - shift) < TOLERANCE_PX and abs(dy - shift) < TOLERANCE_PX, (
        f"True shift {shift:.1f} px, measured (dx={dx:.3f}, dy={dy:.3f}) px; "
        f"error (x={dx - shift:+.3f}, y={dy - shift:+.3f}) exceeds {TOLERANCE_PX} px"
    )
