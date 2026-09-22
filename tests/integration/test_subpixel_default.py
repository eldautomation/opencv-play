"""R2: with sub-pixel refinement off (the default), center finding is unchanged."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


@pytest.mark.integration
def test_blob_22_default_config_center_unchanged(tmp_path: Path) -> None:
    img_path = IMAGE_DIR / "blob-22.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.fail(f"Cannot read image: {img_path}")

    with AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs",
        output_dir=tmp_path,
        input_dir=IMAGE_DIR,
    ) as app:
        mps = app.get_measurement_parameters()
        qls = app.get_quality_limits()

    assert mps.subpixel is False  # default config leaves sub-pixel refinement off

    position, _angles, _overlay, _q = find_cross_center(
        image=img,
        crop_center=(mps.crop_center_x, mps.crop_center_y),
        crop_size=(mps.crop_size_x, mps.crop_size_y),
        roi_size=(mps.roi_size_x, mps.roi_size_y),
        q_limit=qls.rss_ratio,
    )
    # Value measured before the sub-pixel option was added.
    assert position == (883.72, 566.83)
