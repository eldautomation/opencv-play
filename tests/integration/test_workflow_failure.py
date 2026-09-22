"""R1: a measurement that fails the quality check must be reported as success=False."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


@pytest.mark.integration
def test_quality_failure_is_reported_as_failure(tmp_path: Path) -> None:
    # blob-09.jpg fails the RSS quality limit with the default config.
    img_path = IMAGE_DIR / "blob-09.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.fail(f"Cannot read image: {img_path}")

    with AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs",
        output_dir=tmp_path,
        input_dir=IMAGE_DIR,
    ) as app:
        result, _overlay = app.run_center_finding_on_image(img)

    mv = result.measured_values
    assert mv.success is False, (
        f"Expected success=False for blob-09.jpg, got success={mv.success!r} "
        f"with position=({mv.center_position_x!r}, {mv.center_position_y!r}) "
        f"and message={mv.message!r}"
    )
