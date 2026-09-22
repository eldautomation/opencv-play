"""R1: a measurement that fails the quality check must be reported as success=False."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest
import yaml

from autocollimator.app import AutocollimatorApp
from autocollimator.config.models import MeasuredValues

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


def _load_blob_09():
    # blob-09.jpg fails the RSS quality limit with the default config.
    img_path = IMAGE_DIR / "blob-09.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.fail(f"Cannot read image: {img_path}")
    return img


def _app(output_dir: Path) -> AutocollimatorApp:
    return AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs",
        output_dir=output_dir,
        input_dir=IMAGE_DIR,
    )


@pytest.mark.integration
def test_quality_failure_is_reported_as_failure(tmp_path: Path) -> None:
    img = _load_blob_09()

    with _app(tmp_path) as app:
        result, _overlay = app.run_center_finding_on_image(img)

    mv = result.measured_values
    assert mv.success is False, (
        f"Expected success=False for blob-09.jpg, got success={mv.success!r} "
        f"with position=({mv.center_position_x!r}, {mv.center_position_y!r}) "
        f"and message={mv.message!r}"
    )
    assert mv.center_position_x is None and mv.center_position_y is None
    assert mv.measured_angle_a0 is None and mv.measured_angle_a1 is None
    # right and left ROIs exceed the default rss_ratio limit (0.5) on blob-09.
    assert "right (rss_ratio" in mv.message and "left (rss_ratio" in mv.message, mv.message
    assert "top (rss_ratio" not in mv.message and "bottom (rss_ratio" not in mv.message, mv.message


@pytest.mark.integration
def test_failed_measurement_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    img = _load_blob_09()

    with _app(tmp_path) as app:
        result, overlay = app.run_center_finding_on_image(img)
        input_path, overlay_path, yaml_path = app.save_measurement_output(result, img, overlay)

    assert input_path.is_file() and overlay_path.is_file() and yaml_path.is_file()

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    reloaded = MeasuredValues.from_dict(raw["measured_values"])
    assert reloaded == result.measured_values
    assert reloaded.success is False
    assert reloaded.center_position_x is None
