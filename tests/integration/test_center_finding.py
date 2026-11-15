"""R8/R9: two-pass center finding on the test images, checked against
tests/assets/images/manifest.yaml.

Each manifest entry: file, expect (success | failure), optional expected_x / expected_y
(refined position, compared with POSITION_TOLERANCE_PX), optional notes. Propose a manifest with
``python3 scripts/regenerate_expected.py`` and review it before committing.

To keep the debug overlays under outputs/integration/center_finding/, run with
``pytest -m integration --keep-output`` (or KEEP_OUTPUT=1).
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import clear_folder
from center_finding_two_pass import image_files, load_manifest, measure_two_pass

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"
MANIFEST_PATH = IMAGE_DIR / "manifest.yaml"
POSITION_TOLERANCE_PX = 0.2

MANIFEST = load_manifest(MANIFEST_PATH) if MANIFEST_PATH.exists() else None
NO_MANIFEST = "manifest.yaml not committed yet; run scripts/regenerate_expected.py and review its output"


def _keep_output() -> bool:
    return (
        os.getenv("KEEP_OUTPUT", "0") == "1"
        or "--keep-output" in os.getenv("PYTEST_ADDOPTS", "")
        or getattr(pytest, "keep_output_flag", False)
    )


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    if _keep_output():
        path = PROJECT_ROOT / "outputs" / "integration" / "center_finding"
        clear_folder(path, create=True)
        return path
    return tmp_path_factory.mktemp("center_finding")


@pytest.fixture(scope="module")
def app(out_dir):
    with AutocollimatorApp(config_dir=PROJECT_ROOT / "configs", output_dir=out_dir, input_dir=IMAGE_DIR) as a:
        yield a


@pytest.mark.integration
@pytest.mark.parametrize(
    "entry",
    MANIFEST if MANIFEST is not None else [pytest.param(None, marks=pytest.mark.skip(reason=NO_MANIFEST))],
    ids=(lambda e: e["file"] if e else "no-manifest"),
)
def test_center_finding_matches_manifest(entry, app, out_dir) -> None:
    img_path = IMAGE_DIR / entry["file"]
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.fail(f"Cannot read image listed in manifest: {entry['file']}")

    result = measure_two_pass(app, img, debug=True, debug_prefix=out_dir / Path(entry["file"]).stem)
    LOGGER.info("%s: success=%s pass1=%s refined=%s (%s)",
                entry["file"], result.success, result.pass1_position, result.position, result.message)

    if entry["expect"] == "failure":
        assert result.success is False, f"{entry['file']}: expected failure, got center {result.position}"
        return

    assert result.success is True, f"{entry['file']}: expected success, got failure: {result.message}"
    if "expected_x" in entry:
        x, y = result.position
        dx, dy = x - entry["expected_x"], y - entry["expected_y"]
        assert abs(dx) <= POSITION_TOLERANCE_PX and abs(dy) <= POSITION_TOLERANCE_PX, (
            f"{entry['file']}: refined center ({x}, {y}) vs expected "
            f"({entry['expected_x']}, {entry['expected_y']}): diff ({dx:+.3f}, {dy:+.3f}) px, "
            f"tolerance {POSITION_TOLERANCE_PX} px"
        )


@pytest.mark.integration
@pytest.mark.skipif(MANIFEST is None, reason=NO_MANIFEST)
def test_manifest_covers_images() -> None:
    """Images missing from the manifest are reported, not failed; stale entries fail."""
    images = set(image_files(IMAGE_DIR))
    listed = {e["file"] for e in MANIFEST}
    unlisted = sorted(images - listed)
    if unlisted:
        msg = f"{len(unlisted)} image(s) not in manifest.yaml (not tested): {', '.join(unlisted)}"
        LOGGER.warning(msg)
        warnings.warn(msg)
    stale = sorted(listed - images)
    assert not stale, f"manifest.yaml lists images that do not exist: {stale}"
