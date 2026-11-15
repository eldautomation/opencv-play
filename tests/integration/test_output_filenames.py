"""R13: measurement output filenames are unique (millisecond timestamp plus a counter)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.config.store import load_measurement_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"


@pytest.fixture
def measured(tmp_path: Path):
    img = cv2.imread(str(IMAGE_DIR / "blob-22.jpg"))
    if img is None:
        pytest.fail("Cannot read blob-22.jpg")
    with AutocollimatorApp(config_dir=PROJECT_ROOT / "configs", output_dir=tmp_path, input_dir=IMAGE_DIR) as app:
        result, overlay = app.run_center_finding_on_image(img)
        yield app, result, img, overlay


@pytest.mark.integration
def test_quick_successive_saves_get_different_filenames(measured) -> None:
    app, result, img, overlay = measured
    first = app.save_measurement_output(result, img, overlay)
    second = app.save_measurement_output(result, img, overlay)

    assert set(first).isdisjoint(second)
    assert all(p.is_file() for p in (*first, *second))
    names = [load_measurement_output(p).measurement_name for p in (first[2], second[2])]
    assert names[0] != names[1]
    assert [first[2].name, second[2].name] == [f"{n}_measurement.yaml" for n in names]


@pytest.mark.integration
def test_prefix_has_milliseconds(measured) -> None:
    app = measured[0]
    prefix = app.build_output_prefix(datetime(2026, 11, 15, 9, 8, 7, 654321))
    assert prefix.endswith("_2026-11-15_09-08-07-654")
    assert prefix != app.build_output_prefix(datetime(2026, 11, 15, 9, 8, 7, 655321))


@pytest.mark.integration
def test_same_prefix_gets_a_counter_suffix(measured, monkeypatch) -> None:
    app, result, img, overlay = measured
    monkeypatch.setattr(app, "build_output_prefix", lambda timestamp=None: "fixed")
    paths = [app.save_measurement_output(result, img, overlay)[2] for _ in range(3)]
    assert [p.name for p in paths] == ["fixed_measurement.yaml", "fixed_1_measurement.yaml", "fixed_2_measurement.yaml"]
    assert [load_measurement_output(p).measurement_name for p in paths] == ["fixed", "fixed_1", "fixed_2"]
    # A bundle is never split: one existing file of a bundle is enough to skip its prefix.
    (paths[0].parent / "fixed_3_overlay.png").write_bytes(b"")
    assert app.save_measurement_output(result, img, overlay)[2].name == "fixed_4_measurement.yaml"
