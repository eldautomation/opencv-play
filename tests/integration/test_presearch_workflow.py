"""R4: pre-search modes in the measurement workflow (off / every_image / track, failure actions)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autocollimator import workflow as workflow_module
from autocollimator.app import AutocollimatorApp
from autocollimator.config.models import PresearchParameters
from autocollimator.presearch import recommend_region
from autocollimator.target_utils import find_cross_center_detailed

from synthetic import render_crosshair

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SHAPE = (1080, 1920)  # (h, w): hd_1080 capture mode of the default device
POSITION_TOLERANCE_PX = 0.5


def _image(center, seed: int = 0) -> np.ndarray:
    # Mild noise, as in a real image; a perfectly flat background is not realistic.
    img = render_crosshair(3.0, 0.4, 0.0, shape=SHAPE, center=center)
    return np.clip(img + np.random.default_rng(seed).normal(0.0, 2.0, SHAPE), 0, 255)


def _set_presearch(app: AutocollimatorApp, **fields) -> None:
    """Change the active pre-search settings in memory (config files are not changed)."""
    device = app.get_current_config()
    by_id = app.get_library()["measurement_parameters_by_id"]
    mp = app.get_measurement_parameters()
    by_id[device.measurement_parameters_id] = replace(mp, presearch=PresearchParameters(**fields))


def _configured_measurement(app: AutocollimatorApp, img: np.ndarray, center=None, size=None):
    mps, qls = app.get_measurement_parameters(), app.get_quality_limits()
    return find_cross_center_detailed(
        img,
        center or (mps.crop_center_x, mps.crop_center_y),
        size or (mps.crop_size_x, mps.crop_size_y),
        (mps.roi_size_x, mps.roi_size_y),
        qls.rss_ratio,
        subpixel=mps.subpixel,
        subpixel_neighbors=mps.subpixel_neighbors,
        profile_mode=mps.profile_mode,
        gray_mode=mps.gray_mode,
    )


def _near(mv, center) -> bool:
    return math.hypot(mv.center_position_x - center[0], mv.center_position_y - center[1]) <= POSITION_TOLERANCE_PX


@pytest.fixture
def app(tmp_path):
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as a:
        yield a


@pytest.fixture
def presearch_calls(monkeypatch):
    calls = []
    real = workflow_module.presearch

    def counting(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(workflow_module, "presearch", counting)
    return calls


@pytest.mark.integration
def test_off_is_the_configured_crop_and_never_runs_presearch(app, presearch_calls) -> None:
    assert app.get_measurement_parameters().presearch.presearch_mode == "off"
    img = _image((900.0, 560.0))
    output, _overlay = app.run_center_finding_on_image(img)
    expected = _configured_measurement(app, img)
    mv = output.measured_values
    assert mv.success and expected.center is not None
    assert (mv.center_position_x, mv.center_position_y) == expected.center
    assert (mv.measured_angle_a0, mv.measured_angle_a1) == expected.angles
    assert mv.message == "Center found"
    assert presearch_calls == []


@pytest.mark.integration
def test_every_image_measures_with_the_recommended_crop(app, presearch_calls) -> None:
    center = (300.0, 250.0)  # outside the configured crop
    img = _image(center)
    assert not app.run_center_finding_on_image(img)[0].measured_values.success  # off: fails

    _set_presearch(app, presearch_mode="every_image")
    output, _overlay = app.run_center_finding_on_image(img)
    mv = output.measured_values
    assert len(presearch_calls) == 1 and presearch_calls[0].success
    pr = presearch_calls[0]
    assert mv.success and _near(mv, center), mv.message
    expected = _configured_measurement(
        app, img, (pr.recommended_center_x, pr.recommended_center_y), (pr.recommended_size, pr.recommended_size)
    )
    assert (mv.center_position_x, mv.center_position_y) == expected.center
    assert mv.message == "Center found"


@pytest.mark.integration
def test_track_uses_previous_center_and_falls_back_to_presearch(app, presearch_calls) -> None:
    _set_presearch(app, presearch_mode="track")
    assert app.tracked_center is None

    # No previous result: pre-search.
    first = (700.0, 500.0)
    mv = app.run_center_finding_on_image(_image(first, seed=1))[0].measured_values
    assert mv.success and _near(mv, first) and len(presearch_calls) == 1
    assert app.tracked_center == (mv.center_position_x, mv.center_position_y)

    # Small move: measured around the tracked center with the recommended size, no pre-search.
    second = (730.0, 520.0)
    img = _image(second, seed=2)
    region = recommend_region(
        app.tracked_center, (SHAPE[1], SHAPE[0]), app.get_measurement_parameters().presearch,
        app.tracked_line_fwhm_px,
    )
    mv = app.run_center_finding_on_image(img)[0].measured_values
    assert mv.success and _near(mv, second) and len(presearch_calls) == 1
    expected = _configured_measurement(app, img, region.center, (region.size, region.size))
    assert (mv.center_position_x, mv.center_position_y) == expected.center

    # Large jump: the tracked measurement fails, the pre-search finds the new position.
    third = (1500.0, 850.0)
    mv = app.run_center_finding_on_image(_image(third, seed=3))[0].measured_values
    assert mv.success and _near(mv, third), mv.message
    assert len(presearch_calls) == 2
    assert app.tracked_center == (mv.center_position_x, mv.center_position_y)


@pytest.mark.integration
def test_track_with_failing_presearch_applies_failure_action(app, presearch_calls) -> None:
    _set_presearch(app, presearch_mode="track", presearch_failure_action="fail")
    blank = np.clip(30.0 + np.random.default_rng(4).normal(0.0, 2.0, SHAPE), 0, 255)
    mv = app.run_center_finding_on_image(blank)[0].measured_values
    assert mv.success is False
    assert mv.message == "Pre-search failed: vertical and horizontal lines not found"
    assert len(presearch_calls) == 1 and app.tracked_center is None


@pytest.mark.integration
def test_failure_action_use_configured_crop(app) -> None:
    # The crosshair is inside the configured crop; the pre-search fails on purpose (minimum
    # region larger than the image), so the configured crop is used and the failure noted.
    center = (900.0, 560.0)
    img = _image(center)
    _set_presearch(app, presearch_mode="every_image", min_crop_size_px=5000)
    mv = app.run_center_finding_on_image(img)[0].measured_values
    expected = _configured_measurement(app, img)
    assert mv.success and (mv.center_position_x, mv.center_position_y) == expected.center
    assert mv.message.startswith("Center found (pre-search failed: region below minimum size")
    assert mv.message.endswith("; configured crop used)")


@pytest.mark.integration
def test_failure_action_fail(app) -> None:
    img = _image((900.0, 560.0))
    _set_presearch(app, presearch_mode="every_image", min_crop_size_px=5000, presearch_failure_action="fail")
    output, overlay = app.run_center_finding_on_image(img)
    mv = output.measured_values
    assert mv.success is False
    assert mv.message.startswith("Pre-search failed: region below minimum size"), mv.message
    assert (mv.center_position_x, mv.center_position_y, mv.measured_angle_a0, mv.measured_angle_a1) == (None,) * 4
    assert (mv.roi_method_m0, mv.roi_method_m1, mv.roi_method_m2, mv.roi_method_m3) == ("not_run",) * 4
    assert all(math.isnan(q) for q in (mv.rss_ratio_r0, mv.rss_ratio_r1, mv.rss_ratio_r2, mv.rss_ratio_r3))
    assert output.angular_measurement.success is False and output.angular_measurement.tilt_x is None
    assert overlay.shape == (*SHAPE, 3)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.integration
def test_on_demand_presearch_writes_nothing(app) -> None:
    before = _tree_digest(CONFIG_DIR)
    center = (1300.0, 300.0)
    result = app.run_presearch(_image(center))
    assert result.success
    assert math.hypot(result.position_x - center[0], result.position_y - center[1]) <= 2.0
    assert app.tracked_center is None
    assert _tree_digest(CONFIG_DIR) == before
    # Also when the active settings say "off": the on-demand call always runs.
    assert app.get_measurement_parameters().presearch.presearch_mode == "off"
