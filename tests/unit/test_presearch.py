"""R4: pre-search on synthetic crosshairs (image layer)."""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest
import yaml

from autocollimator import presearch as presearch_module
from autocollimator.config.models import PresearchParameters, PresearchResult
from autocollimator.instrument import ReticleType
from autocollimator.presearch import presearch, recommend_region

from synthetic import IMAGE_SHAPE, render_crosshair, render_with_ghost

H, W = IMAGE_SHAPE
DEFAULTS = PresearchParameters()
POSITION_TOLERANCE_PX = 2.0
FWHM = 3.0
TILT_DEG = 0.4


def _run(center, params=DEFAULTS, fwhm=FWHM, tilt_deg=TILT_DEG, **kwargs) -> PresearchResult:
    img = render_crosshair(fwhm, tilt_deg, 0.0, center=center, **kwargs)
    return presearch(img, params, ReticleType.CROSSHAIR)


def _assert_near(result: PresearchResult, center) -> None:
    assert result.success, result.failure_reason
    dx, dy = result.position_x - center[0], result.position_y - center[1]
    assert math.hypot(dx, dy) <= POSITION_TOLERANCE_PX, (result.position_x, result.position_y, center)


def _assert_region_contains(result: PresearchResult, center) -> None:
    half = result.recommended_size / 2
    k = presearch_module.PRESEARCH_CENTER_MARGIN_FRACTION * result.recommended_size
    x0, y0 = result.recommended_center_x - result.recommended_size // 2, result.recommended_center_y - result.recommended_size // 2
    assert 0 <= x0 and x0 + result.recommended_size <= W and 0 <= y0 and y0 + result.recommended_size <= H
    assert x0 + k - 1 <= center[0] <= x0 + 2 * half - k + 1
    assert y0 + k - 1 <= center[1] <= y0 + 2 * half - k + 1


# Inside the default crop, outside it, and near each image edge (still a full-size region).
POSITIONS = [
    (872.0, 544.0), (700.3, 500.7), (250.0, 250.0), (1250.0, 900.0), (200.5, 950.2), (1300.0, 150.0),
    (150.0, 544.0), (W - 150.0, 544.0), (728.0, 150.0), (728.0, H - 150.0),
]


@pytest.mark.unit
@pytest.mark.parametrize("center", POSITIONS)
def test_presearch_finds_crosshair_anywhere(center) -> None:
    result = _run(center)
    _assert_near(result, center)
    assert result.method_used == "sdrm_full_frame"
    assert result.failure_reason is None
    assert result.downsample == DEFAULTS.presearch_downsample
    assert len(result.q_values) == 4 and all(q < DEFAULTS.presearch_q_limit for q in result.q_values)
    _assert_region_contains(result, center)


@pytest.mark.unit
def test_region_is_square_fraction_of_short_side_when_centered() -> None:
    result = _run((872.0, 544.0))
    assert result.recommended_size == int(0.5 * min(W, H))
    assert (result.region_shifted, result.region_shrunk) == (False, False)
    assert abs(result.recommended_center_x - 872) <= 1 and abs(result.recommended_center_y - 544) <= 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "center, edge",
    [((150.0, 544.0), "left"), ((W - 150.0, 544.0), "right"), ((728.0, 150.0), "top"), ((728.0, H - 150.0), "bottom")],
)
def test_region_is_shifted_inward_near_each_edge(center, edge) -> None:
    result = _run(center)
    assert result.success, result.failure_reason
    assert result.region_shifted and not result.region_shrunk
    s = result.recommended_size
    assert s == int(0.5 * min(W, H))
    x0, y0 = result.recommended_center_x - s // 2, result.recommended_center_y - s // 2
    expected = {"left": x0 == 0, "right": x0 + s == W, "top": y0 == 0, "bottom": y0 + s == H}
    assert expected[edge], (x0, y0, s)


@pytest.mark.unit
def test_region_is_shrunk_close_to_edge() -> None:
    # 100 px from the left edge: a 544 px region would put the crosshair outside its middle half.
    center = (100.0, 544.0)
    result = _run(center)
    _assert_near(result, center)
    assert result.region_shrunk and result.region_shifted
    # Largest size keeping the crosshair 0.25 x size from the left edge.
    assert result.recommended_size == math.floor(result.position_x / 0.25)
    _assert_region_contains(result, center)


@pytest.mark.unit
@pytest.mark.parametrize("center", [(50.0, 544.0), (W - 50.0, 544.0), (728.0, 40.0), (728.0, H - 40.0)])
def test_presearch_fails_when_region_would_be_below_minimum(center) -> None:
    result = _run(center)
    assert result.success is False
    assert result.failure_reason.startswith("region below minimum size"), result.failure_reason
    assert result.recommended_size is None and result.recommended_center_x is None
    assert result.region_shrunk


@pytest.mark.unit
def test_recommend_region_rules() -> None:
    size = (W, H)
    r = recommend_region((728.0, 544.0), size, DEFAULTS)
    assert (r.center, r.size, r.shifted, r.shrunk, r.failure_reason) == ((728, 544), 544, False, False, None)

    # min_crop_size_px floor
    r = recommend_region((728.0, 544.0), size, PresearchParameters(presearch_crop_fraction=0.1))
    assert r.size == DEFAULTS.min_crop_size_px
    # 4 x FWHM floor
    r = recommend_region((728.0, 544.0), size, PresearchParameters(presearch_crop_fraction=0.1), line_fwhm_px=100.0)
    assert r.size == 400
    # shrink near the edge: crosshair must stay >= 0.25 x size from the region edge
    r = recommend_region((100.0, 544.0), size, DEFAULTS)
    assert (r.size, r.shrunk, r.shifted, r.center) == (400, True, True, (200, 544))
    # fails below the minimum
    r = recommend_region((50.0, 544.0), size, DEFAULTS)
    assert r.center is None and r.size is None and r.failure_reason.startswith("region below minimum size")
    # minimum larger than the image
    r = recommend_region((728.0, 544.0), size, PresearchParameters(min_crop_size_px=2000))
    assert r.failure_reason.startswith("region below minimum size")


@pytest.mark.unit
def test_region_respects_four_fwhm_for_wide_lines() -> None:
    params = PresearchParameters(presearch_crop_fraction=0.1, min_crop_size_px=50)
    result = _run((728.0, 544.0), params, fwhm=80.0, tilt_deg=0.0)
    assert result.success, result.failure_reason
    assert result.line_fwhm_px == pytest.approx(80.0, rel=0.2)
    assert result.recommended_size == math.ceil(4 * result.line_fwhm_px)


@pytest.mark.unit
@pytest.mark.parametrize("downsample", [1, 2, 4])
def test_coordinates_are_full_resolution_for_any_downsample(downsample) -> None:
    center = (400.3, 700.6)
    result = _run(center, PresearchParameters(presearch_downsample=downsample))
    _assert_near(result, center)
    assert result.downsample == downsample
    assert abs(result.recommended_center_x - center[0]) <= 1.5 and abs(result.recommended_center_y - center[1]) <= 1.5
    vertical = [c for c in result.candidates if c.line == "vertical"]
    assert abs(vertical[0].position - center[0]) <= POSITION_TOLERANCE_PX


@pytest.mark.unit
def test_blank_image_fails_with_reason() -> None:
    result = presearch(np.full(IMAGE_SHAPE, 80.0), DEFAULTS, ReticleType.CROSSHAIR)
    assert result.success is False
    assert result.failure_reason == "vertical and horizontal lines not found"
    assert result.method_used is None and result.candidates == ()


@pytest.mark.unit
def test_pure_noise_fails_with_reason() -> None:
    rng = np.random.default_rng(1)
    img = rng.normal(100.0, 25.0, IMAGE_SHAPE).clip(0, 255).astype(np.uint8)
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    assert result.success is False
    assert result.failure_reason == "vertical and horizontal lines not found"


@pytest.mark.unit
def test_one_missing_line_is_named() -> None:
    img = np.full(IMAGE_SHAPE, 30.0)
    img[:, 700:704] = 200.0  # vertical line only
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    assert result.failure_reason == "horizontal line not found"


@pytest.mark.unit
def test_ghost_above_ambiguity_ratio_warns_and_strongest_is_chosen() -> None:
    center, ghost = (600.0, 500.0), (1000.0, 820.0)
    img = render_with_ghost(FWHM, center, ghost, 0.85)
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    _assert_near(result, center)
    assert len(result.warnings) == 2, result.warnings
    v_warn = next(w for w in result.warnings if "vertical" in w)
    h_warn = next(w for w in result.warnings if "horizontal" in w)
    assert "x=600" in v_warn and "x=1000" in v_warn and "using x=600" in v_warn
    assert "y=500" in h_warn and "y=820" in h_warn and "using y=500" in h_warn
    strengths = {c.line: [] for c in result.candidates}
    for c in result.candidates:
        strengths[c.line].append(c.strength)
    assert all(s == sorted(s, reverse=True) and len(s) >= 2 for s in strengths.values())


@pytest.mark.unit
def test_ghost_below_ambiguity_ratio_is_not_warned() -> None:
    img = render_with_ghost(FWHM, (600.0, 500.0), (1000.0, 820.0), 0.5)
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    _assert_near(result, (600.0, 500.0))
    assert result.warnings == ()
    assert len(result.candidates) >= 4  # the ghost is still listed as a candidate


@pytest.mark.unit
def test_projection_fallback_when_main_method_fails() -> None:
    # A bright patch in the top-left corner pulls the full-frame strips (top and left) away
    # from the crosshair; the validation pass catches that and the projection takes over.
    center = (1000.0, 700.0)
    img = render_crosshair(FWHM, TILT_DEG, 0.0, center=center)
    img[:250, :250] += 120.0
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    _assert_near(result, center)
    assert result.method_used == "projection"
    assert len(result.q_values) == 2 and all(q < DEFAULTS.presearch_q_limit for q in result.q_values)
    assert any(w.startswith("sdrm_full_frame failed (") and "projection fallback" in w for w in result.warnings)
    _assert_region_contains(result, center)


@pytest.mark.unit
def test_projection_is_not_used_when_main_method_succeeds(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("projection must not run")

    monkeypatch.setattr(presearch_module, "_projection", boom)
    result = _run((700.3, 500.7))
    assert result.success and result.method_used == "sdrm_full_frame"


@pytest.mark.unit
def test_both_methods_failing_names_both_reasons() -> None:
    # A strong ramp breaks SDRM everywhere, also in the validation pass of the projection.
    img = render_crosshair(FWHM, TILT_DEG, 0.0, center=(400.0, 300.0))
    img = img + np.linspace(0, 120, W)[None, :] + np.linspace(0, 120, H)[:, None]
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    assert result.success is False and result.method_used == "projection"
    assert result.failure_reason.startswith("sdrm_full_frame: quality check failed for strip"), result.failure_reason
    assert "; projection: validation failed" in result.failure_reason


@pytest.mark.unit
def test_unsupported_reticle_type_fails_with_reason(monkeypatch) -> None:
    img = render_crosshair(FWHM, 0.0, 0.0)
    result = presearch(img, DEFAULTS, "ring")
    assert result.success is False and result.failure_reason == "reticle type 'ring' is not supported by the pre-search"

    monkeypatch.setattr(presearch_module, "_PRESEARCH_BY_RETICLE", {})
    result = presearch(img, DEFAULTS, ReticleType.CROSSHAIR)
    assert result.failure_reason == "reticle type 'crosshair' is not supported by the pre-search"


@pytest.mark.unit
def test_presearch_rejects_invalid_input() -> None:
    with pytest.raises(TypeError):
        presearch([[1, 2]], DEFAULTS, ReticleType.CROSSHAIR)
    with pytest.raises(ValueError):
        presearch(np.zeros((0, 0)), DEFAULTS, ReticleType.CROSSHAIR)
    with pytest.raises(ValueError):
        presearch(np.zeros((100, 100)), DEFAULTS, ReticleType.CROSSHAIR, roi_depth_px=0)


@pytest.mark.unit
def test_presearch_result_invariants_and_round_trip() -> None:
    with pytest.raises(ValueError, match="cannot be None"):
        PresearchResult(True, None, "sdrm_full_frame", 1.0, 2.0, None, None, None)
    with pytest.raises(ValueError, match="failure_reason"):
        PresearchResult(False, None, None, None, None, None, None, None)
    with pytest.raises(ValueError, match="method_used"):
        PresearchResult(False, "x", "magic", None, None, None, None, None)

    result = _run((700.3, 500.7))
    assert PresearchResult.from_dict(asdict(result)) == result
    # Saved through the measurement YAML: plain Python types only.
    reloaded = PresearchResult.from_dict(yaml.safe_load(yaml.safe_dump(asdict(result))))
    assert reloaded == result
