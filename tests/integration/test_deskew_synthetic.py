"""C2: deskew profile mode on synthetic crosshairs (all four profile_mode x subpixel combinations)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from autocollimator.target_utils import find_cross_center, find_cross_center_detailed

IMAGE_SHAPE = (1088, 1456)  # (h, w): configured sensor size
TRUE_CENTER = (872.0, 544.0)  # (x, y) for shift 0: the default crop center
BACKGROUND = 30.0
PEAK = 210.0
SUPERSAMPLE = 5


def render_crosshair(fwhm: float, tilt_deg: float, shift: float) -> np.ndarray:
    """
    Synthetic crosshair: two Gaussian lines (FWHM ``fwhm`` px) rotated by ``tilt_deg``,
    crossing at TRUE_CENTER + (shift, shift), on BACKGROUND with PEAK on the lines.

    Pixel (row r, col c) has its center at (x=c, y=r) and is the mean of a SUPERSAMPLE x
    SUPERSAMPLE grid of point samples. Only strips around the lines are computed; outside
    them (> 10 sigma from both lines) the image is BACKGROUND.
    """
    h, w = IMAGE_SHAPE
    cx, cy = TRUE_CENTER[0] + shift, TRUE_CENTER[1] + shift
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.radians(tilt_deg)
    c, s = np.cos(t), np.sin(t)
    sub = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
    half = int(np.ceil(max(h, w) * abs(np.tan(t)) + 10 * sigma)) + 2

    img = np.full(IMAGE_SHAPE, BACKGROUND, dtype=np.float64)

    def fill(r0: int, r1: int, c0: int, c1: int) -> None:
        ys = (np.arange(r0, r1)[:, None] + sub[None, :]).ravel()[:, None] - cy
        xs = (np.arange(c0, c1)[:, None] + sub[None, :]).ravel()[None, :] - cx
        g_v = np.exp(-0.5 * ((xs * c - ys * s) / sigma) ** 2)
        g_h = np.exp(-0.5 * ((xs * s + ys * c) / sigma) ** 2)
        g = np.maximum(g_v, g_h).reshape(r1 - r0, SUPERSAMPLE, c1 - c0, SUPERSAMPLE).mean(axis=(1, 3))
        img[r0:r1, c0:c1] = BACKGROUND + (PEAK - BACKGROUND) * g

    xc, yc = int(round(cx)), int(round(cy))
    fill(0, h, max(0, xc - half), min(w, xc + half + 1))  # vertical line strip
    fill(max(0, yc - half), min(h, yc + half + 1), 0, w)  # horizontal line strip
    return img


# Default config geometry. q_limit is relaxed to 0.99: at FWHM 1.5 px pass 1 exceeds the
# default 0.5 at some positions, and these tests are about centering accuracy, not the gate.
PARAMS = dict(crop_center=(872, 544), crop_size=(1018, 652), roi_size=(500, 20), q_limit=0.99)
FWHMS = (1.5, 3.0, 8.0)
SHIFTS = [i / 10 for i in range(11)]
DESKEW_TOLERANCE_PX = {False: 0.06, True: 0.02}  # keyed by subpixel

# find_cross_center centers in average mode, measured with the code before the deskew mode
# was added (commit 68f1acf), keyed by (tilt_deg, fwhm, subpixel), one entry per SHIFTS value.
GOLDEN_AVERAGE = {
    (3.0, 1.5, False): [(871.53, 544.0), (872.52, 543.95), (872.52, 543.95), (872.52, 543.95), (872.55, 544.44), (872.55, 544.44), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94)],
    (3.0, 1.5, True): [(871.74, 544.01), (872.31, 544.02), (872.37, 544.07), (872.42, 544.12), (872.48, 544.43), (872.52, 544.49), (872.57, 544.8), (872.61, 544.85), (872.65, 544.9), (872.7, 544.94), (872.75, 544.98)],
    (3.0, 3.0, False): [(871.53, 544.0), (872.52, 543.95), (872.52, 543.95), (872.52, 543.95), (872.55, 544.44), (872.55, 544.44), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94)],
    (3.0, 3.0, True): [(871.82, 544.01), (872.23, 544.06), (872.3, 544.12), (872.37, 544.19), (872.45, 544.43), (872.51, 544.51), (872.57, 544.74), (872.63, 544.81), (872.69, 544.87), (872.76, 544.93), (872.83, 544.99)],
    (3.0, 8.0, False): [(871.53, 544.0), (872.52, 543.95), (872.52, 543.95), (872.52, 543.95), (872.55, 544.44), (872.55, 544.44), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94), (872.58, 544.94)],
    (3.0, 8.0, True): [(871.95, 544.02), (872.1, 544.11), (872.2, 544.21), (872.3, 544.3), (872.39, 544.43), (872.48, 544.52), (872.57, 544.65), (872.67, 544.75), (872.76, 544.84), (872.85, 544.93), (872.95, 545.02)],
    (0.0, 1.5, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 1.5, True): [(872.0, 544.0), (872.03, 544.03), (872.07, 544.07), (872.12, 544.12), (872.17, 544.17), (872.24, 544.24), (872.83, 544.83), (872.88, 544.88), (872.93, 544.93), (872.97, 544.97), (873.0, 545.0)],
    (0.0, 3.0, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 3.0, True): [(872.0, 544.0), (872.06, 544.06), (872.12, 544.12), (872.18, 544.18), (872.25, 544.25), (872.33, 544.33), (872.75, 544.75), (872.82, 544.82), (872.88, 544.88), (872.94, 544.94), (873.0, 545.0)],
    (0.0, 8.0, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 8.0, True): [(872.0, 544.0), (872.09, 544.09), (872.18, 544.18), (872.28, 544.28), (872.37, 544.37), (872.47, 544.47), (872.63, 544.63), (872.72, 544.72), (872.82, 544.82), (872.91, 544.91), (873.0, 545.0)],
}


@functools.lru_cache(maxsize=None)
def _image(fwhm: float, tilt_deg: float, shift: float) -> np.ndarray:
    img = render_crosshair(fwhm, tilt_deg, shift)
    img.setflags(write=False)
    return img


def _centers(fwhm: float, tilt_deg: float, **kwargs) -> list:
    return [find_cross_center_detailed(_image(fwhm, tilt_deg, p), **PARAMS, **kwargs) for p in SHIFTS]


def _max_error_after_offset(results) -> tuple[float, float]:
    """Max |error| in x and y over SHIFTS after removing the mean (constant) offset."""
    err = np.array([[r.center[0] - TRUE_CENTER[0] - p, r.center[1] - TRUE_CENTER[1] - p]
                    for r, p in zip(results, SHIFTS)])
    err -= err.mean(axis=0)
    return float(np.abs(err[:, 0]).max()), float(np.abs(err[:, 1]).max())


@pytest.mark.integration
@pytest.mark.parametrize("fwhm", FWHMS)
def test_average_mode_without_subpixel_is_unchanged(fwhm):
    expected = GOLDEN_AVERAGE[(3.0, fwhm, False)]
    defaults = [find_cross_center(_image(fwhm, 3.0, p), **PARAMS)[0] for p in SHIFTS]
    assert defaults == expected
    explicit = _centers(fwhm, 3.0, profile_mode="average", subpixel=False)
    assert [r.center for r in explicit] == expected
    assert all(r.roi_methods == ("average",) * 4 for r in explicit)


@pytest.mark.integration
@pytest.mark.parametrize("fwhm", FWHMS)
def test_average_mode_with_subpixel_is_unchanged(fwhm):
    results = _centers(fwhm, 3.0, profile_mode="average", subpixel=True)
    assert [r.center for r in results] == GOLDEN_AVERAGE[(3.0, fwhm, True)]


@pytest.mark.integration
@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("fwhm", FWHMS)
def test_deskew_tracks_subpixel_positions_at_3_degrees(fwhm, subpixel):
    results = _centers(fwhm, 3.0, profile_mode="deskew", subpixel=subpixel)
    assert all(r.center is not None for r in results)
    assert all(r.roi_methods == ("deskew",) * 4 for r in results)
    ex, ey = _max_error_after_offset(results)
    tol = DESKEW_TOLERANCE_PX[subpixel]
    assert ex <= tol and ey <= tol, f"FWHM {fwhm} subpixel={subpixel}: max error x={ex:.4f} y={ey:.4f} > {tol}"


@pytest.mark.integration
@pytest.mark.parametrize("subpixel", [False, True])
@pytest.mark.parametrize("fwhm", FWHMS)
def test_deskew_at_0_degrees_falls_back_to_average(fwhm, subpixel):
    deskew = _centers(fwhm, 0.0, profile_mode="deskew", subpixel=subpixel)
    average = _centers(fwhm, 0.0, profile_mode="average", subpixel=subpixel)
    assert all(r.roi_methods == ("fallback",) * 4 for r in deskew)
    assert [r.center for r in deskew] == [r.center for r in average] == GOLDEN_AVERAGE[(0.0, fwhm, subpixel)]
    assert [r.angles for r in deskew] == [r.angles for r in average]
    assert [r.q_ratio_list for r in deskew] == [r.q_ratio_list for r in average]


@pytest.mark.integration
@pytest.mark.parametrize("subpixel", [False, True])
def test_deskew_never_causes_quality_failure(subpixel):
    # Noisy image with q_limit just above pass 1's worst ROI: pass 1 barely passes, and
    # the pass-2 search (no quality limit) must not turn that into a failure.
    rng = np.random.default_rng(12345)
    img = _image(3.0, 3.0, 0.3) + rng.normal(0.0, 25.0, size=_image(3.0, 3.0, 0.3).shape)
    params = dict(PARAMS)
    probe = find_cross_center_detailed(img, **params, subpixel=subpixel)
    params["q_limit"] = min(0.99, max(probe.q_ratio_list) + 1e-6)

    average = find_cross_center_detailed(img, **params, subpixel=subpixel)
    deskew = find_cross_center_detailed(img, **params, subpixel=subpixel, profile_mode="deskew")
    assert average.center is not None
    assert deskew.center is not None and deskew.angles is not None
    assert deskew.q_ratio_list == average.q_ratio_list
    assert "deskew" in deskew.roi_methods


@pytest.mark.integration
def test_deskew_does_not_run_when_pass_1_fails():
    img = _image(3.0, 3.0, 0.3)
    params = dict(PARAMS, q_limit=0.01)
    result = find_cross_center_detailed(img, **params, profile_mode="deskew")
    assert result.center is None and result.angles is None
    assert result.roi_methods == ("average",) * 4
