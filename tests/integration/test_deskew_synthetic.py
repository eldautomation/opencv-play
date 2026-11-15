"""C2: deskew profile mode on synthetic crosshairs (all four profile_mode x subpixel combinations)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from autocollimator.target_utils import find_cross_center, find_cross_center_detailed

from synthetic import TRUE_CENTER, render_crosshair


# Default config geometry. q_limit is relaxed to 0.99: at FWHM 1.5 px pass 1 exceeds the
# default 0.5 at some positions, and these tests are about centering accuracy, not the gate.
PARAMS = dict(crop_center=(872, 544), crop_size=(1018, 652), roi_size=(500, 20), q_limit=0.99)
FWHMS = (1.5, 3.0, 8.0)
SHIFTS = [i / 10 for i in range(11)]
DESKEW_TOLERANCE_PX = {False: 0.06, True: 0.02}  # keyed by subpixel
GOLDEN_TOLERANCE_PX = 0.01

# find_cross_center centers in average mode, keyed by (tilt_deg, fwhm, subpixel), one entry per
# SHIFTS value. Measured before the deskew mode was added (commit 68f1acf) and rounded to 0.01 px
# by line_intersection; re-measured unrounded when that rounding was removed (Phase E, all within
# 0.005 px of the old values). Compared with GOLDEN_TOLERANCE_PX.
GOLDEN_AVERAGE = {
    (3.0, 1.5, False): [(871.526077837999, 543.9986398720401), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5493805242401, 544.4442154786511), (872.5493805242401, 544.4442154786511), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569)],
    (3.0, 1.5, True): [(871.7354199788389, 544.0086470828746), (872.3120888762035, 544.022096834289), (872.3711120812691, 544.0671204255716), (872.4222655805034, 544.1200232942222), (872.4808067912832, 544.4332541499147), (872.5218174219565, 544.4902736473576), (872.5743805583675, 544.8041504910686), (872.6135005782813, 544.8536620200791), (872.6549642734094, 544.8970188071252), (872.7011139664194, 544.9370687211951), (872.7541863473375, 544.9767355975522)],
    (3.0, 3.0, False): [(871.526077837999, 543.9986398720401), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5493805242401, 544.4442154786511), (872.5493805242401, 544.4442154786511), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569)],
    (3.0, 3.0, True): [(871.8210992850214, 544.0143922673367), (872.2265974345836, 544.0571444493055), (872.3045343635777, 544.121219818044), (872.374123774871, 544.1933474157312), (872.4465354742637, 544.4303015736135), (872.5070057698571, 544.505785831338), (872.5744462307803, 544.7433074223974), (872.6333459786557, 544.8117824150298), (872.6941002831736, 544.8743225191313), (872.7589306043456, 544.9339017533537), (872.8304764787673, 544.9931113130758)],
    (3.0, 8.0, False): [(871.526077837999, 543.9986398720401), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5233575820793, 543.9466252815766), (872.5493805242401, 544.4442154786511), (872.5493805242401, 544.4442154786511), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569), (872.5755132580772, 544.9439050256569)],
    (3.0, 8.0, True): [(871.9458830493704, 544.024448796446), (872.1036842390142, 544.1126847907917), (872.2014468583387, 544.2055959517176), (872.2964023674965, 544.3012656525459), (872.3908451641956, 544.4275197944005), (872.4824170949369, 544.5245344415498), (872.5749222276501, 544.6507968569213), (872.6658848961788, 544.7452570102014), (872.7575872419184, 544.8376054706923), (872.8508318601531, 544.9288507379679), (872.946463953826, 545.0199636575481)],
    (0.0, 1.5, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 1.5, True): [(872.0, 544.0), (872.0338747697738, 544.0338747697738), (872.0714632016454, 544.0714632016454), (872.1162556472816, 544.1162556472816), (872.1711941195622, 544.1711941195622), (872.2383138906208, 544.2383138906208), (872.8287786119781, 544.8287786119781), (872.8837193911369, 544.8837193911369), (872.9285135231976, 544.9285135231976), (872.9661030657434, 544.9661030657434), (872.9999784212009, 544.9999784212009)],
    (0.0, 3.0, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 3.0, True): [(872.0, 544.0), (872.0568412255011, 544.0568412255011), (872.1158255802117, 544.1158255802118), (872.1793316871924, 544.1793316871924), (872.2502654647603, 544.2502654647603), (872.3324888952536, 544.3324888952536), (872.7496921056276, 544.7496921056276), (872.82062922033, 544.82062922033), (872.8841377466949, 544.8841377466949), (872.94312381944, 544.94312381944), (872.9999662014789, 544.9999662014789)],
    (0.0, 8.0, False): [(872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (872.0, 544.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0), (873.0, 545.0)],
    (0.0, 8.0, True): [(872.0, 544.0), (872.0905335376726, 544.0905335376726), (872.1819240370453, 544.1819240370453), (872.2750547673679, 544.2750547673679), (872.370863328171, 544.370863328171), (872.4703732967705, 544.4703732967705), (872.6288425784403, 544.6288425784403), (872.7246670516852, 544.7246670516852), (872.8178117232533, 544.8178117232533), (872.9092145294644, 544.9092145294644), (872.9997590267182, 544.999759026718)],
}


def _golden(centers: list) -> list:
    return [pytest.approx(c, abs=GOLDEN_TOLERANCE_PX) for c in centers]


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
    assert defaults == _golden(expected)
    explicit = _centers(fwhm, 3.0, profile_mode="average", subpixel=False)
    assert [r.center for r in explicit] == _golden(expected)
    assert all(r.roi_methods == ("average",) * 4 for r in explicit)


@pytest.mark.integration
@pytest.mark.parametrize("fwhm", FWHMS)
def test_average_mode_with_subpixel_is_unchanged(fwhm):
    results = _centers(fwhm, 3.0, profile_mode="average", subpixel=True)
    assert [r.center for r in results] == _golden(GOLDEN_AVERAGE[(3.0, fwhm, True)])


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
    assert all(r.roi_methods == ("fallback:tilt",) * 4 for r in deskew)
    assert [r.center for r in deskew] == [r.center for r in average]
    assert [r.center for r in average] == _golden(GOLDEN_AVERAGE[(0.0, fwhm, subpixel)])
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
    assert result.roi_methods == ("not_run",) * 4
