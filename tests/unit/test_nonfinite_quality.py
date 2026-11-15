"""R21: a non-finite quality ratio (flat profile) is a quality failure, never a success."""
from __future__ import annotations

import math

import numpy as np
import pytest

from autocollimator.target_utils import find_cross_center_detailed, sdrm_2
from autocollimator.workflow import _quality_failure_message

from synthetic import BACKGROUND, render_crosshair

CROSS_KW = dict(crop_center=(872, 544), crop_size=(1018, 652), roi_size=("auto", 20), q_limit=0.5)
TOP_ROI_ROWS = slice(200, 260)  # contains the top ROI rows 218..237 of the default crop


def _cross_with_flat_top_roi() -> np.ndarray:
    """Untilted crosshair whose vertical line is removed where the top ROI sits."""
    img = render_crosshair(6.0, 0.0, 0.0)
    img[TOP_ROI_ROWS, :] = BACKGROUND
    return img


@pytest.mark.unit
def test_sdrm_2_flat_profile_fails_with_nan_ratio() -> None:
    peak, q = sdrm_2(np.full(41, 50.0), 10, q_limit=0.5, debug=False)
    assert peak is None
    assert math.isnan(q)


@pytest.mark.unit
def test_sdrm_2_finite_ratio_is_unchanged() -> None:
    xs = np.arange(41, dtype=np.float64)
    profile = 30.0 + 180.0 * np.exp(-0.5 * ((xs - 22.0) / 2.0) ** 2)
    peak, q = sdrm_2(profile, 10, q_limit=0.5, debug=False)
    assert peak == 2
    assert math.isfinite(q) and q <= 0.5


@pytest.mark.unit
@pytest.mark.parametrize("profile_mode", ["average", "deskew"])
def test_flat_roi_fails_the_measurement(profile_mode: str, caplog) -> None:
    result = find_cross_center_detailed(_cross_with_flat_top_roi(), profile_mode=profile_mode, **CROSS_KW)
    assert result.center is None and result.angles is None
    assert math.isnan(result.q_ratio_list[0])
    assert all(math.isfinite(q) for q in result.q_ratio_list[1:])
    assert result.roi_methods == ("not_run",) * 4
    assert "ROI top: quality ratio not finite (flat profile)" in caplog.text


@pytest.mark.unit
def test_failure_message_names_the_non_finite_roi() -> None:
    msg = _quality_failure_message([math.nan, 0.2, math.inf, 0.7], 0.5)
    assert "ROI top: quality ratio not finite (flat profile)" in msg
    assert "ROI right: quality ratio not finite (flat profile)" in msg
    assert "left (rss_ratio 0.700 > limit 0.500)" in msg
    assert "bottom" not in msg


@pytest.mark.unit
def test_failure_message_for_finite_ratios_is_unchanged() -> None:
    assert _quality_failure_message([0.6, 0.2, 0.3, 0.7], 0.5) == (
        "Center could not be determined: quality check failed for ROI "
        "top (rss_ratio 0.600 > limit 0.500), left (rss_ratio 0.700 > limit 0.500)"
    )
