"""R12: the workflow measures with the configured ROI length ("auto" or an explicit int)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autocollimator import workflow as workflow_module
from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center_detailed

from synthetic import render_crosshair

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SHAPE = (1080, 1920)  # (h, w): hd_1080 capture mode of the default device


@pytest.mark.integration
@pytest.mark.parametrize("roi_length", ["auto", 400])
def test_workflow_uses_configured_roi_length(tmp_path: Path, monkeypatch, roi_length) -> None:
    widths = []

    def recording(*args, **kwargs):
        result = find_cross_center_detailed(*args, **kwargs)
        widths.extend(t.roi.shape[1] for t in result.rois)
        return result

    monkeypatch.setattr(workflow_module, "find_cross_center_detailed", recording)
    img = np.clip(render_crosshair(6.0, 3.0, 0.3, shape=SHAPE), 0, 255).astype(np.uint8)
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        device = app.get_current_config()
        by_id = app.get_library()["measurement_parameters_by_id"]
        mps = replace(app.get_measurement_parameters(), roi_size_x=roi_length)
        by_id[device.measurement_parameters_id] = mps
        output, _overlay = app.run_center_finding_on_image(img)
        q_limit = app.get_quality_limits().rss_ratio

    expected = find_cross_center_detailed(
        img, (mps.crop_center_x, mps.crop_center_y), (mps.crop_size_x, mps.crop_size_y),
        (roi_length, mps.roi_size_y), q_limit,
    )
    mv = output.measured_values
    assert mv.success and expected.center is not None
    assert (mv.center_position_x, mv.center_position_y) == expected.center
    assert output.measurement_parameters.roi_size_x == roi_length
    assert widths == [652 if roi_length == "auto" else roi_length] * 4  # min(1018, 652) for "auto"
