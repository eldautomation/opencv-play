"""R21: a flat ROI (q = NaN) gives success=False with a clear reason, in average and deskew mode."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autocollimator.app import AutocollimatorApp

from synthetic import BACKGROUND, render_crosshair

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SHAPE = (1080, 1920)  # (h, w): hd_1080 capture mode of the default device


def _set_profile_mode(app: AutocollimatorApp, profile_mode: str) -> None:
    """Change the active profile mode in memory (config files are not changed)."""
    device = app.get_current_config()
    by_id = app.get_library()["measurement_parameters_by_id"]
    by_id[device.measurement_parameters_id] = replace(app.get_measurement_parameters(), profile_mode=profile_mode)


@pytest.mark.integration
@pytest.mark.parametrize("profile_mode", ["average", "deskew"])
def test_flat_roi_is_reported_as_failure(tmp_path: Path, profile_mode: str) -> None:
    img = render_crosshair(6.0, 0.0, 0.0, shape=SHAPE)
    img[200:260, :] = BACKGROUND  # top ROI (rows 218..237 of the default crop) is perfectly flat
    with AutocollimatorApp(config_dir=CONFIG_DIR, output_dir=tmp_path) as app:
        _set_profile_mode(app, profile_mode)
        output, _overlay = app.run_center_finding_on_image(np.clip(img, 0, 255).astype(np.uint8))

    mv = output.measured_values
    assert mv.success is False
    assert mv.center_position_x is None and mv.center_position_y is None
    assert mv.measured_angle_a0 is None and mv.measured_angle_a1 is None
    assert "ROI top: quality ratio not finite (flat profile)" in mv.message, mv.message
