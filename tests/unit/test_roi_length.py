"""R12: ROI length ("auto" or an explicit int), nominal_slant, find_center_pixel methods."""
from __future__ import annotations

import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from autocollimator.config.models import MeasurementParameters, PresearchParameters
from autocollimator.config.store import (
    ConfigError,
    add_measurement_parameters,
    load_library,
    load_main,
    validate_device_references,
)
from autocollimator.target_utils import find_center_pixel, find_cross_center_detailed

from synthetic import render_crosshair

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CROSS_KW = dict(crop_center=(872, 544), crop_size=(1018, 652), q_limit=0.5)
MP_TOML = """
[[measurement_parameters]]
id = "901"
name = "Default Parameters"
crop_center_x = 872
crop_center_y = 544
crop_size_x = 1018
crop_size_y = 652
roi_size_y = 20
"""


def _load(extra: str = "") -> MeasurementParameters:
    return MeasurementParameters.from_dict(tomllib.loads(MP_TOML + extra)["measurement_parameters"][0])


@pytest.fixture(scope="module")
def cross() -> np.ndarray:
    return render_crosshair(6.0, 3.0, 0.3)


# --- find_cross_center ----------------------------------------------------------------------


@pytest.mark.unit
def test_auto_is_min_crop_size(cross) -> None:
    auto = find_cross_center_detailed(cross, roi_size=("auto", 20), **CROSS_KW)
    explicit = find_cross_center_detailed(cross, roi_size=(652, 20), **CROSS_KW)
    assert auto.center is not None
    assert auto.center == explicit.center
    assert auto.angles == explicit.angles
    assert auto.q_ratio_list == explicit.q_ratio_list
    assert all(t.roi.shape == (20, 652) for t in auto.rois)


@pytest.mark.unit
def test_explicit_roi_length_is_honored(cross) -> None:
    result = find_cross_center_detailed(cross, roi_size=(400, 20), **CROSS_KW)
    assert result.center is not None
    assert all(t.roi.shape == (20, 400) for t in result.rois)  # length x in search orientation
    assert result.center == pytest.approx((872.3, 544.3), abs=0.6)


@pytest.mark.unit
@pytest.mark.parametrize("length", [0, -5, 653, 1018, "Auto", "big", True, 400.0, None])
def test_invalid_roi_length_raises(cross, length) -> None:
    with pytest.raises(ValueError, match="roi_size"):
        find_cross_center_detailed(cross, roi_size=(length, 20), **CROSS_KW)


# --- find_center_pixel ----------------------------------------------------------------------


def _strip(center: float, width: int = 41) -> np.ndarray:
    xs = np.arange(width, dtype=np.float64)
    profile = 30.0 + 180.0 * np.exp(-0.5 * ((xs - center) / 2.0) ** 2)
    return np.tile(profile, (10, 1))


@pytest.mark.unit
def test_find_center_pixel_argmax() -> None:
    pos, q = find_center_pixel(_strip(17.3), (20.0, 5.0), 15, 0.5, search_method="argmax")
    assert pos == 17.0
    assert math.isnan(q)


@pytest.mark.unit
def test_find_center_pixel_moments() -> None:
    pos, q = find_center_pixel(_strip(20.4), (20.0, 5.0), 15, 0.5, search_method="moments")
    assert pos == pytest.approx(20.4, abs=0.05)
    assert math.isnan(q)


@pytest.mark.unit
def test_find_center_pixel_sdrm_returns_quality_ratio() -> None:
    pos, q = find_center_pixel(_strip(22.0), (20.0, 5.0), 15, 0.5, search_method="sdrm")
    assert pos == 22.0
    assert math.isfinite(q) and q <= 0.5


# --- MeasurementParameters ------------------------------------------------------------------


@pytest.mark.unit
def test_roi_size_x_defaults_to_auto() -> None:
    assert _load().roi_size_x == "auto"
    assert _load('roi_size_x = "auto"\n').roi_size_x == "auto"


@pytest.mark.unit
@pytest.mark.parametrize("value", [1, 500, 652])
def test_roi_size_x_int_loads(value) -> None:
    assert _load(f"roi_size_x = {value}\n").roi_size_x == value


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-1", '"Auto"', '"500"', "500.0", "true"])
def test_roi_size_x_invalid_raises(value) -> None:
    with pytest.raises(ValueError, match="roi_size_x must be an int > 0 or 'auto'"):
        _load(f"roi_size_x = {value}\n")


@pytest.mark.unit
def test_roi_size_x_larger_than_crop_raises() -> None:
    with pytest.raises(ValueError, match="roi_size_x .653. must not be larger than the crop"):
        _load("roi_size_x = 653\n")


@pytest.mark.unit
def test_repo_config_uses_auto_roi_length() -> None:
    lib = load_library(PROJECT_ROOT / "configs")
    assert all(mp.roi_size_x == "auto" for mp in lib["measurement_parameters_by_id"].values())


@pytest.mark.unit
def test_nominal_slant_default_and_toml() -> None:
    assert _load().nominal_slant == 6.0
    assert _load("nominal_slant = -2.5\n").nominal_slant == -2.5
    assert _load("nominal_slant = 7\n").nominal_slant == 7.0


@pytest.mark.unit
@pytest.mark.parametrize("value", ["nan", "inf", '"6"', "true"])
def test_nominal_slant_invalid_raises(value) -> None:
    with pytest.raises(ValueError, match="nominal_slant must be a finite number"):
        _load(f"nominal_slant = {value}\n")


@pytest.mark.unit
def test_roi_length_and_nominal_slant_round_trip_through_library_crud(tmp_path) -> None:
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    config_dir = tmp_path / "configs"
    for i, mp in enumerate([_load(), _load("roi_size_x = 400\nnominal_slant = 5.5\n")]):
        mp = replace(mp, id=f"99{i}")
        add_measurement_parameters(config_dir, mp)
        assert load_library(config_dir)["measurement_parameters_by_id"][mp.id] == mp


# --- device validation with the pre-search on -----------------------------------------------


def _lib_with_active(config_dir: Path, mp: MeasurementParameters) -> dict:
    device = load_main(config_dir)
    lib = load_library(config_dir)
    lib["measurement_parameters_by_id"][device.measurement_parameters_id] = replace(
        mp, id=device.measurement_parameters_id
    )
    return lib


@pytest.mark.unit
def test_explicit_roi_length_longer_than_presearch_minimum_is_rejected() -> None:
    config_dir = PROJECT_ROOT / "configs"
    device = load_main(config_dir)
    on = PresearchParameters(presearch_mode="every_image", min_crop_size_px=240)
    for mp in (_load("roi_size_x = 240\n"), _load()):  # at the minimum, and "auto": accepted
        validate_device_references(device, _lib_with_active(config_dir, replace(mp, presearch=on)))
    # pre-search off: only the configured crop limits the length
    validate_device_references(device, _lib_with_active(config_dir, _load("roi_size_x = 500\n")))
    with pytest.raises(ConfigError, match="roi_size_x .241. must not be larger than presearch.min_crop_size_px"):
        validate_device_references(
            device, _lib_with_active(config_dir, replace(_load("roi_size_x = 241\n"), presearch=on))
        )
