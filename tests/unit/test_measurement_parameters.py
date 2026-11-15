import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from autocollimator.config.models import MeasurementParameters, PresearchParameters
from autocollimator.config.store import add_measurement_parameters, load_library

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_TOML = """
[[measurement_parameters]]
id = "901"
name = "Default Parameters"
crop_center_x = 872
crop_center_y = 544
crop_size_x = 1018
crop_size_y = 652
roi_size_x = 500
roi_size_y = 20
"""


def _load(extra: str = "") -> MeasurementParameters:
    return MeasurementParameters.from_dict(tomllib.loads(BASE_TOML + extra)["measurement_parameters"][0])


@pytest.mark.unit
def test_measurement_parameters_without_subpixel_fields_get_defaults():
    mp = _load()
    assert mp.subpixel is False
    assert mp.subpixel_neighbors == 2


@pytest.mark.unit
def test_repo_config_library_loads_with_subpixel_defaults():
    lib = load_library(PROJECT_ROOT / "configs")
    for mp in lib["measurement_parameters_by_id"].values():
        assert mp.subpixel is False
        assert mp.subpixel_neighbors == 2


@pytest.mark.unit
def test_measurement_parameters_subpixel_fields_from_toml():
    mp = _load("subpixel = true\nsubpixel_neighbors = 3\n")
    assert mp.subpixel is True
    assert mp.subpixel_neighbors == 3


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "4", "5", "2.0", "true", '"2"'])
def test_measurement_parameters_invalid_neighbors_raise(value):
    with pytest.raises(ValueError, match="subpixel_neighbors must be an int in 1..3"):
        _load(f"subpixel_neighbors = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_invalid_subpixel_type_raises():
    with pytest.raises(TypeError, match="subpixel must be bool"):
        _load('subpixel = "yes"\n')


@pytest.mark.unit
def test_measurement_parameters_override_is_validated():
    with pytest.raises(ValueError, match="subpixel_neighbors must be an int in 1..3"):
        replace(_load(), subpixel_neighbors=0)


@pytest.mark.unit
def test_measurement_parameters_without_profile_fields_get_defaults():
    mp = _load()
    assert mp.profile_mode == "average"
    assert mp.oversample_step == 0.1
    assert mp.fine_window == 40


@pytest.mark.unit
def test_repo_config_library_loads_with_profile_defaults():
    lib = load_library(PROJECT_ROOT / "configs")
    for mp in lib["measurement_parameters_by_id"].values():
        assert mp.profile_mode == "average"
        assert mp.oversample_step == 0.1
        assert mp.fine_window == 40


@pytest.mark.unit
def test_measurement_parameters_profile_fields_from_toml():
    mp = _load('profile_mode = "deskew"\noversample_step = 0.05\nfine_window = 25\nsubpixel = true\n')
    assert mp.profile_mode == "deskew"
    assert mp.oversample_step == 0.05
    assert mp.fine_window == 25
    assert mp.subpixel is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0.02", "0.5"])
def test_measurement_parameters_oversample_step_range_is_inclusive(value):
    assert _load(f"oversample_step = {value}\n").oversample_step == float(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", ['"Deskew"', '"tilted"', '""', "1", "true"])
def test_measurement_parameters_invalid_profile_mode_raises(value):
    with pytest.raises(ValueError, match="profile_mode must be one of"):
        _load(f"profile_mode = {value}\n")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0.01", "0.6", "0", "true", '"0.1"'])
def test_measurement_parameters_invalid_oversample_step_raises(value):
    with pytest.raises(ValueError, match="oversample_step must be a number in 0.02..0.5"):
        _load(f"oversample_step = {value}\n")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["1", "0", "-5", "40.0", "true", '"40"'])
def test_measurement_parameters_invalid_fine_window_raises(value):
    with pytest.raises(ValueError, match="fine_window must be an int >= 2"):
        _load(f"fine_window = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_profile_override_is_validated():
    with pytest.raises(ValueError, match="profile_mode must be one of"):
        replace(_load(), profile_mode="tilted")


@pytest.mark.unit
def test_measurement_parameters_refine_search_defaults_and_loads():
    assert _load().refine_search == 1.5
    for mp in load_library(PROJECT_ROOT / "configs")["measurement_parameters_by_id"].values():
        assert mp.refine_search == 1.5
    assert _load("refine_search = 2.5\n").refine_search == 2.5
    assert _load("refine_search = 3\n").refine_search == 3.0


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0.4", "3.1", "0", "true", '"1.5"'])
def test_measurement_parameters_invalid_refine_search_raises(value):
    with pytest.raises(ValueError, match="refine_search must be a number in 0.5..3.0"):
        _load(f"refine_search = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_refine_search_must_be_below_fine_window():
    assert _load("refine_search = 1.5\nfine_window = 2\n").fine_window == 2
    with pytest.raises(ValueError, match="refine_search .* must be smaller than"):
        _load("refine_search = 2.5\nfine_window = 2\n")


@pytest.mark.unit
def test_measurement_parameters_max_refine_shift_defaults_and_loads():
    assert _load().max_refine_shift == 0.75
    for mp in load_library(PROJECT_ROOT / "configs")["measurement_parameters_by_id"].values():
        assert mp.max_refine_shift == 0.75
    assert _load("max_refine_shift = 0.1\n").max_refine_shift == 0.1
    assert _load("max_refine_shift = 3\n").max_refine_shift == 3.0


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0.05", "3.5", "0", "true", '"0.75"'])
def test_measurement_parameters_invalid_max_refine_shift_raises(value):
    with pytest.raises(ValueError, match="max_refine_shift must be a number in 0.1..3.0"):
        _load(f"max_refine_shift = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_gray_mode_defaults_and_loads():
    assert _load().gray_mode == "luma"
    for mp in load_library(PROJECT_ROOT / "configs")["measurement_parameters_by_id"].values():
        assert mp.gray_mode == "luma"
    assert _load('gray_mode = "green"\n').gray_mode == "green"
    assert _load('gray_mode = "max"\n').gray_mode == "max"


@pytest.mark.unit
@pytest.mark.parametrize("value", ['"Luma"', '"gray"', '""', "1", "true"])
def test_measurement_parameters_invalid_gray_mode_raises(value):
    with pytest.raises(ValueError, match="gray_mode must be one of"):
        _load(f"gray_mode = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_gray_mode_override_is_validated():
    with pytest.raises(ValueError, match="gray_mode must be one of"):
        replace(_load(), gray_mode="red")


@pytest.mark.unit
def test_measurement_parameters_fine_window_mode_defaults_and_loads():
    assert _load().fine_window_mode == "fixed"
    for mp in load_library(PROJECT_ROOT / "configs")["measurement_parameters_by_id"].values():
        assert mp.fine_window_mode == "fixed"
    assert _load('fine_window_mode = "adaptive"\n').fine_window_mode == "adaptive"
    assert _load('fine_window_mode = "fixed"\n').fine_window_mode == "fixed"


@pytest.mark.unit
@pytest.mark.parametrize("value", ['"Adaptive"', '"auto"', '""', "1", "true"])
def test_measurement_parameters_invalid_fine_window_mode_raises(value):
    with pytest.raises(ValueError, match="fine_window_mode must be one of"):
        _load(f"fine_window_mode = {value}\n")


@pytest.mark.unit
def test_measurement_parameters_fine_window_mode_override_is_validated():
    with pytest.raises(ValueError, match="fine_window_mode must be one of"):
        replace(_load(), fine_window_mode="auto")


# ---- Pre-search settings (R4) ----

PRESEARCH_DEFAULTS = dict(
    presearch_mode="off",
    presearch_failure_action="use_configured_crop",
    presearch_downsample=2,
    presearch_crop_fraction=0.5,
    min_crop_size_px=240,
    presearch_q_limit=0.7,
    presearch_agreement_px=5.0,
    presearch_ambiguity_ratio=0.8,
)


@pytest.mark.unit
def test_presearch_settings_default_to_off_without_table():
    mp = _load()
    assert mp.presearch == PresearchParameters()
    assert asdict(mp.presearch) == PRESEARCH_DEFAULTS


@pytest.mark.unit
def test_repo_config_library_loads_with_presearch_defaults():
    lib = load_library(PROJECT_ROOT / "configs")
    for mp in lib["measurement_parameters"]:
        assert mp.presearch == PresearchParameters()


@pytest.mark.unit
def test_presearch_settings_from_toml():
    mp = _load(
        """
[measurement_parameters.presearch]
presearch_mode = "track"
presearch_failure_action = "fail"
presearch_downsample = 4
presearch_crop_fraction = 0.3
min_crop_size_px = 300
presearch_q_limit = 0.6
presearch_agreement_px = 3
presearch_ambiguity_ratio = 0.9
"""
    )
    assert asdict(mp.presearch) == dict(
        presearch_mode="track", presearch_failure_action="fail", presearch_downsample=4,
        presearch_crop_fraction=0.3, min_crop_size_px=300, presearch_q_limit=0.6,
        presearch_agreement_px=3, presearch_ambiguity_ratio=0.9,
    )
    # Partial tables keep the other defaults.
    mp = _load('\n[measurement_parameters.presearch]\npresearch_mode = "every_image"\n')
    assert mp.presearch == PresearchParameters(presearch_mode="every_image")


@pytest.mark.unit
@pytest.mark.parametrize(
    "key, value",
    [
        ("presearch_mode", "always"),
        ("presearch_failure_action", "ignore"),
        ("presearch_downsample", 0),
        ("presearch_downsample", 2.0),
        ("presearch_downsample", True),
        ("presearch_crop_fraction", 0.05),
        ("presearch_crop_fraction", 1.2),
        ("min_crop_size_px", 0),
        ("min_crop_size_px", 240.0),
        ("presearch_q_limit", 0.0),
        ("presearch_q_limit", 1.0),
        ("presearch_agreement_px", 0.0),
        ("presearch_agreement_px", "5"),
        ("presearch_ambiguity_ratio", 0.0),
        ("presearch_ambiguity_ratio", 1.5),
    ],
)
def test_presearch_invalid_values_raise(key, value):
    with pytest.raises(ValueError, match=key):
        PresearchParameters(**{key: value})


@pytest.mark.unit
def test_presearch_table_rejects_unknown_keys_and_non_tables():
    with pytest.raises(ValueError, match="unknown key"):
        _load('\n[measurement_parameters.presearch]\npresearch_modus = "off"\n')
    with pytest.raises(TypeError, match="presearch must be a table"):
        MeasurementParameters.from_dict({**tomllib.loads(BASE_TOML)["measurement_parameters"][0], "presearch": "off"})


@pytest.mark.unit
def test_presearch_settings_round_trip_through_library_crud(tmp_path):
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    config_dir = tmp_path / "configs"
    mp = replace(
        _load(), id="999", presearch=PresearchParameters(presearch_mode="every_image", presearch_downsample=3)
    )
    add_measurement_parameters(config_dir, mp)
    assert load_library(config_dir)["measurement_parameters_by_id"]["999"] == mp
