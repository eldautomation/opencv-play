from dataclasses import replace
from pathlib import Path

import pytest
import tomli

from autocollimator.config.models import MeasurementParameters
from autocollimator.config.store import load_library

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
    return MeasurementParameters.from_dict(tomli.loads(BASE_TOML + extra)["measurement_parameters"][0])


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
