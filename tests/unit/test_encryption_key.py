"""R6: the device encryption key must be set explicitly and never leak into output."""
from pathlib import Path

import pytest

from autocollimator.config.models import Device
from autocollimator.config.store import ConfigError, load_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
KEY_VAR = "DEVICE_ENCRYPTION_KEY"
SECRET = "unit-test-secret-4f2a"


def _config_with_key(tmp_path: Path, key_line: str) -> Path:
    """Copy configs/main.toml into tmp_path with a different encryption_key line."""
    lines = (CONFIG_DIR / "main.toml").read_text(encoding="utf-8").splitlines()
    replaced = [key_line if ln.strip().startswith("encryption_key") else ln for ln in lines]
    assert replaced != lines, "main.toml has no encryption_key line"
    (tmp_path / "main.toml").write_text("\n".join(replaced) + "\n", encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_load_main_resolves_key_from_environment(monkeypatch):
    monkeypatch.setenv(KEY_VAR, SECRET)
    assert load_main(CONFIG_DIR).encryption_key == SECRET


@pytest.mark.unit
def test_load_main_unset_variable_raises(monkeypatch):
    monkeypatch.delenv(KEY_VAR, raising=False)
    with pytest.raises(ConfigError, match=f"set the environment variable {KEY_VAR}"):
        load_main(CONFIG_DIR)


@pytest.mark.unit
def test_load_main_empty_variable_raises(monkeypatch):
    monkeypatch.setenv(KEY_VAR, "")
    with pytest.raises(ConfigError, match=f"set the environment variable {KEY_VAR}"):
        load_main(CONFIG_DIR)


@pytest.mark.unit
def test_load_main_other_unexpanded_variable_is_named(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOCOLLIMATOR_TEST_UNSET_KEY", raising=False)
    config_dir = _config_with_key(tmp_path, 'encryption_key = "${AUTOCOLLIMATOR_TEST_UNSET_KEY}"')
    with pytest.raises(ConfigError, match="set the environment variable AUTOCOLLIMATOR_TEST_UNSET_KEY"):
        load_main(config_dir)


@pytest.mark.unit
def test_load_main_literal_empty_key_raises(tmp_path):
    config_dir = _config_with_key(tmp_path, 'encryption_key = ""')
    with pytest.raises(ConfigError, match=f"set the environment variable {KEY_VAR}"):
        load_main(config_dir)


@pytest.mark.unit
def test_device_repr_hides_key(monkeypatch):
    monkeypatch.setenv(KEY_VAR, SECRET)
    text = repr(load_main(CONFIG_DIR))
    assert SECRET not in text
    assert "encryption_key" not in text


@pytest.mark.unit
def test_device_from_dict_without_key_gives_empty_string(monkeypatch):
    # Saved measurement records omit the key; they must still load.
    monkeypatch.setenv(KEY_VAR, SECRET)
    from dataclasses import asdict

    d = asdict(load_main(CONFIG_DIR))
    del d["encryption_key"]
    assert Device.from_dict(d).encryption_key == ""
