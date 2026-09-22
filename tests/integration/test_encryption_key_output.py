"""R6: the device encryption key must not appear in saved measurements, logs or script errors."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from autocollimator.app import AutocollimatorApp
from autocollimator.config.store import load_measurement_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"
KEY_VAR = "DEVICE_ENCRYPTION_KEY"
SECRET = "integration-secret-9c1e"


@pytest.mark.integration
def test_saved_measurement_and_logs_exclude_key(tmp_path: Path, monkeypatch, caplog) -> None:
    monkeypatch.setenv(KEY_VAR, SECRET)
    img = cv2.imread(str(IMAGE_DIR / "blob-09.jpg"))
    if img is None:
        pytest.fail("Cannot read blob-09.jpg")

    with caplog.at_level(logging.DEBUG):
        with AutocollimatorApp(
            config_dir=PROJECT_ROOT / "configs", output_dir=tmp_path, input_dir=IMAGE_DIR
        ) as app:
            assert app.get_current_config().encryption_key == SECRET  # key is loaded in memory
            result, overlay = app.run_center_finding_on_image(img)
            _in, _ov, yaml_path = app.save_measurement_output(result, img, overlay)

    text = yaml_path.read_text(encoding="utf-8")
    assert SECRET not in text
    assert "encryption_key" not in text
    assert SECRET not in caplog.text

    reloaded = load_measurement_output(yaml_path)
    assert reloaded.hardware.device.encryption_key == ""
    assert reloaded.measured_values == result.measured_values


def _run_without_key(args: list[str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != KEY_VAR}
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, *args], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "script_args",
    [
        ["scripts/demo_app.py"],
        ["scripts/evaluate_images.py", "--out", "{tmp}"],
    ],
    ids=["demo_app", "evaluate_images"],
)
def test_scripts_report_missing_key_clearly(script_args, tmp_path: Path) -> None:
    proc = _run_without_key([a.replace("{tmp}", str(tmp_path)) for a in script_args])
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert f"set the environment variable {KEY_VAR}" in proc.stderr, output
    assert "Traceback" not in output, output
