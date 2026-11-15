"""R14: importing autocollimator must not import matplotlib (only the debug plots need it)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
MODULES = (
    "autocollimator",
    "autocollimator.app",
    "autocollimator.workflow",
    "autocollimator.target_utils",
    "autocollimator.presearch",
    "autocollimator.instrument",
    "autocollimator.config.models",
    "autocollimator.config.store",
)


@pytest.mark.unit
def test_import_does_not_load_matplotlib() -> None:
    # A fresh interpreter: the test session itself may already have imported matplotlib.
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in MODULES)
        + "loaded = sorted(m for m in sys.modules if m == 'matplotlib' or m.startswith('matplotlib.'))\n"
        + "assert not loaded, loaded\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.unit
def test_debug_plot_still_works(tmp_path: Path) -> None:
    from autocollimator.target_utils import plot_xy_positions

    out = tmp_path / "plot.jpg"
    plot_xy_positions([(0.0, 1.0), (1.0, 0.5), (2.0, 2.0)], save_path=str(out))
    assert out.is_file() and out.stat().st_size > 0
