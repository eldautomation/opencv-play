"""
Propose a manifest for tests/integration/test_center_finding.py (R8, R9).

Runs the default configuration (two-pass center finding, as in the integration test) on every
image in tests/assets/images and writes a proposed manifest. It never writes
tests/assets/images/manifest.yaml: review the proposal, then copy it there yourself.

Usage (from the project root, with .venv active):
    python3 scripts/regenerate_expected.py
    python3 scripts/regenerate_expected.py --out outputs/manifest_proposed.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "tests" / "assets" / "images"
MANIFEST_PATH = IMAGE_DIR / "manifest.yaml"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "manifest_proposed.yaml"

sys.path.insert(0, str(PROJECT_ROOT / "tests" / "integration"))

from autocollimator.app import AutocollimatorApp  # noqa: E402
from center_finding_two_pass import image_files, measure_two_pass  # noqa: E402

LOGGER = logging.getLogger("regenerate_expected")

HEADER = """\
# PROPOSED manifest written by scripts/regenerate_expected.py with the default configuration.
# Review every entry (expect, positions, notes) before copying it to tests/assets/images/manifest.yaml.
# expected_x / expected_y are the refined (pass-2) center in px; the test allows +/- 0.2 px.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output file (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = args.out.resolve()
    if out == MANIFEST_PATH.resolve():
        LOGGER.error("Refusing to overwrite %s; write the proposal elsewhere and review it.", MANIFEST_PATH)
        return 2

    entries: list[dict] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp, AutocollimatorApp(
        config_dir=PROJECT_ROOT / "configs", output_dir=Path(tmp), input_dir=IMAGE_DIR
    ) as app:
        for name in image_files(IMAGE_DIR):
            img = cv2.imread(str(IMAGE_DIR / name))
            if img is None:
                LOGGER.error("%s: cannot read image", name)
                errors.append(name)
                continue
            try:
                result = measure_two_pass(app, img)
            except Exception:
                LOGGER.exception("%s: center finding raised", name)
                errors.append(name)
                continue
            if result.success:
                x, y = result.position
                entries.append({"file": name, "expect": "success", "expected_x": x, "expected_y": y})
                LOGGER.info("%s: success (%.3f, %.3f)", name, x, y)
            else:
                entries.append({"file": name, "expect": "failure", "notes": result.message})
                LOGGER.info("%s: failure (%s)", name, result.message)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True), encoding="utf-8")
    n_ok = sum(e["expect"] == "success" for e in entries)
    LOGGER.info("Wrote %s: %d success, %d failure, %d not written (errors)",
                out, n_ok, len(entries) - n_ok, len(errors))
    if errors:
        LOGGER.warning("Not in the proposal because of errors: %s", ", ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
