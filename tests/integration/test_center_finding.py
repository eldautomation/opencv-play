from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import pytest

# Adjust these imports to match your actual module locations
from autocollimator.target_utils import find_cross_center
from autocollimator.target_utils import clear_folder, pct_list_to_int_list  # or wherever these live

LOGGER = logging.getLogger(__name__)


@pytest.mark.integration
def test_center_finding_generates_debug_outputs(tmp_path: Path) -> None:
    """
    Integration test version of run_demo().

    - Loads a fixed set of test images from tests/assets/images/
    - Runs the center-finding pipeline twice (initial + refined)
    - Writes debug images into an output folder so results can be inspected
    - Produces a summary file in the output folder

    To keep outputs after the test, run with:
        pytest -m integration --keep-output
    or set:
        KEEP_OUTPUT=1
    """
    logging.basicConfig(level=logging.INFO)

    # --- Inputs (repo-relative) ---
    project_root = Path(__file__).resolve().parents[2]
    prefix_in = project_root / "tests" / "assets" / "images"

    images = [
        "blob-05.jpg",
        "blob-06.jpg",
        "blob-08.jpg",
        "blob-09.jpg",
        "blob-11.jpg",
        "blob-12.jpg",
    ]

    # Approximate percent centers and crop sizes (original values)
    cx_list = [0.37, 0.6, 0.27, 0.6, 0.3, 0.6, 0.5]
    cy_list = [0.5, 0.7, 0.35, 0.4, 0.4, 0.4, 0.4]
    x_pct_list = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6]
    y_pct_list = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

    # --- Output folder ---
    # By default use pytest temp dir (clean). Optionally persist to outputs/ for inspection.
    keep_output = (
        os.getenv("KEEP_OUTPUT", "0") == "1"
        or "--keep-output" in os.getenv("PYTEST_ADDOPTS", "")
        or getattr(pytest, "keep_output_flag", False)
    )

    if keep_output:
        prefix_out = project_root / "outputs" / "integration" / "center_finding"
        clear_folder(prefix_out, create=True)
    else:
        prefix_out = tmp_path / "center_finding"
        prefix_out.mkdir(parents=True, exist_ok=True)

    # --- Validate inputs exist ---
    missing = [name for name in images if not (prefix_in / name).is_file()]
    if missing:
        pytest.skip(f"Missing integration test images in {prefix_in}: {missing}")

    # --- Determine image size using first image ---
    first = cv2.imread(str(prefix_in / images[0]))
    if first is None:
        pytest.fail(f"Cannot read image: {prefix_in / images[0]}")
    height, width = first.shape[:2]

    cx_px = pct_list_to_int_list(cx_list, width)
    cy_px = pct_list_to_int_list(cy_list, height)
    x_crop_px = pct_list_to_int_list(x_pct_list, width)
    y_crop_px = pct_list_to_int_list(y_pct_list, height)

    summary_lines: list[str] = ["Summary:"]

    for i, fname in enumerate(images, start=1):
        img_path = prefix_in / fname
        img = cv2.imread(str(img_path))
        if img is None:
            LOGGER.warning("Skipping unreadable image: %s", img_path)
            continue

        debug_prefix = str(prefix_out / f"debug-{i}")

        position, angles = find_cross_center(
            image=img,
            crop_center=(cx_px[i - 1], cy_px[i - 1]),
            crop_size=(x_crop_px[i - 1], y_crop_px[i - 1]),
            roi_size=(500, 20),
            debug=True,
            debug_prefix=debug_prefix,
        )
        LOGGER.info("Initial %s position=%s angles=%s", fname, position, angles)

        if position is None:
            summary_lines.append(f"{fname}\tFAILED\tprefix:{debug_prefix}")
            continue

        crop_pixels = min(x_crop_px[i - 1], y_crop_px[i - 1])

        position2, angles2 = find_cross_center(
            image=img,
            crop_center=(int(position[0]), int(position[1])),
            crop_size=(crop_pixels, crop_pixels),
            roi_size=(500, 20),
            debug=True,
            debug_prefix=f"{debug_prefix}_refined",
        )
        LOGGER.info("Refined %s position=%s angles=%s", fname, position2, angles2)

        x0, y0 = float(position[0]), float(position[1])
        x1, y1 = (float(position2[0]), float(position2[1])) if position2 is not None else (float("nan"), float("nan"))

        summary_lines.append(
            f"{fname}\tprefix:{debug_prefix}\tx0:{x0:.3f}\ty0:{y0:.3f}\tx1:{x1:.3f}\ty1:{y1:.3f}"
        )

    # Write summary for inspection
    summary_path = prefix_out / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Minimal assertions: ensure the pipeline produced some output artifacts
    # (You can tighten this by checking expected filenames/patterns.)
    produced_files = list(prefix_out.glob("**/*"))
    assert len(produced_files) > 0, f"No artifacts produced in {prefix_out}"

    expected_summary = project_root / "tests" / "assets" / "expected" / "center_finding_summary.txt"
    if not expected_summary.exists():
        # First-run bootstrap: write expected file so you can review/commit it.
        expected_summary.parent.mkdir(parents=True, exist_ok=True)
        expected_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        pytest.fail(
            f"Expected summary did not exist. Wrote one to {expected_summary}. "
            "Review it and commit it, then re-run the test."
        )

    actual = summary_path.read_text(encoding="utf-8")
    expected = expected_summary.read_text(encoding="utf-8")

    assert actual == expected, (
        "summary.txt differs from expected.  Confirm output is being generated (--keep-output).\n"
        f"Expected: {expected_summary}\n"
        f"Actual:   {summary_path}\n"
        "Run with: pytest -m integration --keep-output\n"
        "Then diff the two files.")