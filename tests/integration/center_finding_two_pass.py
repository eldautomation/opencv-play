"""Two-pass center finding and manifest loading shared by test_center_finding.py and
scripts/regenerate_expected.py (R8, R9).

Pass 1 is the default-config workflow. Pass 2 re-runs find_cross_center on a square crop
centered on the pass-1 position, with the same measurement parameters and q_limit.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import find_cross_center

LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECT_VALUES = ("success", "failure")
MANIFEST_KEYS = {"file", "expect", "expected_x", "expected_y", "notes"}


@dataclass(frozen=True)
class TwoPassResult:
    success: bool
    position: tuple[float, float] | None  # refined (pass-2) position; None on failure
    pass1_position: tuple[float, float] | None
    message: str


def measure_two_pass(
    app: AutocollimatorApp,
    image: np.ndarray,
    *,
    debug: bool = False,
    debug_prefix: str | Path | None = None,
) -> TwoPassResult:
    """Run the default workflow, then refine around its center. Success requires both passes."""
    output, _overlay = app.run_center_finding_on_image(image, debug=debug, debug_prefix=debug_prefix)
    mv = output.measured_values
    if not mv.success:
        return TwoPassResult(False, None, None, f"pass 1: {mv.message}")
    pass1 = (mv.center_position_x, mv.center_position_y)

    mps = app.get_measurement_parameters()
    qls = app.get_quality_limits()
    size = int(min(mps.crop_size_x, mps.crop_size_y))
    position, _angles, _overlay2, q_ratio_list = find_cross_center(
        image=image,
        crop_center=(int(pass1[0]), int(pass1[1])),
        crop_size=(size, size),
        roi_size=(int(mps.roi_size_x), int(mps.roi_size_y)),
        q_limit=qls.rss_ratio,
        subpixel=mps.subpixel,
        subpixel_neighbors=mps.subpixel_neighbors,
        profile_mode=mps.profile_mode,
        oversample_step=mps.oversample_step,
        fine_window=mps.fine_window,
        fine_window_mode=mps.fine_window_mode,
        refine_search=mps.refine_search,
        max_refine_shift=mps.max_refine_shift,
        gray_mode=mps.gray_mode,
        debug=debug,
        debug_prefix=f"{debug_prefix}_refined" if debug_prefix is not None else "dbg",
    )
    if position is None:
        ratios = ", ".join(f"{q:.3f}" for q in q_ratio_list)
        return TwoPassResult(
            False, None, pass1, f"pass 2: quality check failed (rss_ratio limit {qls.rss_ratio}; ratios {ratios})"
        )
    return TwoPassResult(True, (float(position[0]), float(position[1])), pass1, "Center found")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_manifest(path: Path) -> list[dict]:
    """Load and validate a manifest: a YAML list of {file, expect, [expected_x, expected_y], [notes]}."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: top level must be a list of entries")
    seen: set[str] = set()
    for i, e in enumerate(entries):
        where = f"{path.name} entry {i}"
        if not isinstance(e, dict):
            raise ValueError(f"{where}: must be a mapping")
        unknown = set(e) - MANIFEST_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        f = e.get("file")
        if not isinstance(f, str) or not f or "/" in f or "\\" in f:
            raise ValueError(f"{where}: 'file' must be a bare file name, got {f!r}")
        if f in seen:
            raise ValueError(f"{where}: duplicate file {f!r}")
        seen.add(f)
        if e.get("expect") not in EXPECT_VALUES:
            raise ValueError(f"{where} ({f}): 'expect' must be one of {EXPECT_VALUES}, got {e.get('expect')!r}")
        has_x, has_y = "expected_x" in e, "expected_y" in e
        if has_x != has_y:
            raise ValueError(f"{where} ({f}): give both expected_x and expected_y or neither")
        if has_x:
            if e["expect"] != "success":
                raise ValueError(f"{where} ({f}): expected_x/expected_y only apply to expect: success")
            if not (_is_number(e["expected_x"]) and _is_number(e["expected_y"])):
                raise ValueError(f"{where} ({f}): expected_x/expected_y must be numbers")
        if "notes" in e and not isinstance(e["notes"], str):
            raise ValueError(f"{where} ({f}): 'notes' must be a string")
    return entries


def image_files(image_dir: Path) -> list[str]:
    return sorted(p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
