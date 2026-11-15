"""
Evaluate the pre-search (R4) on a folder of images and print Markdown tables:

1. Pre-search alone: success, failure reason, method, recommended center/size, warnings.
2. Measurement with the configured crop (presearch_mode "off") vs. the pre-search crop
   ("every_image"): success counts, images that change status, and the position difference
   for images that succeed both ways.
3. Images that must not produce a center (default: Blur 1, Blur 2, Off 3).

The configuration files are not changed; the pre-search mode is set in memory.

Usage (from the project root, with .venv active):
    python3 scripts/evaluate_presearch.py
    python3 scripts/evaluate_presearch.py --images tests/assets/images --downsample 4
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from dataclasses import replace
from pathlib import Path

import cv2

from autocollimator.app import AutocollimatorApp

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MUST_FAIL = ("Blur 1.jpg", "Blur 2.jpg", "Off 3.jpg")
DIFF_NOTE_PX = 0.5


def set_presearch(app: AutocollimatorApp, **fields) -> None:
    device = app.get_current_config()
    by_id = app.get_library()["measurement_parameters_by_id"]
    mp = app.get_measurement_parameters()
    by_id[device.measurement_parameters_id] = replace(mp, presearch=replace(mp.presearch, **fields))


def cell(v) -> str:
    return "" if v is None else str(v).replace("|", "\\|")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("tests/assets/images"))
    ap.add_argument("--config", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/evaluation_presearch"))
    ap.add_argument("--downsample", type=int, help="presearch_downsample override")
    args = ap.parse_args()
    logging.basicConfig(level=logging.CRITICAL)

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    with AutocollimatorApp(config_dir=args.config, output_dir=args.out) as app:
        if args.downsample is not None:
            set_presearch(app, presearch_downsample=args.downsample)
        mps = app.get_measurement_parameters()
        for path in images:
            img = cv2.imread(str(path))
            row = {"file": path.name, "pre": app.run_presearch(img)}
            set_presearch(app, presearch_mode="off")
            row["off"] = app.run_center_finding_on_image(img)[0]
            set_presearch(app, presearch_mode="every_image")
            row["every"] = app.run_center_finding_on_image(img)[0]
            rows.append(row)

    print(f"Configured crop: center ({mps.crop_center_x}, {mps.crop_center_y}), "
          f"size {mps.crop_size_x}x{mps.crop_size_y}; pre-search settings: {mps.presearch}\n")

    print("## 1. Pre-search alone\n")
    print("| Image | Success | Method | Position | Rec. center | Rec. size | Shifted/shrunk | Failure reason | Warnings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pr = r["pre"]
        pos = "" if pr.position_x is None else f"({pr.position_x:.1f}, {pr.position_y:.1f})"
        center = "" if pr.recommended_center_x is None else f"({pr.recommended_center_x}, {pr.recommended_center_y})"
        adj = "/".join(s for s, on in (("shifted", pr.region_shifted), ("shrunk", pr.region_shrunk)) if on)
        print(f"| {r['file']} | {'yes' if pr.success else '**no**'} | {cell(pr.method_used)} | {pos} | {center} "
              f"| {cell(pr.recommended_size)} | {adj} | {cell(pr.failure_reason)} | {cell('; '.join(pr.warnings))} |")
    n_pre = sum(r["pre"].success for r in rows)
    n_proj = sum(r["pre"].success and r["pre"].method_used == "projection" for r in rows)
    print(f"\nPre-search success: {n_pre}/{len(rows)} ({n_proj} by the projection fallback)\n")

    print("## 2. Measurement: configured crop (off) vs. pre-search crop (every_image)\n")
    n_off = sum(r["off"].measured_values.success for r in rows)
    n_every = sum(r["every"].measured_values.success for r in rows)
    print(f"Success: off {n_off}/{len(rows)}, every_image {n_every}/{len(rows)}\n")
    print("### Images that change status\n")
    print("| Image | off | every_image | every_image crop (source) | every_image message |")
    print("|---|---|---|---|---|")
    for r in rows:
        a, b = r["off"].measured_values, r["every"].measured_values
        if a.success != b.success:
            ps = r["every"].presearch
            crop = (f"({ps.crop_center_x}, {ps.crop_center_y}) {ps.crop_size_x}x{ps.crop_size_y} ({ps.crop_source})"
                    if ps.crop_source else "none")
            print(f"| {r['file']} | {'ok' if a.success else 'fail'} | {'ok' if b.success else 'fail'} | {crop} "
                  f"| {cell(b.message)} |")

    print("\n### Position difference (every_image - off), images that succeed both ways\n")
    print("| Image | off (x, y) | every_image (x, y) | dx | dy | distance | crop off / every_image |")
    print("|---|---|---|---|---|---|---|")
    dists = []
    for r in rows:
        a, b = r["off"].measured_values, r["every"].measured_values
        if not (a.success and b.success):
            continue
        dx, dy = b.center_position_x - a.center_position_x, b.center_position_y - a.center_position_y
        d = math.hypot(dx, dy)
        dists.append(d)
        ps = r["every"].presearch
        flag = " **>0.5**" if d > DIFF_NOTE_PX else ""
        print(f"| {r['file']} | ({a.center_position_x:.2f}, {a.center_position_y:.2f}) "
              f"| ({b.center_position_x:.2f}, {b.center_position_y:.2f}) | {dx:+.3f} | {dy:+.3f} | {d:.3f}{flag} "
              f"| {mps.crop_size_x}x{mps.crop_size_y} @ ({mps.crop_center_x}, {mps.crop_center_y}) / "
              f"{ps.crop_size_x}x{ps.crop_size_y} @ ({ps.crop_center_x}, {ps.crop_center_y}) |")
    if dists:
        print(f"\nBoth succeed: {len(dists)} images; distance median {statistics.median(dists):.3f} px, "
              f"max {max(dists):.3f} px; {sum(d > DIFF_NOTE_PX for d in dists)} above {DIFF_NOTE_PX} px\n")

    print("## 3. Images that must not produce a center\n")
    print("| Image | Pre-search | off | every_image | every_image message |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["file"] in MUST_FAIL:
            a, b = r["off"].measured_values, r["every"].measured_values
            print(f"| {r['file']} | {'success' if r['pre'].success else 'fail'} | {'success' if a.success else 'fail'} "
                  f"| {'success' if b.success else 'fail'} | {cell(b.message)} |")
    ok = all(not (r["off"].measured_values.success or r["every"].measured_values.success)
             for r in rows if r["file"] in MUST_FAIL)
    print(f"\nAll must-fail images fail in both modes: {'yes' if ok else '**NO**'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
