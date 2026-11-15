"""
Diagnose the deskew pass-2 search for one image: per ROI pass-1 vs pass-2 position, where
the fine-grid SDRM minimum lies in the search range, and a plot of the fine RSS curve.

Usage (from the project root, with .venv active):
    PYTHONPATH=src python3 scripts/diagnose_refine.py tests/assets/images/blob-13.jpg --subpixel
    PYTHONPATH=src python3 scripts/diagnose_refine.py tests/assets/images/blob-13.jpg --subpixel --refine-search 2.5

Crop, ROI and q_limit come from the config (--config). The profile mode is always deskew;
the average-mode result with the same subpixel setting is printed for comparison.
Plots are written to --out (default outputs/diagnose/).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autocollimator.app import AutocollimatorApp  # noqa: E402
from autocollimator.target_utils import (  # noqa: E402
    _deskew_profile,
    _sdrm_core,
    find_cross_center_detailed,
    sdrm_ssd_curve,
)

PLOT_RANGE_PX = 3.0


def axis_origin(trace) -> tuple[str, int]:
    """Search axis and ROI offset that turn an ROI column position into a full-image coordinate."""
    return ("x", trace.x0) if trace.name in ("top", "bottom") else ("y", trace.y0)


def fmt(v, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnose"))
    ap.add_argument("--subpixel", action="store_true")
    ap.add_argument("--subpixel-neighbors", type=int, metavar="N")
    ap.add_argument("--oversample-step", type=float, metavar="PX")
    ap.add_argument("--fine-window", type=int, metavar="PX")
    ap.add_argument("--refine-search", type=float, metavar="PX")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        return 2

    overrides: dict = {"profile_mode": "deskew"}
    if args.subpixel:
        overrides["subpixel"] = True
    for key in ("subpixel_neighbors", "oversample_step", "fine_window", "refine_search"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)

    args.out.mkdir(parents=True, exist_ok=True)
    with AutocollimatorApp(config_dir=args.config, output_dir=args.out) as app:
        mp = replace(app.get_measurement_parameters(), **overrides)  # validated by __post_init__
        q_limit = app.get_quality_limits().rss_ratio

    common = dict(
        crop_center=(mp.crop_center_x, mp.crop_center_y),
        crop_size=(mp.crop_size_x, mp.crop_size_y),
        roi_size=(mp.roi_size_x, mp.roi_size_y),
        q_limit=q_limit,
        subpixel=mp.subpixel,
        subpixel_neighbors=mp.subpixel_neighbors,
    )
    average = find_cross_center_detailed(img, **common)
    deskew = find_cross_center_detailed(
        img,
        **common,
        profile_mode="deskew",
        oversample_step=mp.oversample_step,
        fine_window=mp.fine_window,
        refine_search=mp.refine_search,
    )

    step = mp.oversample_step
    search = int(round(mp.refine_search / step))
    wide = int(round(PLOT_RANGE_PX / step))
    readout = f"subpixel n={mp.subpixel_neighbors}" if mp.subpixel else "integer"
    print(f"{args.image.name}: deskew / {readout}, oversample_step={step:g}, fine_window={mp.fine_window}, "
          f"refine_search={mp.refine_search:g} px (= +/-{search} samples), q_limit={q_limit:g}")
    for label, r in (("average", average), ("deskew ", deskew)):
        c = r.center or (None, None)
        a = r.angles or (None, None)
        print(f"  {label}: center=({fmt(c[0], 2)}, {fmt(c[1], 2)})  a0={fmt(a[0])}  a1={fmt(a[1])}")
    if average.center and deskew.center:
        print(f"  shift  : dx={deskew.center[0] - average.center[0]:+.2f}  dy={deskew.center[1] - average.center[1]:+.2f}"
              f"  da0={deskew.angles[0] - average.angles[0]:+.3f}  da1={deskew.angles[1] - average.angles[1]:+.3f}")
    if deskew.center is None:
        print("  pass 1 failed the quality check; pass 2 did not run")
        return 1

    print()
    print(f"  {'ROI':<7}{'axis':<5}{'method':<9}{'slope':>9}{'pass1':>11}{'pass2':>11}{'diff':>8}"
          f"{'min idx':>10}{'':>6}{'min +/-3px':>12}  plot")
    for trace, method in zip(deskew.rois, deskew.roi_methods):
        axis, origin = axis_origin(trace)
        p1 = origin + trace.pos1
        p2 = None if trace.pos2 is None else origin + trace.pos2
        diff = None if p2 is None else p2 - p1
        idx_txt, flag, global_txt, plot_txt = "-", "", "-", "-"

        deskewed = None if trace.slope is None else _deskew_profile(
            trace.roi, trace.pos1, trace.slope, oversample_step=step, fine_window=mp.fine_window, name=trace.name
        )
        if deskewed is not None:
            profile, _empty = deskewed
            ssd = sdrm_ssd_curve(profile, search)
            k_min = int(np.argmin(ssd)) - search
            k_core, _r1 = _sdrm_core(profile, search)
            assert k_min == k_core, (k_min, k_core)  # curve and core agree on the integer minimum
            idx_txt = f"{k_min:+d}/{search}"
            flag = "EDGE" if abs(k_min) == search else ""

            ssd_wide = sdrm_ssd_curve(profile, wide)
            offsets_px = (np.arange(2 * wide + 1) - wide) * step
            global_txt = f"{offsets_px[int(np.argmin(ssd_wide))]:+.2f}"

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axvspan(-mp.refine_search, mp.refine_search, color="0.92", label=f"search +/-{mp.refine_search:g} px")
            ax.plot(offsets_px, np.sqrt(ssd_wide), color="tab:blue", lw=1.2, label="fine RSS")
            ax.axvline(0.0, color="tab:gray", ls="--", lw=1, label="pass-1 estimate")
            if diff is not None:
                ax.axvline(trace.pos2 - trace.pos1, color="tab:red", lw=1.5,
                           label=f"chosen minimum ({diff:+.3f} px)")
            ax.set_xlabel(f"offset from pass-1 estimate along {axis} (px)")
            ax.set_ylabel("RSS")
            ax.set_title(f"{args.image.stem} {trace.name}: {method}, refine_search {mp.refine_search:g} px"
                         + (" [EDGE]" if flag else ""))
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_path = args.out / f"{args.image.stem}_rs{mp.refine_search:g}_{trace.name}.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            plot_txt = str(plot_path)

        print(f"  {trace.name:<7}{axis:<5}{method:<9}{fmt(trace.slope, 5):>9}{fmt(p1):>11}{fmt(p2):>11}"
              f"{fmt(diff):>8}{idx_txt:>10}{flag:>6}{global_txt:>12}  {plot_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
