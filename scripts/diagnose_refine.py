"""
Diagnose the deskew pass-2 search for one image: per ROI pass-1 vs pass-2 position, where
the fine-grid SDRM minimum lies in the search range, and a plot of the fine RSS curve.

Usage (from the project root, with .venv active):
    PYTHONPATH=src python3 scripts/diagnose_refine.py tests/assets/images/blob-13.jpg --subpixel
    PYTHONPATH=src python3 scripts/diagnose_refine.py tests/assets/images/blob-13.jpg --subpixel --refine-search 2.5

Crop, ROI and q_limit come from the config (--config). The profile mode is always deskew;
the average-mode result with the same subpixel setting is printed for comparison.
Plots are written to --out (default outputs/diagnose/).

Also reported per ROI, from the deskewed profile over +/-40 px around pass 1:
- <stem>_profile_<roi>.png: the profile with pass-1 / pass-2 / chosen centers and its mirror
  image about the chosen center (asymmetry shows as a gap between the two curves).
- The pass-2 center for fine_window = 10, 20, 40, 80 px (drift => asymmetric profile).
- Background on each side (mean of the outer 10 px), line FWHM and half-widths, and the
  RSS contrast min/mean (r1) on the configured profile.
- Next to that FWHM: the pass-1 line-width estimate (``_estimate_line_fwhm``) and the
  half-width the adaptive fine_window mode would use (``_adaptive_fine_window``).
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
    _adaptive_fine_window,
    _deskew_position,
    _deskew_profile,
    _sdrm_core,
    find_cross_center_detailed,
    sdrm_ssd_curve,
)

PLOT_RANGE_PX = 3.0
PROFILE_WINDOW_PX = 40
DRIFT_WINDOWS_PX = (10, 20, 40, 80)
BACKGROUND_BAND_PX = (30.0, 40.0)  # outer 10 px on each side of the chosen center
PEAK_SEARCH_PX = 5.0


def half_crossing(u: np.ndarray, s: np.ndarray, i_peak: int, half: float, direction: int) -> float | None:
    """Coordinate where s first drops below ``half`` walking from i_peak in ``direction`` (linear interp)."""
    i = i_peak
    while 0 <= i + direction < len(s):
        j = i + direction
        if s[j] < half:
            return float(u[i] + (half - s[i]) * (u[j] - u[i]) / (s[j] - s[i]))
        i = j
    return None


def profile_stats(u: np.ndarray, profile: np.ndarray, center: float) -> dict:
    """Background per side, polarity, FWHM and half-widths of a profile around ``center``."""
    rel = u - center
    lo, hi = BACKGROUND_BAND_PX
    bg_minus = float(np.mean(profile[(rel >= -hi) & (rel <= -lo)]))
    bg_plus = float(np.mean(profile[(rel >= lo) & (rel <= hi)]))
    bg = 0.5 * (bg_minus + bg_plus)
    near = np.abs(rel) <= PEAK_SEARCH_PX
    sign = 1.0 if profile[near].max() - bg >= bg - profile[near].min() else -1.0
    s = sign * (profile - bg)
    i_peak = int(np.flatnonzero(near)[np.argmax(s[near])])
    half = 0.5 * s[i_peak]
    left = half_crossing(u, s, i_peak, half, -1)
    right = half_crossing(u, s, i_peak, half, +1)
    return {
        "bg_minus": bg_minus,
        "bg_plus": bg_plus,
        "polarity": "bright" if sign > 0 else "dark",
        "amplitude": float(s[i_peak]),
        "peak_at": float(u[i_peak]),
        "fwhm": None if left is None or right is None else right - left,
        "hw_minus": None if left is None else center - left,
        "hw_plus": None if right is None else right - center,
    }


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
    ap.add_argument("--fine-window-mode", choices=("fixed", "adaptive"))
    ap.add_argument("--refine-search", type=float, metavar="PX")
    ap.add_argument("--max-refine-shift", type=float, metavar="PX")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        return 2

    overrides: dict = {"profile_mode": "deskew"}
    if args.subpixel:
        overrides["subpixel"] = True
    for key in ("subpixel_neighbors", "oversample_step", "fine_window", "fine_window_mode", "refine_search",
                "max_refine_shift"):
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
        fine_window_mode=mp.fine_window_mode,
        refine_search=mp.refine_search,
        max_refine_shift=mp.max_refine_shift,
    )

    step = mp.oversample_step
    search = int(round(mp.refine_search / step))
    wide = int(round(PLOT_RANGE_PX / step))
    readout = f"subpixel n={mp.subpixel_neighbors}" if mp.subpixel else "integer"
    print(f"{args.image.name}: deskew / {readout}, oversample_step={step:g}, fine_window={mp.fine_window} ({mp.fine_window_mode}), "
          f"refine_search={mp.refine_search:g} px (= +/-{search} samples), "
          f"max_refine_shift={mp.max_refine_shift:g} px, q_limit={q_limit:g}")
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
    print(f"  {'ROI':<7}{'axis':<5}{'method':<16}{'slope':>9}{'pass1':>11}{'pass2':>11}{'diff':>8}"
          f"{'min idx':>10}{'':>6}{'min +/-3px':>12}  plot")
    for trace, method in zip(deskew.rois, deskew.roi_methods):
        axis, origin = axis_origin(trace)
        p1 = origin + trace.pos1
        p2 = None if trace.pos2 is None else origin + trace.pos2
        diff = None if p2 is None else p2 - p1
        idx_txt, flag, global_txt, plot_txt = "-", "", "-", "-"

        profile = None if trace.slope is None else _deskew_profile(
            trace.roi, trace.pos1, trace.slope, oversample_step=step, fine_window=mp.fine_window, name=trace.name
        )[0]
        if profile is not None:
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
                chosen = "chosen minimum" if method == "deskew" else f"rejected minimum, {method}"
                ax.axvline(trace.pos2 - trace.pos1, color="tab:red", lw=1.5,
                           ls="-" if method == "deskew" else ":", label=f"{chosen} ({diff:+.3f} px)")
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

        print(f"  {trace.name:<7}{axis:<5}{method:<16}{fmt(trace.slope, 5):>9}{fmt(p1):>11}{fmt(p2):>11}"
              f"{fmt(diff):>8}{idx_txt:>10}{flag:>6}{global_txt:>12}  {plot_txt}")

    report_profiles(args, mp, deskew, step)
    return 0


def report_profiles(args, mp, deskew, step: float) -> None:
    """Profile plots, center vs. fine_window, and background/FWHM per ROI."""
    drift_rows, stats_rows = [], []
    for trace, method in zip(deskew.rois, deskew.roi_methods):
        axis, origin = axis_origin(trace)
        if trace.slope is None:
            continue
        profile, _m = _deskew_profile(
            trace.roi, trace.pos1, trace.slope, oversample_step=step, fine_window=PROFILE_WINDOW_PX, name=trace.name
        )
        if profile is None:
            drift_rows.append((trace.name, axis, origin + trace.pos1, [], None))
            stats_rows.append((trace.name, axis, method, None, None, None))
            continue

        half_bins = int(round(PROFILE_WINDOW_PX / step))
        u = origin + trace.pos1 + (np.arange(profile.size) - half_bins) * step
        p1 = origin + trace.pos1
        p2 = None if trace.pos2 is None else origin + trace.pos2
        chosen = p2 if method == "deskew" else p1
        mirror = np.interp(2.0 * chosen - u, u, profile, left=np.nan, right=np.nan)

        # 1. Profile plot with mirror image.
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(u, profile, color="tab:blue", lw=1.2, label="deskewed profile")
        ax.plot(u, mirror, color="tab:orange", lw=1.0, ls="--", label=f"mirror about {chosen:.3f}")
        ax.fill_between(u, profile, mirror, where=~np.isnan(mirror), color="tab:red", alpha=0.15,
                        label="profile - mirror")
        ax.axvline(p1, color="tab:gray", ls="--", lw=1, label=f"pass 1 ({p1:.3f})")
        if p2 is not None:
            ax.axvline(p2, color="tab:red", lw=1.2, ls="-" if method == "deskew" else ":",
                       label=f"pass 2 ({p2:.3f})" + ("" if method == "deskew" else f", rejected: {method}"))
        ax.set_xlabel(f"image {axis} (px)" + ("  [+y = down]" if axis == "y" else ""))
        ax.set_ylabel("intensity")
        ax.set_title(f"{args.image.stem} {trace.name}: deskewed profile +/-{PROFILE_WINDOW_PX} px, {method}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = args.out / f"{args.image.stem}_profile_{trace.name}.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

        # 2. Center vs. fine_window.
        centers = []
        for fw in DRIFT_WINDOWS_PX:
            m, pos = _deskew_position(
                trace.roi, trace.pos1, trace.slope, oversample_step=step, fine_window=fw,
                subpixel=mp.subpixel, subpixel_neighbors=mp.subpixel_neighbors, name=trace.name,
                refine_search=mp.refine_search, max_refine_shift=mp.max_refine_shift,
            )
            centers.append((None if pos is None else origin + pos, m))
        drift_rows.append((trace.name, axis, p1, centers, plot_path))

        # 3. Background, FWHM, RSS contrast.
        cfg_profile, _m = _deskew_profile(
            trace.roi, trace.pos1, trace.slope, oversample_step=step, fine_window=mp.fine_window, name=trace.name
        )
        r1 = None if cfg_profile is None else _sdrm_core(cfg_profile, int(round(mp.refine_search / step)))[1]
        est_window, est_fwhm = _adaptive_fine_window(trace.roi.mean(axis=0), trace.pos1, mp.fine_window)
        st = profile_stats(u, profile, chosen)
        st.update(est_fwhm=est_fwhm, est_window=est_window)
        stats_rows.append((trace.name, axis, method, st, r1, chosen))

    print()
    print("  Center vs. fine_window (image px; * = pass-2 candidate rejected by a guard):")
    head = "".join(f"{f'fw={fw}':>12}" for fw in DRIFT_WINDOWS_PX)
    print(f"  {'ROI':<7}{'axis':<5}{'pass1':>10}{head}{'spread':>9}  profile plot")
    for name, axis, p1, centers, plot_path in drift_rows:
        cells = "".join(
            f"{'-':>12}" if c is None else f"{c:>11.3f}{'*' if m != 'deskew' else ' '}" for c, m in centers
        )
        vals = [c for c, _m in centers if c is not None]
        spread = fmt(max(vals) - min(vals)) if vals else "-"
        print(f"  {name:<7}{axis:<5}{p1:>10.3f}{cells}{spread:>9}  {plot_path or '-'}")

    print()
    print(f"  Profile shape (+/-{PROFILE_WINDOW_PX} px; background = mean of {BACKGROUND_BAND_PX[0]:g}-"
          f"{BACKGROUND_BAND_PX[1]:g} px from the chosen center; '-side' = lower x or y coordinate):")
    print(f"  {'ROI':<7}{'axis':<5}{'center':>10}{'bg -side':>10}{'bg +side':>10}{'+ minus -':>11}"
          f"{'line':>8}{'amp':>8}{'FWHM':>8}{'est':>8}{'adapt':>8}{'hw -':>7}{'hw +':>7}{'peak-ctr':>10}"
          f"{'RSS r1':>9}")
    for name, axis, method, st, r1, chosen in stats_rows:
        if st is None:
            print(f"  {name:<7}{axis:<5}  no deskewed profile ({method})")
            continue
        print(f"  {name:<7}{axis:<5}{chosen:>10.3f}{st['bg_minus']:>10.2f}{st['bg_plus']:>10.2f}"
              f"{st['bg_plus'] - st['bg_minus']:>+11.2f}{st['polarity']:>8}{st['amplitude']:>8.1f}"
              f"{fmt(st['fwhm'], 2):>8}{fmt(st['est_fwhm'], 2):>8}{st['est_window']:>8.2f}"
              f"{fmt(st['hw_minus'], 2):>7}{fmt(st['hw_plus'], 2):>7}"
              f"{st['peak_at'] - chosen:>+10.2f}{fmt(r1, 3):>9}")


if __name__ == "__main__":
    raise SystemExit(main())
