# Phase C2 — opt-in deskew profile mode

Add an opt-in two-pass "deskew" profile mode to find_cross_center. It must be orthogonal to the existing subpixel option: profile_mode controls how the 1D profile is built, subpixel/subpixel_neighbors control how the SDRM minimum is read. All four combinations must work, and the default (profile_mode="average", subpixel=False) must behave exactly as today.

## Refactor first (no behavior change)
Split sdrm_2 so the RSS/sum-of-squares computation and minimum reading (integer or quadratic fit, per subpixel/subpixel_neighbors) live in a core function that does NOT apply q_limit, and sdrm_2 remains a wrapper that applies the q_limit check exactly as today. Offsets from the core are in samples of whatever grid it is given; the ±1 clamp and subpixel_neighbors are also in samples.

## Pass 1 (both modes)
Unchanged: averaged ROI profiles through sdrm_2, honoring subpixel/subpixel_neighbors, producing the four ROI points and the quality ratios. The q_limit decision is made here only, in both modes.

## Pass 2 (only when profile_mode="deskew")
- Slopes from pass 1: vertical line dx/dy from top and bottom points; horizontal line dy/dx from left and right points.
- For each ROI, project every pixel onto the axis across the line: u = col - slope * (row - row_ref), with row_ref = roi_height / 2 (matching the row used for the reported ROI point). Bin u at oversample_step (default 0.1 px), average per bin, fill empty bins by linear interpolation, record the empty-bin fraction.
- Call the SDRM core (no q_limit) on the oversampled profile, searching ±1.5 px around the pass-1 estimate (converted to samples), using a window of ±fine_window px (default 40) around it.
- subpixel=False: take the integer minimum on the fine grid (result quantized to oversample_step). subpixel=True: apply the quadratic fit on the fine grid with subpixel_neighbors counted in fine samples.
- Convert the result from samples to pixels using oversample_step.
- Fall back to that ROI's pass-1 result (which already honors the subpixel setting), with a debug log, when roi_depth * |slope| < 1 px or the empty-bin fraction exceeds 0.2.
- Record per ROI which method was used (deskew or fallback) in MeasuredValues and the log.

## Config and tools
- Add to MeasurementParameters (optional in TOML): profile_mode = "average" | "deskew" (default "average", validate the value), oversample_step (float, default 0.1, valid 0.02–0.5), fine_window (int, default 40). Existing TOML must load unchanged.
- Add --profile-mode, --oversample-step and --fine-window to scripts/evaluate_images.py, combinable with the existing --subpixel and --subpixel-neighbors. Add a column to the report showing profile mode and subpixel setting, and the per-ROI method used (deskew or fallback).

## Tests
- Helper that renders a synthetic crosshair: Gaussian line profile with given FWHM and tilt, pixel-integrated by 5x supersampling, background 30, peak 210, at a known sub-pixel position.
- Test all four combinations at 3° tilt, FWHM 1.5, 3 and 8 px, 11 sub-pixel positions from 0 to 1 px (errors after removing a constant offset):
  - average + subpixel off: identical to today's results.
  - average + subpixel on: identical to the results before this change (Phase C behavior unchanged).
  - deskew + subpixel off: within 0.06 px.
  - deskew + subpixel on: within 0.02 px.
- At 0° tilt, deskew must fall back for every ROI and return exactly the average-mode result with the same subpixel setting — test with subpixel both off and on.
- A pass-2 search must never cause a quality failure: an image that passes pass 1 must not fail in deskew mode.
- Defaults must still give (883.72, 566.83) for blob-22.

Use python3 in any commands or docs. Only tests/integration/test_center_finding.py (R8) may fail. Commit message prefix: "C2:".
