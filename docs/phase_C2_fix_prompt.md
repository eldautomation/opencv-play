# Phase C2 follow-up — pass-2 refinement guards (blob-13)

## Problem
In deskew mode, blob-13.jpg's y position moves by 0.98 px compared with average + subpixel (641.50 -> 640.52), and a1 changes from 3.552 to 3.636 deg. This suggests one horizontal ROI's pass-2 search hit the edge of its ±1.5 px range instead of finding a real minimum.

Work in two stages. Stop after Stage 1 and wait for my confirmation before Stage 2.

## Stage 1 — diagnose (no change to default or current deskew results)
- Add an optional MeasurementParameters field refine_search (px, default 1.5, valid 0.5–3.0, optional in TOML) that controls the ± search range of pass 2. With the default, results must be identical to now.
- Add a --refine-search option to scripts/evaluate_images.py.
- Write scripts/diagnose_refine.py that, for one image and settings given on the command line, prints for each ROI: pass-1 position, pass-2 position, their difference, and the index of the fine-grid minimum relative to the search range (flag it if it is at an edge). It also saves a plot of each ROI's fine RSS curve over ±3 px, with the pass-1 estimate and the chosen minimum marked, to outputs/diagnose/.
- Run it on tests/assets/images/blob-13.jpg with deskew + subpixel at refine_search 1.5 and 2.5. Also run it on blob-22.jpg at 1.5 as a healthy reference.
- Show me the printed tables and the paths of the plots, then STOP. Do not add guards yet.

## Stage 2 — guards (only after I confirm)
- In pass 2, fall back to that ROI's pass-1 result when:
  - the fine minimum lies at either edge of the search range, or
  - |pass2 - pass1| exceeds max_refine_shift (new MeasurementParameters field, default 0.75 px, optional in TOML, valid 0.1–3.0).
- Record the reason in roi_method: "fallback:edge", "fallback:shift", and rename the existing fallbacks to "fallback:tilt" and "fallback:bins". Old measurement files must still load.
- Tests:
  - Regression test: blob-13 in deskew + subpixel mode is within 0.1 px of average + subpixel in x and y.
  - Unit test: a synthetic profile with a second dip near the search edge falls back with reason "fallback:edge".
  - Unit test: a pass-2 result more than max_refine_shift from pass 1 falls back with reason "fallback:shift".
- Default behavior and all existing C2 accuracy tests must stay unchanged.
- Run the full pytest. Only tests/integration/test_center_finding.py (R8) may fail.

Use python3 in all commands and docs. Commit Stage 1 and Stage 2 separately, with the message prefix "C2-fix:".
