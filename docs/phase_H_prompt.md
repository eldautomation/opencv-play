# Phase H — Cleanup (R11–R14) and R21

Read docs/code_review_2026-09.md (R11, R12, R13, R14, R21) first. The decisions below are final; if something in the code makes one of them impossible, stop and ask.

## Behavior guarantee
No measurement result may change, except where step 1 (R21) explicitly changes behavior. All existing tests, tests/assets/images/manifest.yaml and the pinned values must pass WITHOUT edits. If a test needs changing for any other reason, stop and tell me why.

## Steps (each committed separately, prefix "H:")

### 1. R21 — NaN quality ratio passes as success
- In sdrm_2 (and anywhere else a quality ratio is compared with a limit), treat a non-finite q (NaN or inf) as a quality failure, with a clear reason in the message (e.g. "ROI top: quality ratio not finite (flat profile)").
- Tests: a perfectly flat ROI fails with that reason, in average and deskew mode; finite values behave exactly as before.
- Report whether any image in tests/assets/images changes status (expected: none).

### 2. R12 — ROI length, unused parameters, NameError
- `roi_size_x`: accept an integer or "auto". "auto" = min(crop width, crop height), which is today's behavior, and becomes the default in configs/library/measurement_parameters.toml and in the code. An explicit integer is honored (validated: > 0 and not larger than the crop). With "auto", results must be identical to now.
- Remove the unused parameters `roi_w_v` and `roi_w_h` (signature, docstrings, callers).
- Keep `slant` but rename it `nominal_slant` (unit: degrees), default 6.0. It is not used by the measurement yet; document in the docstring that it is the nominal (designed) reticle rotation, to be compared with the measured rotation later (see R24). Add it as an optional MeasurementParameters field (default 6.0) so it is configurable; existing TOML must load unchanged.
- Fix the NameError on `q_ratio` in find_center_pixel for the "argmax" and "moments" methods; add tests that call both methods.

### 3. R11 — type annotations
- Fix return annotations that do not match what functions return, across src/. Annotation changes only; no logic changes.

### 4. R13 — output filename collisions
- Make measurement output filenames unique: timestamp with milliseconds, plus a counter suffix if the name still exists.
- Remove the 1.7 s sleep from scripts/demo_app.py.
- Test: two measurements saved in quick succession get different filenames and both files exist.

### 5. R14 — dead code, prints, imports
- Remove dead code: the old `sdrm`, `run_demo`, the unreachable `_default_crop_from_image` branch, the computed-and-discarded `pairs` block, and any other code shown to be unreachable or unused (list each removal in the commit message).
- Replace stray `print` calls in src/ with logging.
- Import matplotlib lazily inside the debug plotting code only, so importing autocollimator does not import matplotlib (add a test that checks this).

### 6. Archive old experiments
- Move test_scripts/, test.py (repo root), test_out/ and tests/readme-tests.py into a new top-level archive/ folder with `git mv` (history kept).
- Add archive/README.md: one paragraph saying these are old experiments, not maintained, not part of the package or test suite.
- Make sure archive/ is excluded from packaging and from pytest collection.

### 7. Review document and CLAUDE.md
In docs/code_review_2026-09.md:
- Mark R11, R12, R13, R14 and R21 as resolved, with a one-line note each.
- Add:

### R24 — Compare measured rotation with nominal slant
`nominal_slant` (degrees, default 6.0) records the designed reticle rotation relative to the sensor. Add a function that compares the measured rotation (instrument layer, Phase F) with the nominal slant and reports the deviation, e.g. as a setup/assembly check (reticle mounted at the intended angle) and as a sanity check for deskew mode, which needs a minimum tilt (see R20). Define the tolerance and the reporting format when implementing.

### R25 — Run code-quality tools (ruff, mypy)
ruff and mypy are in the dev extras but have never been run. Run both on src/ and tests/, fix style and type issues that do not change logic, and report anything that would need a logic change. To be done as its own phase.

In CLAUDE.md: the line-ending rule becomes "All text files use LF (enforced by .gitattributes)", and add `nominal_slant` and the "auto" ROI length where parameters are described.

### 8. Line endings (last, separate commit, nothing else in it)
- Add .gitattributes: `* text=auto eol=lf`, with binary types (png, jpg, jpeg, bmp, tif, tiff, pdf, zip) marked binary and the existing LFS patterns kept.
- Convert the remaining CRLF files (app.py, config/store.py, scripts/demo_app.py, and any others found) to LF with `git add --renormalize .`.
- Verify `git diff --ignore-all-space HEAD~1` shows no content changes, and run the full pytest.

Run pytest after each step. Use python3. All tests must pass. Nothing under outputs/ is committed and configs/main.toml must not be modified by the tests.
