# Code review — September 2026

Reviewed: full `src/`, `tests/`, `configs/`, `scripts/demo_app.py`. Pipeline was run on all 17
images in `tests/assets/images` with the default config.

## Critical

### R1 — Failed measurements are reported as success
`find_cross_center` returns `(None, None)` for position on a quality failure. `workflow.py` checks
`success = position is not None`, which is always True for a tuple. Result: 9/17 test images
produce `success=True` with null position; 16/33 YAML files in `outputs/demo/` contain
`center_position_x: null` with `success: true`.
Fix: return `None` (or a result object with an explicit success flag) on failure; workflow must
build a `success=False` MeasurementOutput. `MeasuredValues` fields must allow `None` (Optional)
when success is False. Add a test using an image known to fail (e.g. blob-09.jpg with default config).

### R2 — Centering is quantized to ~0.5 px (sub-pixel missing)
`sdrm_2` returns an integer offset (argmin of RSS). Experiment: shifting blob-22.jpg by
0.1…0.7 px with cv2.warpAffine (INTER_CUBIC) gives the same measured shift (+0.54 px) every time.
Parabolic interpolation of the sum-of-squares around the minimum (points m-1, m, m+1):
`offset = k[m] + 0.5*(a - c)/(a - 2b + c)` tracked the true shift within ~0.05 px.
Also remove `round(..., 2)` in `line_intersection`.
Fix: return a float offset from `sdrm_2`; add a regression test that applies known sub-pixel
shifts (0.0–1.0 px in 0.1 px steps) and asserts error < 0.1 px.

### R3 — "Angles" are reticle rotation, not autocollimator tilt
`measured_angle_a0/a1` are image line orientations (degrees). Tilt from displacement
(θ = Δd · pixel_size / (2 · f)) is not implemented; `efl` config values are unused; output units are
hard-coded "Pixels". Angle wrap adds 180° to negatives, so −1.45° is reported as 178.5° (blob-06).
Needs design decisions before coding: which EFL (detect_optics.efl1/efl2?), zero reference,
output units (arcsec), sign convention. Keep reticle rotation as a separate, clearly named field,
wrapped to (−90°, 90°].

## Significant

### R4 — Fixed crop fails on about half the images
Crop center is fixed in config (872, 544); crosshair moves between images. Add a coarse locate
pass (e.g. row/column projection profiles on a downsampled image) then the existing refine pass.

### R5 — Config does not match hardware / design doc
Test images are 1920×1080; configured sensor is IMX296 (1456×1088); no check that image size matches
the sensor. Detect optics EFLs are 50/100 mm, not 200/400 mm.

### R6 — Encryption key written into every measurement YAML
`asdict(measurement)` includes `hardware.device.encryption_key`. Unset env var is silently accepted
as the literal `${DEVICE_ENCRYPTION_KEY}`. Exclude from serialization; fail clearly when unset.

### R7 — Fresh install is broken
`tomli`, `tomli_w`, `matplotlib` imported but not in `pyproject.toml`. `setup.config` is ignored by
setuptools (only `setup.cfg` is read) — delete it; `pyproject.toml` is the single source.

## Tests

### R8 — Integration test crashes
Second `find_cross_center` call unpacks 2 values from a 4-tuple (and omits `q_limit`).

### R9 — Integration test is not portable
Expected summary contains absolute `/home/cherrmann/...` paths and exact float strings. Compare
positions numerically with a tolerance; don't include paths in expected data.

### R10 — Two unit tests pass for the wrong reason
`test_pct_list_to_int_list_invalid_type` and `_none` omit `scale`, so the TypeError comes from the
missing argument.

## Cleanup

### R11 — Wrong type annotations
`workflow.run_center_finding_on_image` and `find_cross_center` return annotations don't match returns.

### R12 — Unused / overridden parameters and a latent NameError
`roi_w` is overwritten with `min(crop_w, crop_h)` so `roi_size_x` config is ignored; `roi_w_v`,
`roi_w_h`, `slant` unused. `find_center_pixel` raises NameError on `q_ratio` for argmax/moments.

### R13 — Output filename collisions
Prefix has 1-second resolution; demo sleeps 1.7 s to avoid overwrites. Add milliseconds or a counter.

### R14 — Dead code and Pi performance
Remove old `sdrm`, `run_demo`, unreachable `_default_crop_from_image` branch, stray `print`s, and the
discarded `pairs` computation. Import matplotlib lazily inside debug plotting. Consider Git LFS for
test images (repo zip is ~307 MB). Add `.gitattributes` for consistent line endings.
