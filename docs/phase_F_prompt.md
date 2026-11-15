# Phase F — Instrument measurements: tilt and rotation in angular units (R3, R5)

Read docs/code_review_2026-09.md (R3, R5) first. The design decisions below are final; do not change them. If something in the code makes one of them impossible, stop and ask.

## Goal
Separate the two kinds of measurement:
- **Image measurement** (pixel domain, existing): what find_cross_center finds in the image. Pixel coordinates are always measured from the image corner, as now.
- **Instrument measurement** (new): what the autocollimator reports, computed from the image measurement plus what is known about the instrument (EFL, pixel pitch, reticle, zero references), in selectable angular units.

The image layer must not know about the instrument; the instrument layer must not touch images.

## 1. Structure
New modules (names may be adjusted to fit the package, keep the separation):
- `src/autocollimator/units.py`
  - `AngleUnit` enum: RAD, DEG, MRAD, URAD, ARCMIN, ARCSEC, each with its conversion factor from radians and a display symbol (rad, °, mrad, µrad, ′, ″).
  - `OutputUnits` frozen dataclass: `tilt_unit` and `rotation_unit`, set independently. Default: both DEG.
- `src/autocollimator/instrument.py`
  - `ReticleType` enum, currently one value: CROSSHAIR. Selected from a list in the config. Designed so other reticle types can be added later.
  - `InstrumentModel` frozen dataclass: reticle_type, collimator_efl_mm, microscope_efl_mm (stored, unused), effective pixel pitch x/y (µm, from the capture mode), expected image size, tilt zero reference (mode + values), rotation zero reference (mode + value). Built from the config.
  - `to_angular(image_measurement, model, units) -> AngularMeasurement`: pure function, no I/O. Internally always radians; units are applied only at the end.
  - `AngularMeasurement` frozen dataclass: tilt_x, tilt_y (tilt_unit), rotation (rotation_unit), orthogonality (rotation_unit), the units used, the zero references used, and the intermediate spot displacement in mm (x, y). For a failed image measurement, all angular values are None and success is False.
- `MeasuredValues` stays the image-measurement record (pixel results). workflow.py composes: image → image measurement → to_angular → MeasurementOutput containing image measurement + angular measurement + InstrumentModel snapshot.

## 2. Conversion (exact)
- Image center: ((W−1)/2, (H−1)/2) in pixel-center coordinates.
- Axes: X right, Y up. Image y points down, so: dx_mm = (x − x0) · pitch_x, dy_mm = −(y − y0) · pitch_y, with pitch in mm.
- Tilt, exact formula (reflection doubles the angle): tilt_x = ½ · atan(dx_mm / collimator_efl_mm), tilt_y = ½ · atan(dy_mm / collimator_efl_mm).
- Names: `tilt_x` = horizontal spot motion, `tilt_y` = vertical spot motion. Positive tilt_x: spot moves right; positive tilt_y: spot moves up.

## 3. Zero references (stored in the device config, set independently)
Tilt zero, selectable mode:
- `image_center` (default): (x0, y0) = image center.
- `calibrated_point`: (x0, y0) in pixels, entered in the config.
- `measured_zero`: (x0, y0) taken from a measurement at runtime and saved to the config.
Rotation zero, selectable mode:
- `image_axes` (default): rotation relative to the image axes.
- `measured_zero`: rotation value taken from a measurement at runtime and saved to the config. Reported rotation = wrap(rotation − zero). Example: zeroed at 1.43°, now 1.57° → reads 0.14°.
- Add app methods to set the tilt zero and the rotation zero from a measurement (each independently) and persist them to the device config through the existing config store. Setting from a failed measurement must raise a clear error.

## 4. Rotation and orthogonality (crosshair)
- Fix the angle wrap (R3): a crosshair is symmetric under 90°, so each line angle and the rotation are wrapped to (−45°, +45°]. Today −3.8° is reported as 176°; it must read −3.8°.
- rotation = mean of the two wrapped line angles (handle the wrap correctly when averaging).
- orthogonality = difference between the two line angles = deviation of the angle between the lines from 90°. Reported only; it is never calibrated or zeroed.
- Check in the code which of a0/a1 belongs to which line and document the sign convention (positive rotation = counter-clockwise as seen in the image with Y up).

## 5. Config changes
Detect optics:
- Rename `efl1` → `collimator_efl_mm`, `efl2` → `microscope_efl_mm` in models, TOML and code. Old TOML keys must still load (map them with a deprecation warning).
- Two product configurations in the library: 200 mm (collimator 200, microscope 45) and 400 mm (collimator 400, microscope 90). Microscope values are placeholders. The device config selects one by id; the default device uses the 200 mm entry.
Image sensor:
- Keep the IMX296 entry: native 1456 × 1088, pixel pitch 3.45 µm.
Capture mode (new library entry, selected by the device config):
- Fields: output width/height, scaler crop on the sensor (x, y, width, height in sensor pixels). Effective pixel pitch x/y is computed: pitch × crop_width / output_width and pitch × crop_height / output_height.
- `native`: output 1456 × 1088, crop (0, 0, 1456, 1088) → 3.45 µm.
- `hd_1080`: output 1920 × 1080, crop (0, 134, 1456, 819) → ≈ 2.61625 µm in x and y. Mark this crop in the TOML and in the docstring as an ASSUMPTION (centered 16:9 crop, not yet verified with libcamera ScalerCrop metadata).
Reticle:
- `reticle_type = "crosshair"` in the device config, validated against ReticleType.
Mismatch policy (new device-config option): `image_size_mismatch = "error" | "warn"` (default "error"). Compares each image's size with the capture mode's output size; "warn" logs a warning and uses the image's own size for the center.
Units (new device-config options): `tilt_unit` and `rotation_unit` (default "deg"), overridable per call in the app API and in scripts/evaluate_images.py (--tilt-unit, --rotation-unit).
Test setup:
- The device config used by the tests selects capture mode `hd_1080` (matches the 1920 × 1080 test images) and the 200 mm optics.
- Add tests for the mismatch policy: a 1920 × 1080 image with capture mode `native` → error with "error", warning + correct center with "warn".

## 6. Output
- Measurement YAML contains the image measurement (pixel results), the angular measurement (with units), and the InstrumentModel snapshot (reticle type, EFLs, pixel pitch, capture mode, zero references). Old measurement files must still load.
- scripts/evaluate_images.py: add tilt_x, tilt_y, rotation, orthogonality columns with units in the header.

## 7. Tests (hand-computed values, not values taken from the code)
- Unit conversion round trips for every AngleUnit; 1 rad = 206264.806″, 1° = 3600″.
- 1 px displacement, pitch 3.45 µm, f = 200 mm: tilt = ½·atan(3.45e-3/200) = 8.625e-6 rad = 4.9418e-4° = 1.7790″. At f = 400 mm: half of that.
- Same with hd_1080 pitch 2.61625 µm, f = 200 mm: 6.5406e-6 rad = 1.3491″.
- Exact vs small-angle: at a large displacement (e.g. 500 px) the result matches ½·atan exactly, not d/(2f).
- Signs: spot right of center → tilt_x > 0; spot above center (smaller image y) → tilt_y > 0.
- Image center: a spot exactly at ((W−1)/2, (H−1)/2) gives tilt (0, 0).
- Zero modes: each tilt mode and each rotation mode, including the 1.43° → 1.57° → 0.14° example, set independently.
- Rotation wrap: line angles 176.2° and 176.3° give rotation ≈ −3.75°, and wrapping across ±45° works.
- Orthogonality is reported and unaffected by the rotation zero.
- Failed image measurement → AngularMeasurement with None values, success False.
- Config: old efl1/efl2 keys load with a warning; both product configurations load; effective pitch for both capture modes.
- Units chosen independently: tilt in mrad with rotation in degrees.

## 8. Review document
Mark R3 and R5 as resolved in docs/code_review_2026-09.md, noting the hd_1080 crop assumption as open (to be verified with libcamera ScalerCrop).

Work in steps, each committed separately with the prefix "F:": (1) units + instrument model + conversion with unit tests, (2) config changes, (3) zero references and app methods, (4) workflow, YAML output and evaluate script, (5) review document. Run pytest after each step. Image-measurement results (pixel positions) must not change. Use python3. All tests must pass.
