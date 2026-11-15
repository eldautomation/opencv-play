# Phase G — Pre-search: find the crosshair anywhere in the image (R4)

Read docs/code_review_2026-09.md (R4) first. The design decisions below are final; if something in the code makes one of them impossible, stop and ask.

## Goal
Add a pre-search routine that searches the whole image, determines where the crosshair is, and recommends a search center and search size for the measurement. The pre-search is its own subroutine, separate from the measurement routine, and is not always called. By default the measurement does not run it.

Default behavior must stay exactly as today (pre-search mode "off"): all existing results, the manifest test and pinned values unchanged.

## 1. Pre-search routine (image layer, e.g. src/autocollimator/presearch.py)
`presearch(image, params, reticle_type) -> PresearchResult`
- Pure image-layer function: no instrument knowledge, no config writing, no I/O.
- Selected by ReticleType (dispatch table); implement CROSSHAIR now. An unsupported reticle type returns a failed result with a clear reason.
- Downsample the image by `presearch_downsample` (default 2) before searching; all reported coordinates and sizes are in full-resolution pixels.
- Main method: the existing SDRM center finding with the ROIs spanning the full image — crop = whole image, ROI strips across the full width and/or height of the image.
- Fallback method, used only if the main method fails: row/column intensity projections of the (downsampled) image to locate the vertical and horizontal line; account for line tilt (smoothing/broadened peaks); integer accuracy is sufficient.
- Candidates: if more than one strong feature is found (e.g. a ghost reflection, R19), choose the strongest; if a second candidate reaches `presearch_ambiguity_ratio` (default 0.8) of the strongest's strength, add a warning naming both positions.

Success criteria — ALL must hold, otherwise success = False:
1. both lines found;
2. quality ratios below `presearch_q_limit` (configurable, separate from the measurement q_limit);
3. the intersection lies inside the image;
4. validation: one center-finding pass in the recommended region (inside the pre-search, on the same downsampled image) agrees with the pre-search center within `presearch_agreement_px` (default 5 px full-resolution).

Recommended region:
- Square. Size = `presearch_crop_fraction` (default 0.5) × min(image width, image height).
- Never smaller than `min_crop_size_px` (stored in the settings; see section 3), and never smaller than 4 × the estimated line FWHM if the pre-search can estimate it.
- Near the image edge: first shift the region inward so it lies fully inside the image; only if shifting is not enough, shrink it; fail (with reason) only if it would drop below the minimum size.

`PresearchResult` (frozen dataclass): success, failure_reason (None on success; a clear, specific text otherwise, e.g. "horizontal line not found", "intersection outside image", "validation disagreement 7.3 px", "region below minimum size"), recommended_center (x, y), recommended_size, method_used ("sdrm_full_frame" | "projection"), quality values, estimated line FWHM if available, warnings (list), candidates (list of positions and strengths), and whether the region was shifted or shrunk.

The pre-search never decides what happens on failure — it only reports. The caller decides.

## 2. Callers
On-demand: an app method that runs the pre-search on a given image and returns the PresearchResult. It does NOT save anything to the config.

Measurement workflow, `presearch_mode` (default "off"):
- "off": use the configured crop, as today; the pre-search is not called.
- "every_image": run the pre-search on each image; on success, measure with the recommended center and size.
- "track": use the previous successful measurement's center (kept in the app's memory, not persisted) with the recommended size; if there is no previous result or the measurement at the tracked position fails, run the pre-search; if that also fails, apply the failure action.
Workflow failure action when the pre-search fails, `presearch_failure_action`:
- "use_configured_crop" (default): measure with the configured crop and record the pre-search failure and its reason.
- "fail": return a failed measurement (success False) whose message contains the pre-search failure reason.

## 3. Settings
Add a pre-search settings group (optional in TOML, defaults as listed; existing TOML must load unchanged):
presearch_mode = "off" | "every_image" | "track" (default "off"), presearch_failure_action (default "use_configured_crop"), presearch_downsample (int ≥ 1, default 2), presearch_crop_fraction (0.1–1.0, default 0.5), min_crop_size_px (default 240), presearch_q_limit (default 0.7), presearch_agreement_px (default 5.0), presearch_ambiguity_ratio (default 0.8). Validate all values. Add matching options to scripts/evaluate_images.py (--presearch-mode etc.).

## 4. Output
- MeasurementOutput / measurement YAML: a pre-search section with the mode, whether the pre-search ran, the full PresearchResult (or None), and the crop actually used for the measurement (center, size, source: configured | presearch | tracked). Old files must still load.
- Evaluation report: columns for pre-search success, failure reason, method, recommended center and size, warnings, and crop source. Draw the recommended region on the overlay in a distinct color.

## 5. Evaluation (report this, do not tune to it)
The pre-search is a separate subroutine, so evaluate it separately from the measurement:
1. Pre-search alone, on all images in tests/assets/images: per image success, failure reason, method, recommended center/size, warnings.
2. Measurement success with the configured crop (presearch_mode "off") vs. with the crop recommended by the pre-search ("every_image"), on all images: success count for each, list of images that change status, and for images that succeed both ways the position difference (median, max) with a short explanation of any difference above 0.5 px (different crop size moves the ROIs along the lines).
3. The images that should not produce a center (Blur 1, Blur 2, Off 3) must still fail.
Show me these tables and stop before committing step 6.

## 6. Tests
Use the existing synthetic crosshair helper (and extend it if needed):
- Crosshair at many positions, including outside the default configured crop and near each image edge: pre-search success and recommended center within 2 px of truth; region shifted inward near edges; shrunk or failed correctly at the extremes.
- Region is square, size = fraction × min(W, H), respects min_crop_size_px and 4 × FWHM.
- Blank image and pure noise: failure with a specific reason.
- Main method failing but projection succeeding: method_used = "projection".
- Second candidate (synthetic ghost line) above the ambiguity ratio: warning recorded, strongest chosen.
- Downsample factor 1, 2, 4: coordinates reported in full-resolution pixels.
- Workflow modes: "off" gives results identical to today; "every_image" uses the recommended crop; "track" uses the previous center and falls back to the pre-search; both failure actions.
- The on-demand app method writes nothing to the config (git status unchanged after the tests).
- Unsupported reticle type: failed result with reason.

## 7. Review document
Mark R4 as resolved in docs/code_review_2026-09.md, describing the pre-search and that it is opt-in.

Work in steps, each committed separately with the prefix "G:": (1) PresearchResult + pre-search routine + unit tests, (2) projection fallback, (3) settings, (4) workflow modes and failure action, (5) output and evaluation report, (6) evaluation per section 5 — stop and show me — then (7) review document after I confirm. Run pytest after each step. Use python3. All tests must pass.
