# Phases N0, N1, N2, N11 — combined prompt

Work through the four parts below IN THIS ORDER: N0 → N1 → N2 → N11. Each part has its own commits with its own prefix ("N0:", "N1:", "N2:", "N11:"). Run pytest after each step.

## Rules for all parts
- Use python3 in all commands and docs.
- Never change expected values, pins or tests/assets/images/manifest.yaml, and never loosen a tolerance, unless a part below explicitly allows it. If a test fails for any other reason, stop and tell me.
- Nothing under outputs/ is committed. configs/main.toml must not be modified by the tests.
- There are THREE mandatory stops (marked STOP below). At each stop, show me the requested output and wait for my reply before continuing:
  1. N0: after the restructure, show the section headings.
  2. N1: after the before/after report and the proposed manifest changes.
  3. N11: after the list of findings whose fix would change logic.
- After N1's stop, I will review and commit the manifest myself. Continue with N1 step 3 only when I say the reviewed manifest is committed.
- At the end, give me a short summary of all commits and anything that differed from this prompt.

## Part 1 — N0: Restructure the review document (new numbering)

Documentation only. No code, tests, configs or other files change.

Restructure docs/code_review_2026-09.md as follows:

1. At the top, after the title, add a short "Numbering" note:
   "Open items use the N series (N1, N2, …) since 2026-11-15. Items closed before that keep their original R numbers, which are referenced in commit messages. Old R numbers of open items are given in brackets."
2. Section "Open items": replace all open, mitigated, prepared and proposed R items (R16, R17, R18, R19, R20, R22, R23, R24, R25) with the N items below, word for word, in this order. Add N2 (new, was proposed as R26) and N9 (open assumption from R5).
3. Section "Resolved (old numbering)": move every closed item (R1–R15, R21) here, unchanged, in numeric order, including their resolution notes.
4. Do not delete any information: if an old open R item has details that are not in its N text below, append them under the N item as "Earlier notes (Rxx):".
5. Update CLAUDE.md: "Known review findings are tracked in docs/code_review_2026-09.md: open items N1–N11, resolved items under their old R numbers. Reference N numbers in new commit messages."

Commit as "N0: renumber open review items (N series)". STOP: show me the resulting section headings and wait for my reply.

### Open items (text to insert)

In the review document, give each item below a level-3 heading (`### N1 — …`); the extra `#` here is only because this prompt nests it.

#### N1 — Default ROI depth 20 → 40 px  [was R20]
Status: decided 2026-11-15; prepared (docs/phase_N1_prompt.md).
The default depth of the four ROI strips (`roi_size_y`, along the line) changes from 20 px to 40 px. Deskew needs roi_depth × tan(tilt) ≥ 1 px: 20 rows → tilt ≥ ~2.9°, 30 → ~1.9°, 40 → ~1.4°, 60 → ~1.0°. Deeper ROIs also average more rows (lower noise) but pick up more line curvature or distortion along the line. This deliberately changes default results; pinned values and the manifest are updated with a before/after report, without loosening tolerances.

#### N2 — Packaging: config package missing from normal installs  [was proposed as R26]
Status: open.
`src/autocollimator/config/` has no `__init__.py`, so setuptools' package discovery finds only `autocollimator`. The editable development install works, but a normal (non-editable) install — as on the Raspberry Pi — leaves out the config package and the app fails to start. Also an empty, untracked `src/autocollimator/utils/` directory next to `utils.py`. Found during Phase H.

#### N3 — ROI edge clipping gives silently biased results  [was R22]
Status: open.
When a line lies at the edge of an ROI (cut off), SDRM is pulled toward the clipped side but can still pass the quality check. blob-16 with the configured crop was off by ~15 px with q 0.39/0.46 (the cause of former R15). The pre-search crop avoids it by centering the lines; the measurement itself has no check. Fix: detect a line minimum near the ROI profile ends and fail or flag it.

#### N4 — Center depends on crop geometry (lines not straight)  [was R23]
Status: open.
Between the configured crop and the 540 px pre-search crop, centers differ by median 1.09 px, max ~2.2 px (excluding blob-16), also with sub-pixel and deskew. Line angles differ by up to ~0.3° (Test4: 0.76° vs 0.45°): the lines are not perfectly straight or even, and two points per line sample them differently. Exceeds the ~0.3 px target. Crops centered on the crosshair (pre-search/track) sample the same reticle parts as the spot moves, which may make differential measurements more consistent; to be verified with bench data. Remedy: several strips per arm with a robust line fit (robustness step 3).

#### N5 — Quality ratio penalizes thin lines  [was R16]
Status: open.
With q_limit 0.5, synthetic 1.5 px FWHM lines fail pass 1 at 6/11 sub-pixel positions. The min/mean RSS ratio depends on line width relative to the search range. Needs a width-independent quality measure (robustness step 5), validated on bench images of the production reticle.

#### N6 — Saturated line profiles  [was R18]
Status: instrumented.
The green channel is clipped at 255 on the line, giving flat tops that carry no position information. Saturation fraction per ROI (`saturation_s0..s3`) and `gray_mode` (luma/green/max) exist. Capture settings are handled separately; software robustness (saturation-masked symmetry) is still to do (robustness step 4).

#### N7 — Window-dependent pass-2 refinement  [was R17]
Status: mitigated.
On asymmetric profiles, pass 2 can run to its search edge (blob-13 right ROI). Caught by the fallback guards (`fallback:edge`, `fallback:shift`) and the opt-in adaptive window. Root causes are N6 (saturation) and N8 (ghost band).

#### N8 — Ghost band in blob images  [was R19]
Status: open.
A bright, near-white, off-center band (likely an internal reflection) sits on the lines in the blob images; absent in the bench images (Test8, Test14). No pre-search ambiguity warnings triggered. Software remedy: local background removal or masking (robustness step 4).

#### N9 — Verify the hd_1080 capture crop  [open assumption from R5]
Status: open.
The `hd_1080` capture mode assumes a centered 16:9 sensor crop (0, 134, 1456, 819), giving 2.61625 µm effective pixels. To be verified against libcamera ScalerCrop metadata on the Pi. An error scales all tilt values by the same factor (a later measured-EFL calibration would absorb it).

#### N10 — Compare measured rotation with nominal slant  [was R24]
Status: open.
`nominal_slant` (degrees, default 6.0) records the designed reticle rotation relative to the sensor. Add a function that compares the measured rotation (instrument layer) with the nominal slant and reports the deviation, as a setup/assembly check and as a sanity check for deskew mode, which needs a minimum tilt (see N1). Define tolerance and reporting format when implementing.

#### N11 — Code-quality tools (ruff, mypy)  [was R25]
Status: prepared (docs/phase_N11_prompt.md).
ruff and mypy are in the dev extras; they were used selectively in Phase H but never run fully. Run both on src/ and tests/, fix style and type issues that do not change logic, report anything that would need a logic change, and add them to the documented workflow.

---

## Part 2 — N1: Default ROI depth 40 px (was 20)

Read N1 in docs/code_review_2026-09.md first. Decision (project owner): the default depth of the four ROI strips (`roi_size_y`, the dimension along the line) changes from 20 px to 40 px. This deliberately changes default results. Work in the order below and stop where indicated.

### 1. Change the default
- Set roi_size_y = 40 in the default measurement parameter set in configs/library/measurement_parameters.toml.
- Change every code-level default for the ROI depth (function defaults, dataclass defaults, scripts/evaluate_images.py, scripts/evaluate_presearch.py, scripts/diagnose_refine.py, scripts/regenerate_expected.py, and any other script) from 20 to 40.
- Tests that pass an explicit depth of 20 (e.g. roi_size=("auto", 20)) test the function with those arguments; leave them unchanged.
- Check that a 40 px ROI fits inside the crop and the image for all test images, in configured-crop mode and in pre-search mode (min_crop_size_px 240). Report any case where it does not and how it is handled. Do not change the crop logic.

### 2. Before/after report — then STOP
Run scripts/evaluate_images.py on tests/assets/images with depth 20 (temporary override, not a config edit) and depth 40, for:
- default settings
- --subpixel
- --profile-mode deskew --subpixel
- --presearch-mode every_image
Show me: success count before/after per mode; images that change status; position changes (median, max) per mode; how many ROIs now use deskew instead of fallback:tilt; Test4, Test7 and blob-06 per-ROI methods.
Then run scripts/regenerate_expected.py and show me a table of manifest entries whose expected position would move by more than 0.1 px or whose expected status would change.
STOP here. Do not change any expected value, pin or the manifest until I confirm.

### 3. After I confirm: update expected values
- Re-pin ONLY values that depend on the default ROI depth (e.g. the blob-22 default and deskew+subpixel pins, default-config pins in test_signal_quality_defaults.py and similar). List old → new for each in the commit message. Do not loosen any tolerance and do not change any other expected value. If a test fails for a reason other than the depth change, stop and tell me.
- Do NOT write tests/assets/images/manifest.yaml yourself: I will copy outputs/manifest_proposed.yaml into place after reviewing it.

### 4. Docs
- Update CLAUDE.md and README text that mentions the ROI depth.
- In docs/code_review_2026-09.md, mark N1 as resolved with the before/after summary.

Commits: "N1: default ROI depth 40 px (was 20)" for steps 1+3, "N1: docs" for step 4. Use python3. All tests must pass at the end, except the manifest test until I commit the reviewed manifest.

---

## Part 3 — N2: Packaging: make normal installs complete

Read N2 in docs/code_review_2026-09.md first. No measurement behavior may change; all existing tests, the manifest and pins must pass without edits.

### 1. Package structure
- Add an empty `src/autocollimator/config/__init__.py` (or one with only a docstring and the existing public names, if other modules expect them).
- Check every directory under src/autocollimator/ for a missing `__init__.py` and fix any other case the same way; list them.
- Remove the empty, untracked `src/autocollimator/utils/` directory (it sits next to utils.py and would shadow it if it ever got an __init__.py).
- Check pyproject.toml package discovery (`[tool.setuptools.packages.find]` or equivalent) so that all subpackages are included and archive/, tests/, scripts/ and outputs/ are not.

### 2. Packaging test
- Add a test (pytest marker "packaging", excluded from the default run if it takes more than a few seconds) that builds a wheel into a temporary directory with `python3 -m pip wheel . --no-deps --no-build-isolation -w <tmp>` and asserts that the wheel contains `autocollimator/__init__.py`, `autocollimator/config/__init__.py`, `autocollimator/config/models.py` and `autocollimator/config/store.py`, and does not contain anything from archive/ or tests/.
- Add a second check that installs that wheel into a fresh temporary virtual environment (`python3 -m venv`, `--system-site-packages` allowed for numpy/opencv) and runs `python3 -c "import autocollimator.config.models, autocollimator.config.store, autocollimator.app"` there. If this needs network access that is not available, skip with a clear reason instead of failing.

### 3. Report only (do not change)
- Does the installed package depend on files outside the package at runtime (e.g. configs/ found via a repo-relative path, scripts/)? List each such dependency and how the app locates it, so we can decide how configs are deployed to the Pi.

### 4. Docs
- In docs/code_review_2026-09.md, mark N2 as resolved.
- In CLAUDE.md, add the packaging test command.

Commit as "N2: complete package structure and packaging test" (code) and "N2: docs". Use python3. All tests must pass.

---

## Part 4 — N11: Code-quality tools (ruff, mypy)

Read N11 in docs/code_review_2026-09.md first.

### Behavior guarantee
No measurement result may change. All existing tests, tests/assets/images/manifest.yaml and the pinned values must pass WITHOUT edits to expected values. Before starting (i.e. after N1 and N2 are complete, with the new ROI depth), save the workflow results for all images in tests/assets/images in four modes (default, --subpixel, --profile-mode deskew --subpixel, --presearch-mode every_image) and compare after each step; every value must be identical.

### 1. Configure (commit "N11: configure ruff and mypy")
- In pyproject.toml, configure ruff: target Python 3.10, line length matching the existing code (check what is common; do not reformat the whole codebase to a new length), rule sets E, F, W, I (imports) and B (bugbear). Exclude archive/ and outputs/.
- Configure mypy for src/: Python 3.10, `ignore_missing_imports` only for libraries without type information (cv2 and similar), `warn_unused_ignores`, `check_untyped_defs`. Do not enable strict mode.
- Run both and show me the counts per rule/error code before fixing anything.

### 2. Fix — mechanical only
- ruff: apply safe automatic fixes (`ruff check --fix`, not `--unsafe-fixes`) and import sorting. Then fix remaining issues by hand only where the fix cannot change behavior.
- mypy: fix type errors by correcting annotations, adding precise types, or narrowing Optional values where the code already guarantees them. Use `# type: ignore[code]` only with a short comment explaining why.
- Do NOT reformat code with a formatter (no `ruff format` / black) in this phase.
- One commit per tool ("N11: ruff fixes", "N11: mypy fixes"), each listing the categories fixed.

### 3. Report — do not fix
List every finding whose fix would change logic or behavior (e.g. B-rules about mutable defaults, unreachable branches, possible None dereferences in the measurement path), with file, line, what the tool says and what the fix would change. Stop and show me this list.

### 4. Workflow (commit "N11: docs and workflow")
- CLAUDE.md: add `ruff check src tests` and `mypy src` to the commands, and the rule that both must be clean before a phase is finished.
- Add a unit test or a pytest marker "lint" that runs both tools, excluded from the default run, so they can be run with one command.
- In docs/code_review_2026-09.md, mark N11 as resolved, noting how many findings were fixed and how many were reported for later.

Use python3. All tests must pass and the four-mode comparison must be identical after every step.
