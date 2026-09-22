# CLAUDE.md — autocollimator (opencv-play)

## What this project is
Software for a Raspberry Pi–based electro-optical autocollimator. A crosshair reticle is
imaged by a camera; we find the crosshair center to sub-pixel precision and convert its
displacement into a mirror tilt angle. Product configurations: 200 mm and 400 mm focal
length. Target resolution ~±0.5 arcsec (200 mm), which requires roughly ≤0.3 px centering
precision with 3.45 µm pixels. Precision is the product — treat it as a first-class requirement.

## Environment
- Runs in WSL Ubuntu 22.04; deploy target is Raspberry Pi OS. Python 3.10 (do not use 3.11+ only features such as `tomllib` without a fallback).
- Virtual env: `.venv/` in the project root. Activate with `source .venv/bin/activate`.
- Install: `pip install -e ".[dev]"`

## Commands
- Unit tests: `pytest -m unit`
- Integration tests: `pytest -m integration` (add `--keep-output` to keep artifacts under `outputs/integration/`)
- All tests: `pytest`
- Demo: `PYTHONPATH=src python3 scripts/demo_app.py`

## Layout
- `src/autocollimator/app.py` — AutocollimatorApp: lifecycle, config access, user-facing API
- `src/autocollimator/workflow.py` — orchestration (center finding → MeasurementOutput)
- `src/autocollimator/target_utils.py` — image processing: SDRM center finding, find_cross_center, drawing
- `src/autocollimator/config/models.py` — frozen dataclasses + validation (`from_dict`)
- `src/autocollimator/config/store.py` — TOML config library loading/CRUD, measurement YAML output
- `configs/main.toml` + `configs/library/*.toml` — device config referencing library entries by string id
- `tests/unit`, `tests/integration`, `tests/assets/images` (test images), `tests/assets/expected`
- `test_scripts/` — old experiments; not part of the package

## Rules
- Work in small steps. One logical change per commit. Run `pytest -m unit` before saying a step is done, and `pytest` before finishing a phase.
- Never modify files in `tests/assets/expected/` or loosen a test assertion to make a test pass. If an expected value seems wrong, stop and ask.
- Never round values in the measurement path (positions, angles, quality ratios). Rounding is only allowed when formatting for display.
- Measurement failures must be explicit: a failed measurement must produce `success=False` and never a success record with null/placeholder values.
- Keep secrets (e.g. `encryption_key`) out of logs and output files.
- Use `logging` (module-level `LOGGER`), never `print`, inside `src/`.
- Keep type hints accurate; return annotations must match what the function actually returns.
- Do not add runtime dependencies without adding them to `pyproject.toml`.
- Do not commit anything under `outputs/`.
- Source files currently use CRLF line endings; preserve the existing ending in files you edit unless the task is line-ending normalization.
- Known review findings are tracked in `docs/code_review_2026-09.md` with IDs R1–R14. Reference those IDs in commit messages.
- Always write `python3` (not `python`) in commands and documentation.

