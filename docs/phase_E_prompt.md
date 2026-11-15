# Phase E — Portable integration test (R8, R9) and leftovers

Fix R8 and R9 from docs/code_review_2026-09.md, plus two leftovers. Work in the order below and stop where indicated.

## 1. R8/R9 — integration test
- Rewrite tests/integration/test_center_finding.py to read tests/assets/images/manifest.yaml.
  Manifest fields per entry: file, expect (success | failure), optional expected_x, expected_y, notes.
- No file paths and no exact float strings in expected data. Compare positions numerically with a 0.2 px tolerance.
- Fix the missing q_limit in the refinement call (the cause of R8).
- Images in tests/assets/images that are not in the manifest are reported, not failed.
- Write scripts/regenerate_expected.py that runs the default configuration on every image and writes a proposed manifest to outputs/manifest_proposed.yaml (it must not overwrite tests/assets/images/manifest.yaml).
- Do NOT generate the manifest values yourself and do not commit tests/assets/images/manifest.yaml. STOP after this step and tell me to run the script and review its output.

## 2. Remove rounding in line_intersection (leftover from R2)
Only after I confirm step 1:
- Remove round(..., 2) in line_intersection.
- Update the tests that pin default results (e.g. blob-22 default (883.72, 566.83), test_signal_quality_defaults.py, the blob-22 deskew+subpixel pin) to the new unrounded values, compared with a 0.01 px tolerance instead of exact equality. List old → new values in the commit message. I authorize only these pin updates; do not loosen any other tolerance or change any other expected value.

## 3. not_run for failed pass 1
When pass 1 fails, record roi_method_m0..m3 as "not_run" instead of "average". Old measurement files must still load.

Commit each step separately with the prefix "E:". Use python3. At the end, the full pytest must pass completely (R8 fixed), except tests that depend on the manifest I have not committed yet.
