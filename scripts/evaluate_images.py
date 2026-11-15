"""
Run center finding on every image in a folder and build a browsable HTML report.

Usage (from the project root, with .venv active):
    python3 scripts/evaluate_images.py
    python3 scripts/evaluate_images.py --images tests/assets/images --out outputs/evaluation

Try a different crop box without editing the config files:
    python3 scripts/evaluate_images.py --crop-center 960 540 --crop-size 1400 900

Turn on sub-pixel centering (R2) without editing the config files:
    python3 scripts/evaluate_images.py --subpixel --subpixel-neighbors 2

Use the two-pass deskew profile (combinable with --subpixel):
    python3 scripts/evaluate_images.py --profile-mode deskew --oversample-step 0.1 --fine-window 40 --subpixel
    python3 scripts/evaluate_images.py --profile-mode deskew --subpixel --refine-search 2.5

Reduce BGR to the green channel or the per-pixel channel max instead of luma (R18):
    python3 scripts/evaluate_images.py --gray-mode green

Find the crosshair anywhere in the image first (pre-search, R4) and measure in the recommended crop:
    python3 scripts/evaluate_images.py --presearch-mode every_image
    python3 scripts/evaluate_images.py --presearch-mode every_image --presearch-downsample 4 --presearch-failure-action fail

Report tilt in arcsec and rotation/orthogonality in degrees (defaults come from main.toml):
    python3 scripts/evaluate_images.py --tilt-unit arcsec --rotation-unit deg

Open the report in your Windows browser:
    explorer.exe "$(wslpath -w outputs/evaluation/latest/index.html)"

If tests/assets/images/manifest.yaml exists, each image's result is compared with its
"expect: success|failure" entry and mismatches are highlighted.
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from dataclasses import replace

from autocollimator.app import AutocollimatorApp
from autocollimator.config.models import PRESEARCH_FAILURE_ACTIONS, PRESEARCH_MODES
from autocollimator.config.store import ConfigError
from autocollimator.target_utils import SATURATION_LEVEL, SATURATION_WARN_FRACTION
from autocollimator.units import AngleUnit, OutputUnits

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROI_NAMES = ("top", "bottom", "right", "left")  # order of rss_ratio_r0..r3
ANGULAR_FIELDS = ("tilt_x", "tilt_y", "rotation", "orthogonality")


def angular_labels(units: OutputUnits) -> dict[str, str]:
    """Column header with unit symbol for each angular field, e.g. 'tilt_x [″]'."""
    return {
        "tilt_x": f"tilt_x [{units.tilt_unit.symbol}]",
        "tilt_y": f"tilt_y [{units.tilt_unit.symbol}]",
        "rotation": f"rotation [{units.rotation_unit.symbol}]",
        "orthogonality": f"orthogonality [{units.rotation_unit.symbol}]",
    }


def load_manifest(image_dir: Path) -> dict[str, dict]:
    path = image_dir / "manifest.yaml"
    if not path.exists():
        return {}
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {e["file"]: e for e in entries if isinstance(e, dict) and "file" in e}


def fmt(v, digits: int = 2) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def mode_label(mp) -> str:
    """Profile mode and SDRM read-out, e.g. 'deskew / subpixel n=2' or 'average / integer'."""
    readout = f"subpixel n={mp.subpixel_neighbors}" if mp.subpixel else "integer"
    label = f"{mp.profile_mode} / {readout} / gray {mp.gray_mode}"
    if mp.profile_mode == "deskew":
        window = f">= {mp.fine_window} (adaptive)" if mp.fine_window_mode == "adaptive" else f"{mp.fine_window}"
        label += (f" (step {mp.oversample_step:g}, window {window}, search {mp.refine_search:g},"
                  f" max shift {mp.max_refine_shift:g})")
    ps = mp.presearch
    if ps.presearch_mode != "off":
        label += (f" / presearch {ps.presearch_mode} (downsample {ps.presearch_downsample},"
                  f" fraction {ps.presearch_crop_fraction:g}, on failure {ps.presearch_failure_action})")
    return label


def presearch_columns(record) -> dict:
    """Report columns for the pre-search section of a measurement (R4)."""
    pr = record.result if record is not None else None
    cols = {
        "ps_ran": None if record is None else record.ran,
        "crop_source": "" if record is None else (record.crop_source or "none"),
        "ps_success": None, "ps_reason": "", "ps_method": "", "ps_center_x": None, "ps_center_y": None,
        "ps_size": None, "ps_warnings": "",
    }
    if pr is not None:
        cols.update(
            ps_success=pr.success, ps_reason=pr.failure_reason or "", ps_method=pr.method_used or "",
            ps_center_x=pr.recommended_center_x, ps_center_y=pr.recommended_center_y,
            ps_size=pr.recommended_size, ps_warnings="; ".join(pr.warnings),
        )
    return cols


def apply_overrides(app: AutocollimatorApp, overrides: dict) -> None:
    """
    Replace the active measurement parameters in memory (config files are not changed).

    ``overrides["presearch"]``, if present, is a dict of PresearchParameters fields.
    """
    if not overrides:
        return
    device = app.get_current_config()
    by_id = app.get_library()["measurement_parameters_by_id"]
    mp = app.get_measurement_parameters()
    fields = {k: v for k, v in overrides.items() if k != "presearch"}
    if overrides.get("presearch"):
        fields["presearch"] = replace(mp.presearch, **overrides["presearch"])
    by_id[device.measurement_parameters_id] = replace(mp, **fields)


def evaluate(
    image_dir: Path, config_dir: Path, run_dir: Path, overrides: dict, units: dict
) -> tuple[list[dict], OutputUnits]:
    """Run every image; ``units`` holds optional tilt_unit / rotation_unit overrides."""
    overlay_dir = run_dir / "overlays"
    debug_dir = run_dir / "debug"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(image_dir)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and ":" not in p.name)
    rows: list[dict] = []

    with AutocollimatorApp(config_dir=config_dir, output_dir=run_dir) as app:
        apply_overrides(app, overrides)
        mp = app.get_measurement_parameters()
        logging.getLogger(__name__).warning(
            "Using crop_center=(%d, %d) crop_size=(%d, %d) roi_size=(%d, %d) subpixel=%s subpixel_neighbors=%d "
            "profile_mode=%s oversample_step=%g fine_window=%d fine_window_mode=%s refine_search=%g max_refine_shift=%g "
            "gray_mode=%s",
            mp.crop_center_x, mp.crop_center_y, mp.crop_size_x, mp.crop_size_y, mp.roi_size_x, mp.roi_size_y,
            mp.subpixel, mp.subpixel_neighbors, mp.profile_mode, mp.oversample_step, mp.fine_window,
            mp.fine_window_mode, mp.refine_search, mp.max_refine_shift, mp.gray_mode,
        )
        mode = mode_label(mp)
        capture_mode = app.get_capture_mode()
        output_units = app.get_output_units(**units)
        for path in images:
            row: dict = {"file": path.name, "expected": manifest.get(path.name, {}).get("expect", ""), "mode": mode}
            row["notes"] = manifest.get(path.name, {}).get("notes", "")
            img = cv2.imread(str(path))
            if img is None:
                row.update(status="unreadable")
                rows.append(row)
                continue

            h, w = img.shape[:2]
            row["size"] = f"{w}x{h}"
            row["size_ok"] = (w, h) == (capture_mode.output_width, capture_mode.output_height)

            t0 = time.perf_counter()
            try:
                result, overlay = app.run_center_finding_on_image(
                    img, debug=True, debug_prefix=debug_dir / path.stem, **units
                )
            except Exception as exc:  # report and keep going
                row.update(status="error", message=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                continue
            row["ms"] = round(1000 * (time.perf_counter() - t0))

            mv = result.measured_values
            # Judge success by the position itself, so this works before and after the R1 fix.
            found = mv.center_position_x is not None and mv.center_position_y is not None
            row["status"] = "success" if found else "failure"
            row["reported_success"] = mv.success
            row["x"], row["y"] = mv.center_position_x, mv.center_position_y
            row["a0"], row["a1"] = mv.measured_angle_a0, mv.measured_angle_a1
            am = result.angular_measurement
            for name in ANGULAR_FIELDS:
                row[name] = getattr(am, name)
            q = [mv.rss_ratio_r0, mv.rss_ratio_r1, mv.rss_ratio_r2, mv.rss_ratio_r3]
            row["q"] = dict(zip(ROI_NAMES, q))
            methods = [mv.roi_method_m0, mv.roi_method_m1, mv.roi_method_m2, mv.roi_method_m3]
            sat = [mv.saturation_s0, mv.saturation_s1, mv.saturation_s2, mv.saturation_s3]
            row["sat"] = dict(zip(ROI_NAMES, sat))
            win = [mv.fine_window_used_f0, mv.fine_window_used_f1, mv.fine_window_used_f2, mv.fine_window_used_f3]
            row["win"] = dict(zip(ROI_NAMES, win))
            row["methods"] = " ".join(f"{name}:{m}" for name, m in zip(ROI_NAMES, methods))
            row["message"] = mv.message
            row.update(presearch_columns(result.presearch))

            overlay_name = f"{path.stem}.jpg"
            cv2.imwrite(str(overlay_dir / overlay_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            row["overlay"] = f"overlays/{overlay_name}"
            rows.append(row)

    for row in rows:
        exp = row.get("expected")
        row["match"] = "" if not exp else ("yes" if exp == row.get("status") else "NO")
    return rows, output_units


def write_csv(rows: list[dict], path: Path, units: OutputUnits) -> None:
    labels = angular_labels(units)
    fields = ["file", "size", "status", "reported_success", "expected", "match", "x", "y",
              *labels.values(), "a0", "a1", "q_top", "q_bottom", "q_right", "q_left",
              "sat_top", "sat_bottom", "sat_right", "sat_left",
              "win_top", "win_bottom", "win_right", "win_left", "crop_source", "ps_ran", "ps_success",
              "ps_method", "ps_center_x", "ps_center_y", "ps_size", "ps_reason", "ps_warnings",
              "mode", "methods", "ms", "message", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            out = dict(r)
            for name in ROI_NAMES:
                out[f"q_{name}"] = (r.get("q") or {}).get(name)
                out[f"sat_{name}"] = (r.get("sat") or {}).get(name)
                out[f"win_{name}"] = (r.get("win") or {}).get(name)
            for name, label in labels.items():
                out[label] = r.get(name)
            wr.writerow(out)


def write_html(rows: list[dict], path: Path, title: str, q_limit: float, units: OutputUnits) -> None:
    labels = angular_labels(units)
    # Display precision only; the CSV keeps full precision.
    digits = {"tilt_x": 4, "tilt_y": 4, "rotation": 4, "orthogonality": 4}
    n_ok = sum(r.get("status") == "success" for r in rows)
    n_mismatch = sum(r.get("match") == "NO" for r in rows)
    n_lie = sum(r.get("status") == "failure" and r.get("reported_success") is True for r in rows)

    def q_cells(r: dict) -> str:
        cells = []
        for name in ROI_NAMES:
            v = (r.get("q") or {}).get(name)
            bad = v is not None and v > q_limit
            cells.append(f'<td class="{"bad" if bad else ""}">{fmt(v, 3)}</td>')
        return "".join(cells)

    def sat_cell(r: dict) -> str:
        sat = {k: v for k, v in (r.get("sat") or {}).items() if v is not None}
        if not sat:
            return "<td></td>"
        worst = max(sat.values())
        tip = ", ".join(f"{name} {100 * v:.2f} %" for name, v in sat.items())
        cls = "bad" if worst > SATURATION_WARN_FRACTION else ""
        return f'<td class="{cls}" title="{html.escape(tip)}">{fmt(100 * worst)}</td>'

    def presearch_cells(r: dict) -> str:
        if not r.get("ps_ran"):
            ps = "not run"
        else:
            ps = "ok" if r.get("ps_success") else "FAILED"
        cls = "bad" if r.get("ps_ran") and not r.get("ps_success") else ""
        center = (f"{r['ps_center_x']}, {r['ps_center_y']}" if r.get("ps_center_x") is not None else "")
        details = "; ".join(s for s in (r.get("ps_reason", ""), r.get("ps_warnings", "")) if s)
        return (f'<td class="{cls}">{ps}</td><td>{html.escape(r.get("ps_method", ""))}</td><td>{center}</td>'
                f'<td>{r.get("ps_size") or ""}</td><td>{html.escape(str(r.get("crop_source", "")))}</td>'
                f'<td class="msg">{html.escape(details)}</td>')

    body = []
    for r in rows:
        status = r.get("status", "")
        cls = {"success": "ok", "failure": "fail"}.get(status, "err")
        if r.get("match") == "NO":
            cls += " mismatch"
        thumb = (f'<a href="{r["overlay"]}" target="_blank"><img src="{r["overlay"]}" loading="lazy"></a>'
                 if r.get("overlay") else "")
        size_cell = html.escape(r.get("size", ""))
        if r.get("size") and not r.get("size_ok"):
            size_cell += ' <span class="warn" title="Does not match the capture mode output size">⚠</span>'
        flag = ' <span class="warn" title="App reported success=True (R1 bug)">R1</span>' \
            if status == "failure" and r.get("reported_success") is True else ""
        body.append(
            f'<tr class="{cls}"><td>{thumb}</td><td>{html.escape(r["file"])}</td><td>{size_cell}</td>'
            f"<td><b>{status}</b>{flag}</td><td>{html.escape(str(r.get('expected', '')))}</td>"
            f"<td>{r.get('match', '')}</td><td>{fmt(r.get('x'))}</td><td>{fmt(r.get('y'))}</td>"
            + "".join(f"<td>{fmt(r.get(name), digits[name])}</td>" for name in ANGULAR_FIELDS) +
            f"<td>{fmt(r.get('a0'), 3)}</td><td>{fmt(r.get('a1'), 3)}</td>{q_cells(r)}{sat_cell(r)}{presearch_cells(r)}"
            f"<td>{html.escape(r.get('mode', ''))}</td><td class='methods'>{html.escape(r.get('methods', ''))}</td>"
            f"<td>{r.get('ms', '')}</td><td class='msg'>{html.escape(str(r.get('message', '')))}"
            f"<br><i>{html.escape(str(r.get('notes', '')))}</i></td></tr>"
        )

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 1.5rem; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 6px; vertical-align: middle; }}
 th {{ background: #f0f0f0; position: sticky; top: 0; }}
 img {{ width: 240px; display: block; }}
 tr.ok td:nth-child(4) {{ color: #137a2a; }}
 tr.fail td:nth-child(4), tr.err td:nth-child(4) {{ color: #b3261e; }}
 tr.mismatch {{ outline: 3px solid #e69500; }}
 td.bad {{ background: #fde2e1; }}
 .warn {{ color: #b36b00; font-weight: bold; }}
 .msg {{ max-width: 280px; }}
 .methods {{ max-width: 140px; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p><b>{n_ok}/{len(rows)}</b> centers found &middot; <b>{n_mismatch}</b> mismatches vs manifest
 &middot; <b>{n_lie}</b> failures reported as success (R1) &middot; quality limit q &gt; {q_limit} shaded red
 &middot; sat = max saturated-pixel % over the ROIs (any channel &gt;= {SATURATION_LEVEL}), shaded above
 {100 * SATURATION_WARN_FRACTION:g} %, hover for per-ROI values.
 Click a thumbnail for the full overlay (blue = crop, yellow = ROIs, green = ROI centers, red = crosshair center,
 orange = pre-search recommended region).</p>
<table><tr><th>Overlay</th><th>File</th><th>Size</th><th>Result</th><th>Expected</th><th>Match</th>
<th>x</th><th>y</th>{''.join(f'<th>{html.escape(label)}</th>' for label in labels.values())}<th>a0</th><th>a1</th><th>q top</th><th>q bottom</th><th>q right</th><th>q left</th>
<th>sat max %</th><th>Pre-search</th><th>PS method</th><th>PS center</th><th>PS size</th><th>Crop source</th>
<th>PS reason / warnings</th><th>Mode</th><th>ROI methods</th><th>ms</th><th>Message / notes</th></tr>
{''.join(body)}
</table></body></html>"""
    path.write_text(page, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("tests/assets/images"))
    ap.add_argument("--config", type=Path, default=Path("configs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/evaluation"))
    ap.add_argument("--crop-center", type=int, nargs=2, metavar=("X", "Y"),
                    help="Crop box center in pixels (overrides measurement_parameters.toml)")
    ap.add_argument("--crop-size", type=int, nargs=2, metavar=("W", "H"),
                    help="Crop box width and height in pixels")
    ap.add_argument("--roi-thickness", type=int, metavar="T",
                    help="Thickness of the four ROI strips in pixels (roi_size_y)")
    ap.add_argument("--subpixel", action="store_true",
                    help="Refine SDRM offsets to sub-pixel precision (measurement_parameters.subpixel)")
    ap.add_argument("--subpixel-neighbors", type=int, metavar="N",
                    help="Points on each side of the minimum for the sub-pixel fit, 1-3 (used with --subpixel)")
    ap.add_argument("--profile-mode", choices=("average", "deskew"),
                    help="How each ROI profile is built (measurement_parameters.profile_mode)")
    ap.add_argument("--oversample-step", type=float, metavar="PX",
                    help="Bin width of the deskewed profile in px, 0.02-0.5 (used with --profile-mode deskew)")
    ap.add_argument("--fine-window", type=int, metavar="PX",
                    help="Half-width of the deskewed profile in px; a minimum with --fine-window-mode adaptive "
                         "(used with --profile-mode deskew)")
    ap.add_argument("--fine-window-mode", choices=("fixed", "adaptive"),
                    help="fixed: use --fine-window as is; adaptive: grow it to 1.5 x line FWHM per ROI "
                         "(measurement_parameters.fine_window_mode, used with --profile-mode deskew)")
    ap.add_argument("--refine-search", type=float, metavar="PX",
                    help="Pass-2 search half-range in px, 0.5-3.0 (used with --profile-mode deskew)")
    ap.add_argument("--max-refine-shift", type=float, metavar="PX",
                    help="Max pass-2 shift from pass 1 in px before falling back, 0.1-3.0 (deskew only)")
    ap.add_argument("--gray-mode", choices=("luma", "green", "max"),
                    help="How BGR is reduced to one channel (measurement_parameters.gray_mode)")
    ap.add_argument("--presearch-mode", choices=PRESEARCH_MODES,
                    help="off: configured crop; every_image: pre-search each image and use its crop; "
                         "track: previous center, pre-search when needed (measurement_parameters.presearch)")
    ap.add_argument("--presearch-failure-action", choices=PRESEARCH_FAILURE_ACTIONS,
                    help="What a measurement does when the pre-search fails")
    ap.add_argument("--presearch-downsample", type=int, metavar="N", help="Pre-search downsample factor, int >= 1")
    ap.add_argument("--presearch-crop-fraction", type=float, metavar="F",
                    help="Recommended crop size as a fraction of min(image width, height), 0.1-1.0")
    ap.add_argument("--min-crop-size", type=int, metavar="PX", help="Smallest recommended crop size (min_crop_size_px)")
    ap.add_argument("--presearch-q-limit", type=float, metavar="Q", help="Pre-search RSS-ratio limit, 0 < Q < 1")
    ap.add_argument("--presearch-agreement", type=float, metavar="PX",
                    help="Max distance between pre-search and validation center (presearch_agreement_px)")
    ap.add_argument("--presearch-ambiguity-ratio", type=float, metavar="R",
                    help="Warn when a second candidate line reaches this fraction of the strongest")
    unit_choices = [u.value for u in AngleUnit]
    ap.add_argument("--tilt-unit", choices=unit_choices,
                    help="Unit for tilt_x / tilt_y (default: device.tilt_unit in main.toml)")
    ap.add_argument("--rotation-unit", choices=unit_choices,
                    help="Unit for rotation / orthogonality (default: device.rotation_unit in main.toml)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.crop_center:
        overrides.update(crop_center_x=args.crop_center[0], crop_center_y=args.crop_center[1])
    if args.crop_size:
        overrides.update(crop_size_x=args.crop_size[0], crop_size_y=args.crop_size[1])
    if args.roi_thickness:
        overrides.update(roi_size_y=args.roi_thickness)
    if args.subpixel:
        overrides.update(subpixel=True)
    if args.subpixel_neighbors is not None:
        overrides.update(subpixel_neighbors=args.subpixel_neighbors)
    if args.profile_mode:
        overrides.update(profile_mode=args.profile_mode)
    if args.oversample_step is not None:
        overrides.update(oversample_step=args.oversample_step)
    if args.fine_window is not None:
        overrides.update(fine_window=args.fine_window)
    if args.fine_window_mode:
        overrides.update(fine_window_mode=args.fine_window_mode)
    if args.refine_search is not None:
        overrides.update(refine_search=args.refine_search)
    if args.max_refine_shift is not None:
        overrides.update(max_refine_shift=args.max_refine_shift)
    if args.gray_mode:
        overrides.update(gray_mode=args.gray_mode)
    presearch_overrides = {
        key: value
        for key, value in (
            ("presearch_mode", args.presearch_mode),
            ("presearch_failure_action", args.presearch_failure_action),
            ("presearch_downsample", args.presearch_downsample),
            ("presearch_crop_fraction", args.presearch_crop_fraction),
            ("min_crop_size_px", args.min_crop_size),
            ("presearch_q_limit", args.presearch_q_limit),
            ("presearch_agreement_px", args.presearch_agreement),
            ("presearch_ambiguity_ratio", args.presearch_ambiguity_ratio),
        )
        if value is not None
    }
    if presearch_overrides:
        overrides["presearch"] = presearch_overrides

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).setLevel(logging.WARNING)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        with AutocollimatorApp(config_dir=args.config, output_dir=run_dir) as app:
            q_limit = app.get_quality_limits().rss_ratio
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    units = {"tilt_unit": args.tilt_unit, "rotation_unit": args.rotation_unit}
    rows, output_units = evaluate(args.images, args.config, run_dir, overrides, units)
    write_csv(rows, run_dir / "results.csv", output_units)
    write_html(rows, run_dir / "index.html", f"Center-finding evaluation {stamp}", q_limit, output_units)

    latest = args.out / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)

    n_ok = sum(r.get("status") == "success" for r in rows)
    print(f"{n_ok}/{len(rows)} centers found. Report: {latest / 'index.html'}")
    for r in rows:
        print(f"  {r['file']:<40} {r.get('status', ''):<9} {r.get('match', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
