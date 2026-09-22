"""
Run center finding on every image in a folder and build a browsable HTML report.

Usage (from the project root, with .venv active):
    python scripts/evaluate_images.py
    python scripts/evaluate_images.py --images tests/assets/images --out outputs/evaluation

Try a different crop box without editing the config files:
    python scripts/evaluate_images.py --crop-center 960 540 --crop-size 1400 900

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
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from dataclasses import replace

from autocollimator.app import AutocollimatorApp

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROI_NAMES = ("top", "bottom", "right", "left")  # order of rss_ratio_r0..r3


def load_manifest(image_dir: Path) -> dict[str, dict]:
    path = image_dir / "manifest.yaml"
    if not path.exists():
        return {}
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {e["file"]: e for e in entries if isinstance(e, dict) and "file" in e}


def fmt(v, digits: int = 2) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def apply_overrides(app: AutocollimatorApp, overrides: dict) -> None:
    """Replace the active measurement parameters in memory (config files are not changed)."""
    if not overrides:
        return
    device = app.get_current_config()
    by_id = app.get_library()["measurement_parameters_by_id"]
    by_id[device.measurement_parameters_id] = replace(app.get_measurement_parameters(), **overrides)


def evaluate(image_dir: Path, config_dir: Path, run_dir: Path, overrides: dict) -> list[dict]:
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
            "Using crop_center=(%d, %d) crop_size=(%d, %d) roi_size=(%d, %d)",
            mp.crop_center_x, mp.crop_center_y, mp.crop_size_x, mp.crop_size_y, mp.roi_size_x, mp.roi_size_y,
        )
        sensor = app.get_image_sensor()
        for path in images:
            row: dict = {"file": path.name, "expected": manifest.get(path.name, {}).get("expect", "")}
            row["notes"] = manifest.get(path.name, {}).get("notes", "")
            img = cv2.imread(str(path))
            if img is None:
                row.update(status="unreadable")
                rows.append(row)
                continue

            h, w = img.shape[:2]
            row["size"] = f"{w}x{h}"
            row["size_ok"] = (w, h) == (sensor.pixels_x, sensor.pixels_y)

            t0 = time.perf_counter()
            try:
                result, overlay = app.run_center_finding_on_image(
                    img, debug=True, debug_prefix=debug_dir / path.stem
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
            q = [mv.rss_ratio_r0, mv.rss_ratio_r1, mv.rss_ratio_r2, mv.rss_ratio_r3]
            row["q"] = dict(zip(ROI_NAMES, q))
            row["message"] = mv.message

            overlay_name = f"{path.stem}.jpg"
            cv2.imwrite(str(overlay_dir / overlay_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            row["overlay"] = f"overlays/{overlay_name}"
            rows.append(row)

    for row in rows:
        exp = row.get("expected")
        row["match"] = "" if not exp else ("yes" if exp == row.get("status") else "NO")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["file", "size", "status", "reported_success", "expected", "match", "x", "y",
              "a0", "a1", "q_top", "q_bottom", "q_right", "q_left", "ms", "message", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            out = dict(r)
            for name in ROI_NAMES:
                out[f"q_{name}"] = (r.get("q") or {}).get(name)
            wr.writerow(out)


def write_html(rows: list[dict], path: Path, title: str, q_limit: float) -> None:
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
            size_cell += ' <span class="warn" title="Does not match configured sensor">⚠</span>'
        flag = ' <span class="warn" title="App reported success=True (R1 bug)">R1</span>' \
            if status == "failure" and r.get("reported_success") is True else ""
        body.append(
            f'<tr class="{cls}"><td>{thumb}</td><td>{html.escape(r["file"])}</td><td>{size_cell}</td>'
            f"<td><b>{status}</b>{flag}</td><td>{html.escape(str(r.get('expected', '')))}</td>"
            f"<td>{r.get('match', '')}</td><td>{fmt(r.get('x'))}</td><td>{fmt(r.get('y'))}</td>"
            f"<td>{fmt(r.get('a0'), 3)}</td><td>{fmt(r.get('a1'), 3)}</td>{q_cells(r)}"
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
</style></head><body>
<h1>{html.escape(title)}</h1>
<p><b>{n_ok}/{len(rows)}</b> centers found &middot; <b>{n_mismatch}</b> mismatches vs manifest
 &middot; <b>{n_lie}</b> failures reported as success (R1) &middot; quality limit q &gt; {q_limit} shaded red.
 Click a thumbnail for the full overlay (blue = crop, yellow = ROIs, green = ROI centers, red = crosshair center).</p>
<table><tr><th>Overlay</th><th>File</th><th>Size</th><th>Result</th><th>Expected</th><th>Match</th>
<th>x</th><th>y</th><th>a0</th><th>a1</th><th>q top</th><th>q bottom</th><th>q right</th><th>q left</th>
<th>ms</th><th>Message / notes</th></tr>
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
    args = ap.parse_args()

    overrides: dict = {}
    if args.crop_center:
        overrides.update(crop_center_x=args.crop_center[0], crop_center_y=args.crop_center[1])
    if args.crop_size:
        overrides.update(crop_size_x=args.crop_size[0], crop_size_y=args.crop_size[1])
    if args.roi_thickness:
        overrides.update(roi_size_y=args.roi_thickness)

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).setLevel(logging.WARNING)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.out / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with AutocollimatorApp(config_dir=args.config, output_dir=run_dir) as app:
        q_limit = app.get_quality_limits().rss_ratio

    rows = evaluate(args.images, args.config, run_dir, overrides)
    write_csv(rows, run_dir / "results.csv")
    write_html(rows, run_dir / "index.html", f"Center-finding evaluation {stamp}", q_limit)

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
