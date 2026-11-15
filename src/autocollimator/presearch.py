"""
Pre-search (R4): find the reticle anywhere in the image and recommend a search region (crop
center and size) for the measurement.

This is an image-layer routine, separate from the measurement: it knows nothing about the
instrument, writes no config and does no I/O. It only reports; the caller decides what to do
with a failed result. Selected by reticle type (``_PRESEARCH_BY_RETICLE``).

Crosshair pre-search, on the image downsampled by ``presearch_downsample``:

1. Candidates: column / row projection profiles, background-subtracted; each peak above the
   noise is a candidate line. The strongest per line is used; a second one reaching
   ``presearch_ambiguity_ratio`` of its strength is warned about (e.g. a ghost, R19).
2. Main method "sdrm_full_frame": the existing SDRM centering (``find_center_pixel``) on four
   ROI strips with the crop = whole image (strips across the full width at the top and bottom
   edges, across the full height at the left and right edges); center = intersection of the
   two lines, as in ``find_cross_center_detailed``.
3. Checks: both lines found, all quality ratios below ``presearch_q_limit``, the intersection
   inside the image, a valid region (``recommend_region``), and a validation pass
   (``find_cross_center_detailed`` in the recommended region, same downsampled image) that
   agrees within ``presearch_agreement_px``.

All reported positions and sizes are in full-resolution pixels.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from autocollimator.config.models import PresearchCandidate, PresearchParameters, PresearchResult
from autocollimator.instrument import ReticleType
from autocollimator.target_utils import (
    ROI_NAMES,
    _fwhm_over_background,
    _half_crossing,
    _roi_point,
    ensure_gray_f64,
    find_center_pixel,
    find_cross_center_detailed,
    line_intersection,
)

__all__ = ["PresearchResult", "RecommendedRegion", "presearch", "recommend_region"]

LOGGER = logging.getLogger(__name__)

# After shifting the region inside the image, the crosshair must lie at least this fraction of
# the region size from every region edge (in its middle half); otherwise the region shrinks.
PRESEARCH_CENTER_MARGIN_FRACTION = 0.25
# The region is never smaller than this many line FWHMs.
PRESEARCH_MIN_FWHM_MULTIPLE = 4.0
# A projection peak is a candidate line when it is above both PRESEARCH_MIN_PEAK_SNR x the
# profile noise (robust, from the MAD) and PRESEARCH_MIN_PEAK_LEVEL gray levels.
PRESEARCH_MIN_PEAK_SNR = 8.0
PRESEARCH_MIN_PEAK_LEVEL = 2.0
# Profile background: running median over this fraction of the profile length.
PRESEARCH_BACKGROUND_FRACTION = 0.25
# Box smoothing of the background-subtracted profile before the peak search (downsampled px).
PRESEARCH_SMOOTH_PX = 5
# At most this many candidates per line; peaks closer than max(PRESEARCH_MIN_SEPARATION_PX,
# 2 x FWHM) (downsampled px) to a stronger one are the same line.
PRESEARCH_MAX_CANDIDATES = 5
PRESEARCH_MIN_SEPARATION_PX = 5
# Smallest strip depth and downsampled image side the search works with.
PRESEARCH_MIN_STRIP_DEPTH_PX = 2
PRESEARCH_MIN_IMAGE_PX = 16
# Smallest half-width (downsampled px) of a strip window around the strongest of several candidates.
PRESEARCH_MIN_WINDOW_PX = 10


@dataclass(frozen=True)
class RecommendedRegion:
    """Square search region in full-resolution px; center and size are None on failure."""

    center: tuple[int, int] | None
    size: int | None
    shifted: bool
    shrunk: bool
    failure_reason: str | None


@dataclass(frozen=True)
class _Peak:
    """A candidate line in one projection profile (downsampled px)."""

    position: float
    strength: float
    fwhm: float | None


@dataclass(frozen=True)
class _Estimate:
    """A center estimate of one method, before the checks."""

    method: str
    center: tuple[float, float] | None  # full-resolution px
    q_values: tuple[float, ...]
    failure_reason: str | None


def _to_full(u: float, f: int) -> float:
    """Downsampled pixel coordinate -> full-resolution pixel coordinate (pixel centers)."""
    return float((u + 0.5) * f - 0.5)


def _to_down(x: float, f: int) -> float:
    return (x + 0.5) / f - 0.5


def _downsample(gray: np.ndarray, f: int) -> np.ndarray:
    """Block mean over f x f pixels; rows/columns beyond a multiple of f are dropped."""
    if f == 1:
        return gray
    h, w = gray.shape[0] // f, gray.shape[1] // f
    return gray[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))


def _background(profile: np.ndarray) -> np.ndarray:
    """Running median over PRESEARCH_BACKGROUND_FRACTION of the profile (edge-padded)."""
    n = profile.size
    win = max(3, int(n * PRESEARCH_BACKGROUND_FRACTION) | 1)
    half = win // 2
    padded = np.pad(profile, half, mode="edge")
    return np.median(sliding_window_view(padded, win), axis=1)


def _line_peaks(profile: np.ndarray) -> list[_Peak]:
    """
    Candidate lines in a projection profile, strongest first (downsampled px).

    The profile minus its running-median background is box-smoothed (this also evens out the
    broadened peak of a tilted line); its polarity follows the larger deviation, so dark lines
    are found too. A peak's position is the midpoint of its half-maximum crossings (the
    maximum when a crossing is missing); its FWHM is taken on the unsmoothed profile.
    """
    residual = profile - _background(profile)
    if -residual.min() > residual.max():
        residual = -residual
    kernel = np.ones(PRESEARCH_SMOOTH_PX) / PRESEARCH_SMOOTH_PX
    s = np.convolve(residual, kernel, mode="same")
    noise = 1.4826 * float(np.median(np.abs(s - np.median(s))))
    threshold = max(PRESEARCH_MIN_PEAK_SNR * noise, PRESEARCH_MIN_PEAK_LEVEL)

    peaks: list[_Peak] = []
    work = s.copy()
    separation = PRESEARCH_MIN_SEPARATION_PX
    for _ in range(PRESEARCH_MAX_CANDIDATES):
        i = int(np.argmax(work))
        strength = float(work[i])
        if not strength > threshold:
            break
        # Width of the projected line (unsmoothed; includes the broadening by line tilt).
        lo, hi = max(0, i - PRESEARCH_SMOOTH_PX), min(s.size, i + PRESEARCH_SMOOTH_PX + 1)
        fwhm = _fwhm_over_background(residual, lo, hi, 0.0)
        left = _half_crossing(s, i, 0.5 * s[i], -1)
        right = _half_crossing(s, i, 0.5 * s[i], +1)
        position = 0.5 * (left + right) if left is not None and right is not None else float(i)
        peaks.append(_Peak(position, strength, fwhm))
        if len(peaks) == 1 and fwhm is not None:
            separation = max(separation, int(math.ceil(2.0 * fwhm)))
        work[max(0, i - separation) : i + separation + 1] = -np.inf
    return peaks


def recommend_region(
    position: tuple[float, float],
    image_size: tuple[int, int],
    params: PresearchParameters,
    line_fwhm_px: float | None = None,
) -> RecommendedRegion:
    """
    Square search region for a crosshair at ``position`` in an image of ``image_size`` (w, h).

    Size = presearch_crop_fraction x min(w, h), but at least min_crop_size_px and, when the
    line FWHM is known, PRESEARCH_MIN_FWHM_MULTIPLE x FWHM. Near the image edge the region is
    first shifted inward to lie fully inside the image, keeping the crosshair at least
    PRESEARCH_CENTER_MARGIN_FRACTION x size from every region edge; only when that is not
    possible is the region shrunk, and it fails when it would drop below the minimum size.
    """
    w, h = image_size
    cx, cy = position
    minimum = params.min_crop_size_px
    if line_fwhm_px is not None:
        minimum = max(minimum, int(math.ceil(PRESEARCH_MIN_FWHM_MULTIPLE * line_fwhm_px)))
    desired = max(int(params.presearch_crop_fraction * min(w, h)), minimum)

    k = PRESEARCH_CENTER_MARGIN_FRACTION
    largest = min(float(w), float(h), cx / k, (w - cx) / k, cy / k, (h - cy) / k)
    shrunk = desired > largest
    size = int(math.floor(largest)) if shrunk else desired
    if size < minimum:
        return RecommendedRegion(
            None, None, False, shrunk,
            f"region below minimum size ({size} px < {minimum} px for a crosshair at "
            f"({cx:.1f}, {cy:.1f}) in a {w}x{h} image)",
        )

    x_ideal, y_ideal = int(round(cx - size / 2)), int(round(cy - size / 2))
    x0 = min(max(x_ideal, 0), w - size)
    y0 = min(max(y_ideal, 0), h - size)
    shifted = (x0, y0) != (x_ideal, y_ideal)
    return RecommendedRegion((x0 + size // 2, y0 + size // 2), size, shifted, shrunk, None)


def _strip_window(peaks: list[_Peak], n: int) -> tuple[int, int]:
    """Whole profile for one candidate; around the strongest of several, half-way to the next."""
    if len(peaks) < 2:
        return 0, n
    c = peaks[0].position
    half = max(PRESEARCH_MIN_WINDOW_PX, int(min(abs(p.position - c) for p in peaks[1:]) / 2))
    return max(0, int(c) - half), min(n, int(c) + half + 1)


def _sdrm_full_frame(
    g: np.ndarray, f: int, params: PresearchParameters, depth: int, v_peaks: list[_Peak], h_peaks: list[_Peak]
) -> _Estimate:
    """SDRM on ROI strips spanning the whole (downsampled) image; see the module docstring."""
    h, w = g.shape
    x_lo, x_hi = _strip_window(v_peaks, w)
    y_lo, y_hi = _strip_window(h_peaks, h)
    strips = {  # name: (x0, y0, roi in search orientation), in ROI_NAMES order
        "top": (x_lo, 0, g[0:depth, x_lo:x_hi]),
        "bottom": (x_lo, h - depth, g[h - depth : h, x_lo:x_hi]),
        "right": (w - depth, y_lo, g[y_lo:y_hi, w - depth : w].T),
        "left": (0, y_lo, g[y_lo:y_hi, 0:depth].T),
    }
    q_limit = float(params.presearch_q_limit)
    points: dict[str, tuple[float, float]] = {}
    q_values: list[float] = []
    failed: list[str] = []
    for name in ROI_NAMES:
        x0, y0, roi = strips[name]
        if np.ptp(roi.mean(axis=0)) == 0:
            q_values.append(math.nan)
            failed.append(f"{name} (flat profile)")
            continue
        pos, q = find_center_pixel(
            roi,
            center_position=(roi.shape[1] / 2.0, roi.shape[0] / 2.0),
            search_size=max(1, roi.shape[1] // 2 - 2),
            q_limit=q_limit,
            subpixel=True,
        )
        q_values.append(float(q))
        if pos is None:
            failed.append(f"{name} {q:.3f}")
            continue
        points[name] = _roi_point(name, x0, y0, roi, pos)

    if failed:
        return _Estimate(
            "sdrm_full_frame", None, tuple(q_values),
            f"quality check failed for strip {', '.join(failed)} (presearch_q_limit {q_limit:.3f})",
        )
    try:
        cx, cy = line_intersection((points["top"], points["bottom"]), (points["left"], points["right"]))
    except ValueError as exc:
        return _Estimate("sdrm_full_frame", None, tuple(q_values), f"lines do not intersect: {exc}")
    return _Estimate("sdrm_full_frame", (_to_full(cx, f), _to_full(cy, f)), tuple(q_values), None)


def _validate(
    g: np.ndarray, f: int, params: PresearchParameters, depth: int, region: RecommendedRegion,
    center: tuple[float, float], gray_mode: str,
) -> str | None:
    """One center-finding pass in the region on the downsampled image; failure reason or None."""
    assert region.center is not None and region.size is not None
    size = max(2 * PRESEARCH_MIN_STRIP_DEPTH_PX, region.size // f)
    crop_center = (int(round(_to_down(region.center[0], f))), int(round(_to_down(region.center[1], f))))
    result = find_cross_center_detailed(
        g, crop_center, (size, size), (size, depth), float(params.presearch_q_limit),
        subpixel=True, gray_mode=gray_mode,
    )
    if result.center is None:
        ratios = ", ".join(f"{name} {q:.3f}" for name, q in zip(ROI_NAMES, result.q_ratio_list))
        return f"validation failed: center finding in the recommended region failed the quality check ({ratios})"
    vx, vy = _to_full(result.center[0], f), _to_full(result.center[1], f)
    distance = math.hypot(vx - center[0], vy - center[1])
    if distance > params.presearch_agreement_px:
        return (
            f"validation disagreement {distance:.1f} px (validation ({vx:.1f}, {vy:.1f}) vs pre-search "
            f"({center[0]:.1f}, {center[1]:.1f}); presearch_agreement_px {params.presearch_agreement_px:g})"
        )
    return None


def _check(
    estimate: _Estimate, g: np.ndarray, f: int, params: PresearchParameters, depth: int,
    image_size: tuple[int, int], line_fwhm_px: float | None, gray_mode: str, common: dict,
) -> PresearchResult:
    """Apply the success criteria to one estimate and build the result."""
    def failed(reason: str, **extra) -> PresearchResult:
        return PresearchResult(
            success=False, failure_reason=reason, method_used=estimate.method,
            position_x=None if estimate.center is None else estimate.center[0],
            position_y=None if estimate.center is None else estimate.center[1],
            recommended_center_x=None, recommended_center_y=None, recommended_size=None,
            q_values=estimate.q_values, **common, **extra,
        )

    if estimate.center is None:
        return failed(estimate.failure_reason or "no center")
    w, h = image_size
    cx, cy = estimate.center
    if not (0.0 <= cx <= w - 1 and 0.0 <= cy <= h - 1):
        return failed(f"intersection outside image ({cx:.1f}, {cy:.1f}) for a {w}x{h} image")
    region = recommend_region((cx, cy), image_size, params, line_fwhm_px)
    if region.failure_reason is not None:
        return failed(region.failure_reason, region_shrunk=region.shrunk)
    reason = _validate(g, f, params, depth, region, (cx, cy), gray_mode)
    if reason is not None:
        return failed(reason, region_shifted=region.shifted, region_shrunk=region.shrunk)
    assert region.center is not None
    return PresearchResult(
        success=True, failure_reason=None, method_used=estimate.method, position_x=cx, position_y=cy,
        recommended_center_x=region.center[0], recommended_center_y=region.center[1],
        recommended_size=region.size, q_values=estimate.q_values,
        region_shifted=region.shifted, region_shrunk=region.shrunk, **common,
    )


def _ambiguity_warning(line: str, axis: str, peaks: list[_Peak], f: int, ratio: float) -> str | None:
    if len(peaks) < 2 or peaks[1].strength < ratio * peaks[0].strength:
        return None
    a, b = peaks[0], peaks[1]
    return (
        f"ambiguous {line} line: {axis}={_to_full(a.position, f):.1f} (strength {a.strength:.1f}) and "
        f"{axis}={_to_full(b.position, f):.1f} (strength {b.strength:.1f}); using {axis}={_to_full(a.position, f):.1f}"
    )


def _presearch_crosshair(
    image: np.ndarray, params: PresearchParameters, *, gray_mode: str, roi_depth_px: int
) -> PresearchResult:
    f = params.presearch_downsample
    gray = ensure_gray_f64(image, gray_mode)
    h_full, w_full = gray.shape
    g = _downsample(gray, f)
    if min(g.shape) < PRESEARCH_MIN_IMAGE_PX:
        return _failed(f"image {w_full}x{h_full} too small for presearch_downsample {f}", downsample=f)
    depth = max(PRESEARCH_MIN_STRIP_DEPTH_PX, int(round(roi_depth_px / f)))

    v_peaks = _line_peaks(g.mean(axis=0))  # vertical line: column profile, position is x
    h_peaks = _line_peaks(g.mean(axis=1))  # horizontal line: row profile, position is y
    candidates = tuple(
        [PresearchCandidate("vertical", _to_full(p.position, f), p.strength) for p in v_peaks]
        + [PresearchCandidate("horizontal", _to_full(p.position, f), p.strength) for p in h_peaks]
    )
    ratio = params.presearch_ambiguity_ratio
    warnings = [
        msg for msg in (
            _ambiguity_warning("vertical", "x", v_peaks, f, ratio),
            _ambiguity_warning("horizontal", "y", h_peaks, f, ratio),
        ) if msg is not None
    ]
    for msg in warnings:
        LOGGER.warning("Pre-search: %s", msg)
    fwhms = [p[0].fwhm for p in (v_peaks, h_peaks) if p and p[0].fwhm is not None]
    line_fwhm_px = f * float(np.mean(fwhms)) if fwhms else None
    common = dict(line_fwhm_px=line_fwhm_px, candidates=candidates, downsample=f)

    missing = [line for line, peaks in (("vertical", v_peaks), ("horizontal", h_peaks)) if not peaks]
    if missing:
        noun = "line" if len(missing) == 1 else "lines"
        return _failed(f"{' and '.join(missing)} {noun} not found", warnings=tuple(warnings), **common)

    result = _check(
        _sdrm_full_frame(g, f, params, depth, v_peaks, h_peaks), g, f, params, depth,
        (w_full, h_full), line_fwhm_px, gray_mode, dict(common, warnings=tuple(warnings)),
    )
    LOGGER.info(
        "Pre-search %s: method=%s position=(%s, %s) region center=(%s, %s) size=%s reason=%s",
        "succeeded" if result.success else "failed", result.method_used, result.position_x, result.position_y,
        result.recommended_center_x, result.recommended_center_y, result.recommended_size, result.failure_reason,
    )
    return result


def _failed(reason: str, **fields) -> PresearchResult:
    return PresearchResult(
        success=False, failure_reason=reason, method_used=None, position_x=None, position_y=None,
        recommended_center_x=None, recommended_center_y=None, recommended_size=None, **fields,
    )


_PresearchFn = Callable[..., PresearchResult]
_PRESEARCH_BY_RETICLE: dict[ReticleType, _PresearchFn] = {ReticleType.CROSSHAIR: _presearch_crosshair}


def presearch(
    image: np.ndarray,
    params: PresearchParameters,
    reticle_type: ReticleType | str,
    *,
    gray_mode: str = "luma",
    roi_depth_px: int = 20,
) -> PresearchResult:
    """
    Search the whole image for the reticle and recommend a search region (see module docstring).

    ``gray_mode`` reduces a BGR image to one channel (see ``ensure_gray_f64``);
    ``roi_depth_px`` is the strip depth in full-resolution px (the measurement's roi_size_y).
    An unsupported reticle type gives a failed result. Never raises for an image without a
    reticle; raises TypeError / ValueError for an invalid image or argument.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"image must be a non-empty 2D or 3D array, got shape {image.shape}")
    if isinstance(roi_depth_px, bool) or not isinstance(roi_depth_px, int) or roi_depth_px <= 0:
        raise ValueError(f"roi_depth_px must be a positive int, got {roi_depth_px!r}")
    try:
        reticle = ReticleType(reticle_type)
    except ValueError:
        reticle = None
    fn = _PRESEARCH_BY_RETICLE.get(reticle) if reticle is not None else None
    if fn is None:
        name = reticle_type.value if isinstance(reticle_type, ReticleType) else reticle_type
        return _failed(f"reticle type {name!r} is not supported by the pre-search", downsample=params.presearch_downsample)
    return fn(image, params, gray_mode=gray_mode, roi_depth_px=roi_depth_px)
