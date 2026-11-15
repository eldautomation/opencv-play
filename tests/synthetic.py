"""Synthetic crosshair images shared by the tests (deskew C2, pre-search R4)."""
from __future__ import annotations

import numpy as np

IMAGE_SHAPE = (1088, 1456)  # (h, w): configured sensor size
TRUE_CENTER = (872.0, 544.0)  # (x, y) for shift 0: the default crop center
BACKGROUND = 30.0
PEAK = 210.0
SUPERSAMPLE = 5


def render_crosshair(
    fwhm: float,
    tilt_deg: float,
    shift: float,
    *,
    shape: tuple[int, int] = IMAGE_SHAPE,
    center: tuple[float, float] = TRUE_CENTER,
    background: float = BACKGROUND,
    peak: float = PEAK,
) -> np.ndarray:
    """
    Synthetic crosshair: two Gaussian lines (FWHM ``fwhm`` px) rotated by ``tilt_deg``,
    crossing at ``center`` + (shift, shift), on ``background`` with ``peak`` on the lines.
    ``shape`` is (h, w).

    Pixel (row r, col c) has its center at (x=c, y=r) and is the mean of a SUPERSAMPLE x
    SUPERSAMPLE grid of point samples. Only strips around the lines are computed; outside
    them (> 10 sigma from both lines) the image is ``background``.
    """
    h, w = shape
    cx, cy = center[0] + shift, center[1] + shift
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.radians(tilt_deg)
    c, s = np.cos(t), np.sin(t)
    sub = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
    half = int(np.ceil(max(h, w) * abs(np.tan(t)) + 10 * sigma)) + 2

    img = np.full(shape, background, dtype=np.float64)

    def fill(r0: int, r1: int, c0: int, c1: int) -> None:
        if r1 <= r0 or c1 <= c0:
            return
        ys = (np.arange(r0, r1)[:, None] + sub[None, :]).ravel()[:, None] - cy
        xs = (np.arange(c0, c1)[:, None] + sub[None, :]).ravel()[None, :] - cx
        g_v = np.exp(-0.5 * ((xs * c - ys * s) / sigma) ** 2)
        g_h = np.exp(-0.5 * ((xs * s + ys * c) / sigma) ** 2)
        g = np.maximum(g_v, g_h).reshape(r1 - r0, SUPERSAMPLE, c1 - c0, SUPERSAMPLE).mean(axis=(1, 3))
        img[r0:r1, c0:c1] = background + (peak - background) * g

    xc, yc = int(round(cx)), int(round(cy))
    fill(0, h, max(0, xc - half), min(w, xc + half + 1))  # vertical line strip
    fill(max(0, yc - half), min(h, yc + half + 1), 0, w)  # horizontal line strip
    return img


def render_with_ghost(
    fwhm: float,
    center: tuple[float, float],
    ghost_center: tuple[float, float],
    ghost_ratio: float,
    *,
    shape: tuple[int, int] = IMAGE_SHAPE,
    tilt_deg: float = 0.0,
) -> np.ndarray:
    """A crosshair at ``center`` plus a weaker copy (``ghost_ratio`` x contrast) at ``ghost_center``."""
    main = render_crosshair(fwhm, tilt_deg, 0.0, shape=shape, center=center)
    ghost = render_crosshair(fwhm, tilt_deg, 0.0, shape=shape, center=ghost_center)
    return np.maximum(main, BACKGROUND + ghost_ratio * (ghost - BACKGROUND))
