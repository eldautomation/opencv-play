"""
functions.py

Computer-vision utilities for crosshair / line-center detection.

Edits applied:
- Consistent type hints and input validation
- Improved readability and docstrings
- Replaced ad-hoc prints with logging-based debug output
- Fixed several logic errors (undefined helpers, incorrect ndim checks, uninitialized returns)
- Added safer ROI clipping and angle computation (atan2)
- Prevented side effects on import (moved test harness under __main__)
"""

from __future__ import annotations


import logging
import math
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import statistics
import hashlib

from dataclasses import dataclass

from pathlib import Path
from typing import Iterable, Sequence

from autocollimator.config.models import (
    FINE_WINDOW_MODES,
    GRAY_MODES,
    MAX_REFINE_SHIFT_RANGE,
    OVERSAMPLE_STEP_RANGE,
    PROFILE_MODES,
    REFINE_SEARCH_RANGE,
)

LOGGER = logging.getLogger(__name__)

# Order of the ROIs measured by find_cross_center (and of its q_ratio_list).
ROI_NAMES: tuple[str, str, str, str] = ("top", "bottom", "right", "left")


# ----------------------------
# Filesystem helpers
# ----------------------------
def clear_folder(folder_path: str | os.PathLike[str], *, create: bool = False) -> None:
    """
    Delete all files and subdirectories inside ``folder_path`` (the folder itself is preserved).

    Parameters
    ----------
    folder_path:
        Path to the folder whose contents should be deleted.
    create:
        If True, create the folder if it does not exist.

    Raises
    ------
    FileNotFoundError:
        If the folder does not exist and ``create`` is False.
    NotADirectoryError:
        If the path exists but is not a directory.
    """
    path = Path(folder_path)

    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            return
        raise FileNotFoundError(f"Folder does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {path}")

    for entry in path.iterdir():
        try:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
        except Exception as exc:  # pragma: no cover
            raise OSError(f"Failed to delete '{entry}': {exc}") from exc


def image_md5(image: np.ndarray) -> str:
    """
    Compute an MD5 hash for an image array.

    This is intended for data integrity / change detection only, not security.

    The hash includes:
    - image dtype
    - image shape
    - contiguous pixel bytes

    Parameters
    ----------
    image:
        Image as a NumPy array.

    Returns
    -------
    str
        Hex MD5 digest.

    Raises
    ------
    TypeError
        If image is not a NumPy array.
    ValueError
        If image is empty.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("image is empty")

    image_c = np.ascontiguousarray(image)

    hasher = hashlib.md5()
    hasher.update(str(image_c.dtype).encode("utf-8"))
    hasher.update(str(image_c.shape).encode("utf-8"))
    hasher.update(image_c.tobytes())

    digest = hasher.hexdigest()

    LOGGER.debug(
        "Computed image MD5: %s (shape=%s dtype=%s)",
        digest,
        image_c.shape,
        image_c.dtype,
    )

    return digest

def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute an MD5 hash of a file's raw bytes.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

# ----------------------------
# Numeric helpers
# ----------------------------
def pct_list_to_int_list(pct_list: Sequence[float], scale: int) -> list[int]:
    """
    Convert a list of fractions (0..1 typically) into integer pixel values for a given scale.

    The conversion is rounded down to the nearest *even* integer:
        floor(pct * scale / 2) * 2

    Parameters
    ----------
    pct_list:
        Sequence of floats.
    scale:
        Integer scaling factor (e.g., image width or height).

    Returns
    -------
    list[int]
    """
    if not isinstance(scale, (int, np.integer)):
        raise TypeError("scale must be an integer")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    if not isinstance(pct_list, Sequence) or isinstance(pct_list, (str, bytes)):
        raise TypeError("pct_list must be a sequence of floats")

    out: list[int] = []
    for i, entry in enumerate(pct_list):
        if not isinstance(entry, (float, int, np.floating, np.integer)):
            raise TypeError(f"pct_list[{i}] must be numeric, got {type(entry)!r}")
        val = math.floor(float(entry) * float(scale) / 2.0) * 2
        out.append(int(val))
    return out


def pixels_to_pct(pixel_coords: tuple[int, int], image_size: tuple[int, int]) -> tuple[float, float]:
    """
    Convert pixel coordinates to fractions of overall image size.

    Parameters
    ----------
    pixel_coords:
        (pix_x, pix_y) integers.
    image_size:
        (size_x, size_y) integers.

    Returns
    -------
    (pct_x, pct_y):
        Fractions in [0, 1]. Note that 1.0 is allowed when pix==size.

    Raises
    ------
    TypeError, ValueError
    """
    if (
        not isinstance(pixel_coords, tuple)
        or not isinstance(image_size, tuple)
        or len(pixel_coords) != 2
        or len(image_size) != 2
    ):
        raise TypeError("pixel_coords and image_size must be tuples of length 2")

    pix_x, pix_y = pixel_coords
    size_x, size_y = image_size

    if not all(isinstance(v, (int, np.integer)) for v in (pix_x, pix_y, size_x, size_y)):
        raise TypeError("pixel_coords and image_size must contain integers")

    if size_x <= 0 or size_y <= 0:
        raise ValueError("image_size values must be positive")

    if not (0 <= pix_x <= size_x and 0 <= pix_y <= size_y):
        raise ValueError("pixel_coords must be within image bounds")

    return float(pix_x) / float(size_x), float(pix_y) / float(size_y)

def line_intersection(line1: Line, line2: Line) -> Point:
    """
    Compute the intersection point of two infinite lines defined by two points each.

    Parameters
    ----------
    line1 : ((x0, y0), (x1, y1))
    line2 : ((x2, y2), (x3, y3))

    Returns
    -------
    (x, y) : tuple[float, float]
        Intersection point rounded to nearest hundredth (2 decimal places).

    Raises
    ------
    TypeError
        If inputs are not properly structured numeric pairs.
    ValueError
        If lines are degenerate (identical points) or parallel.
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    def _validate_line(line, name: str):
        if not isinstance(line, (tuple, list)) or len(line) != 2:
            raise TypeError(f"{name} must be a tuple/list of two points")

        for i, pt in enumerate(line):
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                raise TypeError(f"{name}[{i}] must be a tuple/list of two numeric values")
            if not all(isinstance(v, (int, float)) for v in pt):
                raise TypeError(f"{name}[{i}] must contain numeric values")

        (x0, y0), (x1, y1) = line
        if x0 == x1 and y0 == y1:
            raise ValueError(f"{name} defines a degenerate line (identical points)")

    _validate_line(line1, "line1")
    _validate_line(line2, "line2")

    (x0, y0), (x1, y1) = line1
    (x2, y2), (x3, y3) = line2

    # Convert to float for numerical stability
    x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
    x2, y2, x3, y3 = map(float, (x2, y2, x3, y3))

    # ----------------------------
    # Compute intersection
    # ----------------------------
    # Line1: (x0,y0) + t*(dx1,dy1)
    # Line2: (x2,y2) + u*(dx2,dy2)

    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x3 - x2
    dy2 = y3 - y2

    # Determinant
    denom = dx1 * dy2 - dy1 * dx2

    if math.isclose(denom, 0.0, abs_tol=1e-12):
        raise ValueError("Lines are parallel or coincident; no unique intersection")

    # Solve for t
    t = ((x2 - x0) * dy2 - (y2 - y0) * dx2) / denom

    x_int = x0 + t * dx1
    y_int = y0 + t * dy1

    # Round to nearest hundredth
    x_int = round(x_int, 2)
    y_int = round(y_int, 2)

    return (x_int, y_int)

# ----------------------------
# Image type helpers
# ----------------------------
def _as_uint8_channel(a: np.ndarray) -> np.ndarray:
    """Convert a single-channel array to uint8 safely."""
    if a is None:
        raise ValueError("channel is None")

    a = np.asarray(a)

    if a.dtype == np.uint8:
        return a

    if np.issubdtype(a.dtype, np.floating):
        amin = float(np.nanmin(a))
        amax = float(np.nanmax(a))

        # Common cases: [0..1] or [0..255]
        if 0.0 <= amin and amax <= 1.0:
            return np.clip(a * 255.0, 0, 255).astype(np.uint8)
        if 0.0 <= amin and amax <= 255.0:
            return np.clip(a, 0, 255).astype(np.uint8)

        # Generic min-max scaling
        if abs(amax - amin) < 1e-12:
            return np.zeros_like(a, dtype=np.uint8)
        scaled = (a - amin) * (255.0 / (amax - amin))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    # Integers / other: clip
    return np.clip(a, 0, 255).astype(np.uint8)


def ensure_gray_f64(image: np.ndarray, gray_mode: str = "luma") -> np.ndarray:
    """
    Ensure a grayscale float64 image.

    Accepts:
      - grayscale (H,W): returned as float64, whatever ``gray_mode``
      - BGR color (H,W,3): reduced to one channel according to ``gray_mode``:
        "luma" (default) cv2 BGR2GRAY, "green" the G channel only, "max" the per-pixel
        max of B, G, R

    Returns
    -------
    np.ndarray float64 of shape (H,W)
    """
    if gray_mode not in GRAY_MODES:
        raise ValueError(f"gray_mode must be one of {GRAY_MODES}, got {gray_mode!r}")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image is empty")

    if image.ndim == 2:
        return np.asarray(image, dtype=np.float64)

    if image.ndim == 3 and image.shape[2] == 3:
        # Use OpenCV luminance conversion for stability
        u8 = ensure_bgr_u8(image)
        if gray_mode == "green":
            return u8[:, :, 1].astype(np.float64)
        if gray_mode == "max":
            return u8.max(axis=2).astype(np.float64)
        gray_u8 = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY)
        return gray_u8.astype(np.float64)

    raise ValueError("image must be grayscale (H,W) or BGR color (H,W,3)")


def ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is BGR uint8.

    Accepts grayscale (H,W) or BGR (H,W,3) and converts as needed.
    """
    if image is None:
        raise ValueError("image is None")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image is empty")

    if image.ndim == 2:
        gray_u8 = _as_uint8_channel(image)
        return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype == np.uint8:
            return image.copy()
        b = _as_uint8_channel(image[:, :, 0])
        g = _as_uint8_channel(image[:, :, 1])
        r = _as_uint8_channel(image[:, :, 2])
        return np.dstack([b, g, r]).astype(np.uint8)

    raise ValueError("image must be grayscale (H,W) or BGR color (H,W,3)")


# ----------------------------
# Drawing helpers
# ----------------------------
def draw_crosshair(
    image: np.ndarray,
    point: tuple[float, float],
    leg_length: int,
    width: int,
    color: tuple[int, int, int] = (0, 255, 255),  # yellow in BGR
) -> np.ndarray:
    """Draw a crosshair centered at ``point`` and return a BGR uint8 copy."""
    if not isinstance(leg_length, (int, np.integer)) or leg_length <= 0:
        raise ValueError("leg_length must be an integer > 0")
    if not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError("width must be an integer > 0")
    if point is None or not isinstance(point, tuple) or len(point) != 2:
        raise ValueError("point must be a tuple (x, y)")
    if not (isinstance(color, tuple) and len(color) == 3 and all(0 <= int(c) <= 255 for c in color)):
        raise ValueError("color must be a BGR tuple with values in [0,255]")

    out = ensure_bgr_u8(image)
    h, w = out.shape[:2]

    cx = int(round(float(point[0])))
    cy = int(round(float(point[1])))

    # Horizontal segment
    x0 = max(0, cx - leg_length)
    x1 = min(w - 1, cx + leg_length)
    if 0 <= cy < h and x1 >= x0:
        cv2.line(out, (x0, cy), (x1, cy), color, thickness=int(width))

    # Vertical segment
    y0 = max(0, cy - leg_length)
    y1 = min(h - 1, cy + leg_length)
    if 0 <= cx < w and y1 >= y0:
        cv2.line(out, (cx, y0), (cx, y1), color, thickness=int(width))

    return out


def draw_box_on_image(
    image: np.ndarray,
    corner1: tuple[int, int],
    corner2: tuple[int, int],
    width: int,
    color: tuple[int, int, int] = (0, 255, 255),  # yellow in BGR
) -> np.ndarray:
    """Draw a rectangle on an image and return a BGR uint8 copy."""
    if not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError("width must be an integer > 0")
    if corner1 is None or corner2 is None or len(corner1) != 2 or len(corner2) != 2:
        raise ValueError("corner1 and corner2 must be (x, y) tuples")

    out = ensure_bgr_u8(image)
    h, w = out.shape[:2]

    x1, y1 = int(corner1[0]), int(corner1[1])
    x2, y2 = int(corner2[0]), int(corner2[1])

    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    # Clip to bounds (OpenCV rectangle uses inclusive coordinates)
    x_min = max(0, min(w - 1, x_min))
    x_max = max(0, min(w - 1, x_max))
    y_min = max(0, min(h - 1, y_min))
    y_max = max(0, min(h - 1, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Rectangle corners produce a degenerate box after clipping.")

    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, thickness=int(width))
    return out


def draw_line_through_points(
    image: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    thickness: int = 2,
    color: tuple[int, int, int] | int = (0, 0, 255),
    extend: bool = True,
) -> np.ndarray:
    """
    Draw a line on an image that crosses two points.

    If extend=True, the line is extended to the image borders (infinite line clipped to image).
    If extend=False, a segment is drawn between the two points.

    Returns a copy of the input with the line drawn.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (color)")
    if image.size == 0:
        raise ValueError("image is empty")

    h, w = image.shape[:2]

    if (
        not isinstance(p0, tuple)
        or not isinstance(p1, tuple)
        or len(p0) != 2
        or len(p1) != 2
        or not all(isinstance(v, (int, float, np.floating, np.integer)) for v in (*p0, *p1))
    ):
        raise TypeError("p0 and p1 must be numeric tuples (x, y)")

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        raise ValueError("p0 and p1 must be different points")

    if not isinstance(thickness, (int, np.integer)) or thickness < 1:
        raise ValueError("thickness must be an integer >= 1")

    # Normalize color for grayscale vs BGR
    if image.ndim == 2:
        if isinstance(color, tuple):
            if len(color) != 1:
                raise ValueError("For grayscale images, color must be an int or a 1-tuple")
            c = int(color[0])
        else:
            c = int(color)
        if not (0 <= c <= 255):
            raise ValueError("Grayscale color must be in [0, 255]")
        draw_color: int | tuple[int, int, int] = c
    else:
        if not (isinstance(color, tuple) and len(color) == 3):
            raise TypeError("For color images, color must be a 3-tuple (B,G,R)")
        if not all(0 <= int(c) <= 255 for c in color):
            raise ValueError("Color components must be in [0, 255]")
        draw_color = (int(color[0]), int(color[1]), int(color[2]))

    out = image.copy()

    def _clip_infinite_line_to_image(
        ax: float, ay: float, bx: float, by: float, w_: int, h_: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Return endpoints of the infinite line through (ax,ay)-(bx,by) clipped to the image rectangle.
        """
        dx_ = bx - ax
        dy_ = by - ay
        eps = 1e-12
        pts: list[tuple[float, float]] = []

        # x = 0 and x = w-1
        if abs(dx_) > eps:
            t = (0.0 - ax) / dx_
            y = ay + t * dy_
            if 0.0 <= y <= (h_ - 1):
                pts.append((0.0, y))

            t = ((w_ - 1) - ax) / dx_
            y = ay + t * dy_
            if 0.0 <= y <= (h_ - 1):
                pts.append((float(w_ - 1), y))

        # y = 0 and y = h-1
        if abs(dy_) > eps:
            t = (0.0 - ay) / dy_
            x = ax + t * dx_
            if 0.0 <= x <= (w_ - 1):
                pts.append((x, 0.0))

            t = ((h_ - 1) - ay) / dy_
            x = ax + t * dx_
            if 0.0 <= x <= (w_ - 1):
                pts.append((x, float(h_ - 1)))

        # Deduplicate
        uniq: list[tuple[float, float]] = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
                uniq.append(p)

        if len(uniq) < 2:
            raise ValueError("Line does not intersect the image bounds")

        # Farthest pair
        best_i, best_j, best_d2 = 0, 1, -1.0
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d2 = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
                if d2 > best_d2:
                    best_d2 = d2
                    best_i, best_j = i, j
        p0_, p1_ = uniq[best_i], uniq[best_j]
        return (int(round(p0_[0])), int(round(p0_[1]))), (int(round(p1_[0])), int(round(p1_[1])))

    if extend:
        pt_a, pt_b = _clip_infinite_line_to_image(x0, y0, x1, y1, w, h)
    else:
        pt_a = (int(round(x0)), int(round(y0)))
        pt_b = (int(round(x1)), int(round(y1)))

    cv2.line(out, pt_a, pt_b, draw_color, int(thickness), lineType=cv2.LINE_AA)
    return out


# ----------------------------
# Plotting helper
# ----------------------------
def plot_xy_positions(
    positions: np.ndarray,
    *,
    x_label: str = "Index",
    y_label: str = "Value",
    title: str = "Position Plot",
    save_path: str = "example.jpg",
    dpi: int = 300,
) -> None:
    """
    Plot (x,y) positions and save as a JPG.

    Accepts:
      - 1D array y -> x is index
      - 2D array of shape (N,2) where column 0 is x and column 1 is y
    """
    if positions is None:
        raise ValueError("positions is None")
    if not isinstance(dpi, (int, np.integer)) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim == 1:
        y = arr
        x = np.arange(arr.size, dtype=np.float64)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        x = arr[:, 0]
        y = arr[:, 1]
    else:
        raise ValueError("positions must be 1D or shape (N,2)")

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=int(dpi), format="jpg")
    plt.close()


# ----------------------------
# Center finding
# ----------------------------
def sdrm(values, search_size, debug = True, debug_prefix = "./test_out/test-"):
    """

    --------
    Inputs
    --------
    values: 
        A 1D array of floating point number. 
        This is the data who's center we want to find. 
    search_size: int 
        This is the size of the window we want to search. 
    debug: bool 
        This determines if the debug information is output 
    debug_prefix: str 
        A prefix used for saving images. 

    --------        
    Outputs: 
    --------
    peak_from_left: 
        The position of the pixel, measured from the left of the area
    peak_from_center: 
        The position of the pixel, easured from the distance from the center pixel.


    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Input must be a 1D list or array")
    if not isinstance(search_size,int):
        raise ValueError("search_size must be an integer")
    if search_size < 0:
        raise ValueError("search_size must be >= 0")

    indices = np.arange(len(values), dtype=np.float64)
    values_flipped = np.flip(values)


    # print(f"length of indices:{len(indices)}")
    # print(f"length of values:{len(values)}")
    # print(f"length of values_flipped:{len(values_flipped)}")
    # Assume uniform spacing

    step = indices[1] - indices[0]

    sums = []
    sums2 = []
    sums3 = []

    x = np.average(values) # adjusts to background (average) to help avoid issues with rss going high because the backgroudn is bright. 

    for i in range (2*search_size+1):
        j = 2*search_size-i
        k = -1*search_size+i # Calcualted as the offset from the center pixel

        # Generate new index ranges
        lower_indices = indices[0] - step * np.arange(j, 0, -1)
        upper_indices = indices[-1] + step * np.arange(1, i)

        padded_indices = np.concatenate([lower_indices, indices, upper_indices])
        padded_values  = np.concatenate([np.ones(j)*x, values,         np.ones(i)*x])
        flipped_values =np.concatenate([np.ones(i)*x , values_flipped, np.ones(j)*x]) 

        diff = padded_values - flipped_values
        rss = math.sqrt(float(np.sum(diff * diff)))

        # print(f"i :{i}\tj :{j}\tindex:{indices[i]}\tl_pad:{l_pad}\tl_flip:{l_flip}\trss:{rss}")
        # print(f"i :{i}\tj :{j}\tindex:{indices[i]}\trss:{rss}")
        
        # print(f"i :{i}\tj :{j}\tk:{k}\tindex:{indices[i]}\trss:{rss}")

        sums.append((i,rss))
        sums2.append((indices[i],rss))
        sums3.append((k,rss))


    min_point = min(sums, key=lambda t: t[1]) # 
    min_point2 = min(sums2, key=lambda t: t[1])# 
    min_point3 = min(sums3, key=lambda t: t[1])# Calculated

    if debug: 
        plot_xy_positions(padded_values,x_label = "Pixel Value", y_label = "Pixel Intensity", title = "Padded Values", save_path = debug_prefix+"---sdrm-padded.jpg",dpi = 300)
        plot_xy_positions(flipped_values,x_label = "Pixel Value", y_label = "PIxel Intensity", title = "Flipped Values", save_path = debug_prefix+"---sdrm-flpped.jpg",dpi = 300)
        plot_xy_positions(sums2,x_label = "Pixel Value", y_label = "Rss Values", title = "Rss value", save_path = debug_prefix+"---sdrm-rss-from-left.jpg",dpi = 300)
        plot_xy_positions(sums3,x_label = "Pixel Value, corrected", y_label = "Rss Values", title = "Rss value", save_path = debug_prefix+"---srdm-rss-centered.jpg",dpi = 300)

        print(f"search_size:\t{search_size}")
        # print(f"min point1:{min_point}")
        # print(f"min point2:{min_point2}")
        print(f"min point3:{min_point3}")

        # print(f"sums:\n{sums}")
        # print(f"sums2:\n{sums2}")

    peak_from_left  = min_point[0]
    peak_from_center = min_point3[0]

    return peak_from_center


def sdrm_2(
    values,
    search_size: int,
    q_limit:float,
    debug: bool = True,
    debug_prefix: str = "./test_out/test-",
    *,
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
) -> tuple[int | float | None, float]:
    """
    Symmetric Difference RMS Minimum (SDRM) center offset estimator for 1D signals, with
    a quality check.

    Runs ``_sdrm_core`` (see there for the algorithm and the sub-pixel refinement) and
    rejects the result when the quality ratio exceeds ``q_limit``.

    Parameters
    ----------
    values:
        1D sequence/array of numeric values.
    search_size:
        Non-negative integer window size to search on each side of center.
    q_limit:
        Non-Negative integer between 0 and 1.  Used to determine if the response is stable.
    debug:
        If True, write debug plots and print key diagnostics.
    debug_prefix:
        File prefix for debug plot outputs.
    subpixel:
        If True, refine the offset to sub-pixel precision. Default False.
    subpixel_neighbors:
        Points on each side of the minimum used in the quadratic fit; int in 1..3.

    Returns
    -------
    (peak_from_center, r1):
        ``peak_from_center`` is the offset (negative/positive) from the center sample that
        minimizes RSS: an int when ``subpixel`` is False, a float when True, and None when
        the quality ratio exceeds ``q_limit``. ``r1`` is min(RSS) / mean(RSS).

    Raises
    ------
    ValueError
        If ``subpixel_neighbors`` is not an int in 1..3.
    """
    if not isinstance(q_limit,(float,np.float32, np.float64)):
        raise TypeError(f"q_limit must be a float, got {type(q_limit).__name__}")
    if not 0 < q_limit < 1:
        raise ValueError(f"q_limit must be between 0 and 1, got {q_limit}")

    peak_from_center, r1 = _sdrm_core(
        values,
        search_size,
        debug=debug,
        debug_prefix=debug_prefix,
        subpixel=subpixel,
        subpixel_neighbors=subpixel_neighbors,
    )
    if r1 > q_limit:
        return None, r1
    return peak_from_center, r1


def _sdrm_core(
    values,
    search_size: int,
    *,
    debug: bool = False,
    debug_prefix: str = "./test_out/test-",
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
) -> tuple[int | float, float]:
    """
    SDRM center offset estimator for 1D signals, without a quality check. Returns
    (peak_from_center, r1); see ``_sdrm_core_detail``.
    """
    peak_from_center, r1, _k_int = _sdrm_core_detail(
        values,
        search_size,
        debug=debug,
        debug_prefix=debug_prefix,
        subpixel=subpixel,
        subpixel_neighbors=subpixel_neighbors,
    )
    return peak_from_center, r1


def _sdrm_core_detail(
    values,
    search_size: int,
    *,
    debug: bool = False,
    debug_prefix: str = "./test_out/test-",
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
) -> tuple[int | float, float, int]:
    """
    SDRM center offset estimator for 1D signals, without a quality check.

    This function preserves the original SDRM algorithm:
    - Reverse the signal
    - Pad both original and reversed signals with a baseline value (mean)
    - For each offset in [-search_size, +search_size], compute RSS between padded arrays
    - Return the offset (from center) that minimizes RSS

    All offsets are in samples of the grid ``values`` is sampled on (pixels for a pixel
    profile, ``oversample_step`` px for an oversampled one).

    With ``subpixel=True``, the integer minimum is refined by a least-squares quadratic fit
    (``numpy.polyfit``, degree 2) to the sum of squared differences (RSS**2) at the minimum
    index m and ``subpixel_neighbors`` samples on each side. The vertex offset is added to
    the integer offset. With ``subpixel_neighbors=1`` this equals the 3-point parabola
    ``0.5*(a - c)/(a - 2b + c)``. The integer minimum is used instead when the fit window
    would extend past the search range or the fitted curvature is not positive, and the
    vertex offset is clamped to +/-1 sample.

    Returns
    -------
    (peak_from_center, r1, k_int):
        ``peak_from_center`` is the offset in samples that minimizes RSS: an int when
        ``subpixel`` is False, a float when True. ``r1`` is min(RSS) / mean(RSS).
        ``k_int`` is the integer offset of the RSS minimum (before any sub-pixel fit), in
        -search_size..+search_size.
    """
    # ----------------------------
    # Type + shape checks
    # ----------------------------
    if not isinstance(search_size, int):
        raise TypeError(f"search_size must be int, got {type(search_size).__name__}")
    if search_size < 0:
        raise ValueError(f"search_size must be >= 0, got {search_size}")

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1D; got shape {arr.shape}")
    n = int(arr.size)
    if n == 0:
        raise ValueError("values is empty")
    if n < 2:
        # The algorithm uses indices[1] - indices[0] in the original code.
        # With fewer than 2 points, the notion of "step" is undefined.
        raise ValueError("values must contain at least 2 elements")

    if not isinstance(debug, (bool, np.bool_)):
        raise TypeError(f"debug must be bool, got {type(debug).__name__}")
    if not isinstance(debug_prefix, str):
        raise TypeError(f"debug_prefix must be str, got {type(debug_prefix).__name__}")

    if not isinstance(subpixel, (bool, np.bool_)):
        raise TypeError(f"subpixel must be bool, got {type(subpixel).__name__}")
    if (
        isinstance(subpixel_neighbors, (bool, np.bool_))
        or not isinstance(subpixel_neighbors, (int, np.integer))
        or not 1 <= subpixel_neighbors <= 3
    ):
        raise ValueError(f"subpixel_neighbors must be an int in 1..3, got {subpixel_neighbors!r}")

    # ----------------------------
    # Core algorithm (kept same)
    # ----------------------------
    indices = np.arange(n, dtype=np.float64)
    values_flipped = np.flip(arr)

    step = float(indices[1] - indices[0])  # uniform spacing; equals 1.0 with arange

    # Baseline padding value (original code uses average to reduce background bias)
    baseline = float(np.average(arr))

    # Records for diagnostics / plotting
    rss_by_i = []        # (i, rss)
    rss_by_x = []        # (indices[i], rss)  NOTE: indices[i] uses i in [0..2*search_size]
    rss_by_k = []        # (k, rss) where k is offset from center: [-search_size .. +search_size]
    ssd_by_i = []        # rss**2 (sum of squared differences), indexed by i

    last_padded_values = None
    last_flipped_values = None

    # i ranges over total padding distribution (left/right)
    # original: for i in range (2*search_size+1):
    for i in range(2 * search_size + 1):
        j = 2 * search_size - i          # left pad count
        k = -search_size + i             # offset from center

        # Build padded index ranges (kept for parity; not used in RSS)
        # lower_indices = indices[0] - step * np.arange(j, 0, -1)
        # upper_indices = indices[-1] + step * np.arange(1, i)
        # padded_indices = np.concatenate([lower_indices, indices, upper_indices])

        padded_values = np.concatenate(
            [np.full(j, baseline, dtype=np.float64), arr, np.full(i, baseline, dtype=np.float64)]
        )
        flipped_values = np.concatenate(
            [np.full(i, baseline, dtype=np.float64), values_flipped, np.full(j, baseline, dtype=np.float64)]
        )

        diff = padded_values - flipped_values
        ssd = float(np.sum(diff * diff))
        rss = math.sqrt(ssd)

        rss_by_i.append((i, rss))
        ssd_by_i.append(ssd)

        # Preserve the original behavior: this indexing assumes n is large enough
        # for indices[i] when i <= 2*search_size. If not, we fail fast with a clear message.
        if i >= n:
            raise ValueError(
                f"search_size too large for values length: i={i} exceeds n-1={n-1}. "
                f"Require len(values) > 2*search_size (got len={n}, search_size={search_size})."
            )

        rss_by_x.append((float(indices[i]), rss))
        rss_by_k.append((int(k), rss))

        last_padded_values = padded_values
        last_flipped_values = flipped_values

    arr = np.asarray(rss_by_i, dtype=float)   # shape (N, 2)
    y_values = arr[:, 1]

    me = statistics.mean(y_values)
    ma = max(y_values)
    mi = min(y_values)
    r1 = mi/me 
    r2 = mi/ma


    # Choose minima (original did 3 mins; only k is used for return)
    min_i = min(rss_by_i, key=lambda t: t[1])    # (i, rss)
    min_k = min(rss_by_k, key=lambda t: t[1])    # (k, rss)

    # ----------------------------
    # Debug outputs / plots
    # ----------------------------
    if debug:
        # Mirror the original plotting behavior: show padded arrays and RSS curves.
        # These plots require plot_xy_positions to exist in the caller's module.
        if last_padded_values is not None:
            plot_xy_positions(
                last_padded_values,
                x_label="Pixel Position",
                y_label="Pixel Intensity",
                title="Padded Values (last iteration)",
                save_path=f"{debug_prefix}-padded_values.jpg",
                dpi=300,
            )
        if last_flipped_values is not None:
            plot_xy_positions(
                last_flipped_values,
                x_label="Pixel Position",
                y_label="Pixel Intensity",
                title="Flipped Values (last iteration)",
                save_path=f"{debug_prefix}-flipped_values.jpg",
                dpi=300,
            )

        plot_xy_positions(
            rss_by_x,
            x_label="Pixel Position",
            y_label="RSS",
            title="RSS vs position (indices[i])",
            save_path=f"{debug_prefix}-rss_vs_pos.jpg",
            dpi=300,
        )
        plot_xy_positions(
            rss_by_k,
            x_label="Offset from center (k)",
            y_label="RSS",
            title="RSS vs center offset (k)",
            save_path=f"{debug_prefix}-rss_vs_offset.jpg",
            dpi=300,
        )

        LOGGER.debug(f"SDRM - len(values)={n}")
        LOGGER.debug(f"SDRM - search_size={search_size}")
        LOGGER.debug(f"SDRM - baseline(mean)={baseline:.6g}")
        LOGGER.debug(f"SDRM - min_i=(i={min_i[0]}, rss={min_i[1]:.6g})")
        LOGGER.debug(f"SDRM - min_i=(i={min_i[0]}, rss={min_i[1]:.6g})")
        LOGGER.debug(f"SDRM - min_k=(k={min_k[0]}, rss={min_k[1]:.6g})")
        LOGGER.debug(f"SDRM - mean rss value={mi}")
        LOGGER.debug(f"SDRM - min rss value={ma}")
        LOGGER.debug(f"SDRM - min rss / mean rss={r1}")
        LOGGER.debug(f"SDRM - min rss / max rss={r2}")    


        # if r1 > 0.7:
        #     src = f"{debug_prefix}-rss_vs_pos.jpg"
        #     dst = f"{debug_prefix}-rss_vs_pos-007.jpg"
        #     shutil.copyfile(src,dst)
        # elif r1 > 0.6:
        #     src = f"{debug_prefix}-rss_vs_pos.jpg"
        #     dst = f"{debug_prefix}-rss_vs_pos-006.jpg"
        #     shutil.copyfile(src,dst)
        # elif r1 > 0.5:
        #     src = f"{debug_prefix}-rss_vs_pos.jpg"
        #     dst = f"{debug_prefix}-rss_vs_pos-005.jpg"
        #     shutil.copyfile(src,dst)
        # elif r1 > 0.4:
        #     src = f"{debug_prefix}-rss_vs_pos.jpg"
        #     dst = f"{debug_prefix}-rss_vs_pos-004.jpg"
        #     shutil.copyfile(src,dst)
        # elif r1 > 0.3:
        #     src = f"{debug_prefix}-rss_vs_pos.jpg"
        #     dst = f"{debug_prefix}-rss_vs_pos-003.jpg"
        #     shutil.copyfile(src,dst)


    # Original function returned peak_from_center only
    k_int = int(min_k[0])
    peak_from_center: int | float = k_int
    if subpixel:
        peak_from_center = _subpixel_offset(ssd_by_i, int(min_i[0]), peak_from_center, subpixel_neighbors)

    return peak_from_center, r1, k_int


def _subpixel_offset(ssd: list[float], m: int, k_m: int, neighbors: int) -> float:
    """
    Refine the integer SDRM minimum ``k_m`` (at index ``m`` of ``ssd``) with a least-squares
    quadratic fit over ``m - neighbors .. m + neighbors`` (in samples). See ``_sdrm_core``.
    """
    if m - neighbors < 0 or m + neighbors > len(ssd) - 1:
        LOGGER.debug(
            "SDRM subpixel: fit window m=%d +/-%d exceeds search range [0, %d]; using integer offset %d",
            m, neighbors, len(ssd) - 1, k_m,
        )
        return float(k_m)

    x = np.arange(-neighbors, neighbors + 1, dtype=np.float64)
    y = np.asarray(ssd[m - neighbors : m + neighbors + 1], dtype=np.float64)
    a, b, _c = np.polyfit(x, y, 2)
    if not a > 0:
        LOGGER.debug("SDRM subpixel: fitted curvature %.6g is not positive; using integer offset %d", a, k_m)
        return float(k_m)

    offset = -b / (2.0 * a)
    if abs(offset) > 1.0:
        clamped = math.copysign(1.0, offset)
        LOGGER.debug("SDRM subpixel: vertex offset %.6g clamped to %+.0f samples", offset, clamped)
        offset = clamped
    return float(k_m + offset)

def find_center_pixel(
    image: np.ndarray,
    center_position: tuple[float, float],
    search_size: int,
    q_limit:float,
    *,
    search_method: str = "sdrm",
    debug: bool = False,
    debug_prefix: str = "dbg",
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
) -> tuple[float, float]:
    """
    Estimate the center position of a 1D intensity distribution along the x-axis of a narrow strip image.

    Parameters
    ----------
    image:
        Grayscale strip image (H,W) or (H,W,3). If color, converted to grayscale.
    center_position:
        Approximate (x,y) in *strip coordinates*.
    search_size:
        Half-width of search window (pixels) around center_position[0], along x.
    search_method:
        "argmax", "moments", or "sdrm".
    debug:
        If True, emit debug logging and optionally write plots.
    debug_prefix:
        Prefix for debug artifacts (filenames).
    subpixel, subpixel_neighbors:
        Passed to ``sdrm_2`` (sdrm method only): refine the offset to sub-pixel precision.

    Returns
    -------
    pos_x:
        Estimated center position along x in strip coordinates.
    pos_left:
        Same position measured from the left edge (alias of pos_x for backwards compatibility).
    """
    gray = ensure_gray_f64(image)
    h, w = gray.shape[:2]

    if not isinstance(search_size, (int, np.integer)) or int(search_size) <= 0:
        raise ValueError("search_size must be an integer > 0")
    search_size = int(search_size)

    if center_position is None or not isinstance(center_position, tuple) or len(center_position) != 2:
        raise ValueError("center_position must be a tuple (x, y)")
    cx = float(center_position[0])

    profile_full = gray.mean(axis=0)  # (W,)

    # Clip window
    x0 = max(0, int(math.floor(cx - search_size)))
    x1 = min(w, int(math.ceil(cx + search_size)) + 1)  # exclusive

    if x1 <= x0:
        raise ValueError("Search window is empty after clipping")

    profile = profile_full[x0:x1]
    xs = np.arange(x0, x1, dtype=np.float64)

    method = search_method.lower().strip()
    if method == "argmax":
        idx = int(np.argmax(profile))
        pos_x = float(xs[idx])

    elif method == "moments":
        baseline = float(np.min(profile))
        wts = profile - baseline
        wts[wts < 0] = 0.0
        s = float(np.sum(wts))
        if s <= 1e-12:
            idx = int(np.argmax(profile))
            pos_x = float(xs[idx])
        else:
            pos_x = float(np.sum(xs * wts) / s)

    elif method == "sdrm":
        # Search offset relative to the *window center*
        window_center = int(round(cx))
        window_center = max(x0, min(x1 - 1, window_center))
        # translate to window coordinates
        center_in_window = window_center - x0
        max_off = min(search_size, center_in_window, (profile.size - 1) - center_in_window)
        if max_off <= 0:
            # Fallback if window is too small near edge
            idx = int(np.argmax(profile))
            pos_x = float(xs[idx])
        else:
            # Work on a symmetric chunk around the center to make SDRM meaningful
            chunk = profile[center_in_window - max_off : center_in_window + max_off + 1]
            # best_kk = sdrm(chunk,max_off,debug = True, debug_prefix = f"{debug_prefix}")
            best_k,q_ratio = sdrm_2(
                chunk,
                max_off,
                q_limit=q_limit,
                debug=debug,
                debug_prefix=f"{debug_prefix}",
                subpixel=subpixel,
                subpixel_neighbors=subpixel_neighbors,
            )
            if best_k == None:
                return None,q_ratio
            pos_x = float(window_center + best_k)

            if debug:
                pairs = np.array(
                    [(k, float(np.sqrt(np.mean((chunk - np.pad(chunk[::-1], (max_off, max_off), 'constant',
                                                              constant_values=float(np.mean(chunk)))[max_off + k : max_off + k + chunk.size]))**2)))
                     for k in range(-max_off, max_off + 1)],
                    dtype=np.float64,
                )
                # plot_xy_positions(
                #     pairs,
                #     x_label="Offset (px)",
                #     y_label="RMS",
                #     title="SDRM score vs offset",
                #     save_path=f"{debug_prefix}_sdrm.jpg",
                # )

    else:
        raise ValueError(f"Unknown search_method: {search_method!r}")

    if debug:
        LOGGER.debug(
            "find_center_pixel: w=%d, x0=%d, x1=%d, method=%s -> pos_x=%.3f",
            w,
            x0,
            x1,
            method,
            pos_x,
        )

    return pos_x, q_ratio

# ----------------------------
# Crosshair center estimation
# ----------------------------
def _clip_roi(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """Clip ROI bounds to [0,w]x[0,h] (x1,y1 are exclusive)."""
    x0c = max(0, min(w, x0))
    x1c = max(0, min(w, x1))
    y0c = max(0, min(h, y0))
    y1c = max(0, min(h, y1))
    return x0c, y0c, x1c, y1c


# Deskew pass 2: default search half-range around the pass-1 estimate, in pixels.
DESKEW_SEARCH_PX = 1.5
# Deskew pass 2: default limit on |pass2 - pass1| in pixels before falling back.
DESKEW_MAX_REFINE_SHIFT_PX = 0.75
# Deskew pass 2 falls back to the pass-1 result when more bins than this are empty.
DESKEW_MAX_EMPTY_FRACTION = 0.2
# A pixel counts as saturated when any channel (uint8 scale) is at least this value (R18).
SATURATION_LEVEL = 250
# A warning is logged when more than this fraction of a ROI's pixels is saturated.
SATURATION_WARN_FRACTION = 0.01
# Deskew pass 2: the profile half-width is at least this many line FWHMs (from pass 1).
DESKEW_WINDOW_PER_FWHM = 1.5
# Line-width estimate: the peak is searched within this many px of the pass-1 position.
LINE_PEAK_SEARCH_PX = 5


@dataclass(frozen=True)
class RoiTrace:
    """Per-ROI intermediate values of ``find_cross_center_detailed`` (for diagnostics)."""

    name: str
    x0: int
    y0: int
    # ROI pixels in search orientation (transposed for left/right).
    roi: np.ndarray
    # Pass-2 slope (dx/dy for top/bottom, dy/dx for left/right); None when pass 2 did not run.
    slope: float | None
    # Positions in ROI column coordinates. pos2 is the pass-2 candidate, also when it was
    # rejected (fallback:edge / fallback:shift); None when pass 2 did not compute one.
    pos1: float
    pos2: float | None
    # Fraction of ROI pixels with any channel >= SATURATION_LEVEL (original image, uint8 scale).
    saturation: float | None = None
    # Deskew half-width given to pass 2, and the pass-1 line FWHM estimate it was derived from
    # in adaptive mode (see _adaptive_fine_window). Both None when pass 2 did not run;
    # line_width is also None in fixed mode or when the FWHM could not be estimated.
    line_width: float | None = None
    fine_window_used: float | None = None


@dataclass(frozen=True)
class CrossCenterResult:
    """Result of ``find_cross_center_detailed``."""

    center: tuple[float, float] | None
    angles: tuple[float, float] | None
    overlay: np.ndarray
    # One RSS ratio per ROI, in ROI_NAMES order (always from pass 1).
    q_ratio_list: list[float]
    # Profile method per ROI, in ROI_NAMES order: "average", "deskew", or a pass-1 fallback
    # "fallback:tilt", "fallback:bins", "fallback:edge", "fallback:shift" (see _deskew_position).
    roi_methods: tuple[str, ...]
    # Intermediate values per ROI, in ROI_NAMES order.
    rois: tuple[RoiTrace, ...] = ()
    # Saturated-pixel fraction per ROI, in ROI_NAMES order (see roi_saturation_fraction).
    saturation: tuple[float, ...] = ()
    # Deskew half-width in px given to pass 2 per ROI, in ROI_NAMES order (fine_window in fixed
    # mode); None when pass 2 did not run for that ROI (average mode, or pass 1 failed).
    fine_windows: tuple[float | None, ...] = ()


def roi_saturation_fraction(roi_u8: np.ndarray, level: int = SATURATION_LEVEL) -> float:
    """
    Fraction of pixels in a uint8 ROI where any channel is >= ``level``.

    ``roi_u8`` is (H,W) or (H,W,C); a 2D ROI counts its single channel.
    """
    roi_u8 = np.asarray(roi_u8)
    if roi_u8.ndim not in (2, 3) or roi_u8.size == 0:
        raise ValueError(f"roi_u8 must be a non-empty 2D or 3D array, got shape {roi_u8.shape}")
    hot = roi_u8 >= level
    if hot.ndim == 3:
        hot = hot.any(axis=2)
    return float(np.count_nonzero(hot) / hot.size)


def _validate_profile_options(
    profile_mode: str, oversample_step: float, fine_window: int, refine_search: float, max_refine_shift: float
) -> None:
    if profile_mode not in PROFILE_MODES:
        raise ValueError(f"profile_mode must be one of {PROFILE_MODES}, got {profile_mode!r}")
    lo, hi = OVERSAMPLE_STEP_RANGE
    if (
        isinstance(oversample_step, (bool, np.bool_))
        or not isinstance(oversample_step, (int, float, np.floating, np.integer))
        or not lo <= oversample_step <= hi
    ):
        raise ValueError(f"oversample_step must be a number in {lo}..{hi}, got {oversample_step!r}")
    if (
        isinstance(fine_window, (bool, np.bool_))
        or not isinstance(fine_window, (int, np.integer))
        or fine_window < 2
    ):
        raise ValueError(f"fine_window must be an int >= 2, got {fine_window!r}")
    lo, hi = REFINE_SEARCH_RANGE
    if (
        isinstance(refine_search, (bool, np.bool_))
        or not isinstance(refine_search, (int, float, np.floating, np.integer))
        or not lo <= refine_search <= hi
    ):
        raise ValueError(f"refine_search must be a number in {lo}..{hi}, got {refine_search!r}")
    if not refine_search < fine_window:
        raise ValueError(f"refine_search ({refine_search}) must be smaller than fine_window ({fine_window})")
    lo, hi = MAX_REFINE_SHIFT_RANGE
    if (
        isinstance(max_refine_shift, (bool, np.bool_))
        or not isinstance(max_refine_shift, (int, float, np.floating, np.integer))
        or not lo <= max_refine_shift <= hi
    ):
        raise ValueError(f"max_refine_shift must be a number in {lo}..{hi}, got {max_refine_shift!r}")


def _half_crossing(s: np.ndarray, i_peak: int, half: float, direction: int) -> float | None:
    """Index where ``s`` first drops below ``half`` walking from ``i_peak`` (linear interpolation)."""
    i = i_peak
    while 0 <= i + direction < s.size:
        j = i + direction
        if s[j] < half:
            return i + (half - s[i]) * (j - i) / (s[j] - s[i])
        i = j
    return None


def _estimate_line_fwhm(profile: np.ndarray, pos1: float) -> float | None:
    """
    FWHM in px of the line in a pass-1 profile (ROI averaged along the line), or None.

    Background is the profile median (the line covers a small part of the ROI). Polarity
    and peak are taken within +/-``LINE_PEAK_SEARCH_PX`` of ``pos1``; the half-maximum
    crossings on both sides are linearly interpolated. Returns None when the profile has
    no peak there or a crossing is missing (line runs off the ROI).
    """
    p = np.asarray(profile, dtype=np.float64)
    c = int(round(pos1))
    lo, hi = max(0, c - LINE_PEAK_SEARCH_PX), min(p.size, c + LINE_PEAK_SEARCH_PX + 1)
    if hi <= lo:
        return None
    bg = float(np.median(p))
    near = p[lo:hi]
    sign = 1.0 if near.max() - bg >= bg - near.min() else -1.0
    s = sign * (p - bg)
    i_peak = lo + int(np.argmax(s[lo:hi]))
    if not s[i_peak] > 0:
        return None
    half = 0.5 * s[i_peak]
    left = _half_crossing(s, i_peak, half, -1)
    right = _half_crossing(s, i_peak, half, +1)
    if left is None or right is None:
        return None
    return float(right - left)


def _adaptive_fine_window(profile: np.ndarray, pos1: float, fine_window: float) -> tuple[float, float | None]:
    """
    Deskew half-width for pass 2: max(fine_window, DESKEW_WINDOW_PER_FWHM * FWHM), with the
    FWHM from ``_estimate_line_fwhm``. Never smaller than ``fine_window``; equal to it when
    the FWHM is unknown. Returns (window, fwhm).
    """
    fwhm = _estimate_line_fwhm(profile, pos1)
    if fwhm is None:
        return float(fine_window), None
    return max(float(fine_window), DESKEW_WINDOW_PER_FWHM * fwhm), fwhm


def _deskew_profile(
    roi: np.ndarray,
    pos1: float,
    slope: float,
    *,
    oversample_step: float,
    fine_window: float,
    name: str,
) -> tuple[np.ndarray | None, str]:
    """
    Deskewed, oversampled profile of one ROI (in search orientation: the line runs along
    the rows, the profile is taken across the columns).

    Every pixel is projected onto the axis across the line, u = col - slope * (row - row_ref)
    with row_ref = roi_height / 2, and binned at ``oversample_step`` within +/-``fine_window``
    px of the pass-1 estimate ``pos1``; empty bins are filled by linear interpolation. The
    center sample (index ``round(fine_window / oversample_step)``) is at ``pos1``.

    Returns (profile, "deskew"), or (None, reason) when pass 2 does not apply: "fallback:tilt"
    when the line shifts less than 1 px over the ROI depth, "fallback:bins" when more than
    ``DESKEW_MAX_EMPTY_FRACTION`` of the bins are empty.
    """
    depth = roi.shape[0]
    if not math.isfinite(slope) or depth * abs(slope) < 1.0:
        LOGGER.debug(
            "Deskew %s: roi_depth * |slope| = %d * %.4g < 1 px; using pass-1 result", name, depth, abs(slope)
        )
        return None, "fallback:tilt"

    rows, cols = np.indices(roi.shape, dtype=np.float64)
    u = cols - slope * (rows - depth / 2.0)

    half_bins = int(round(fine_window / oversample_step))
    n_bins = 2 * half_bins + 1
    bin_index = np.rint((u - pos1) / oversample_step).astype(np.int64) + half_bins
    inside = (bin_index >= 0) & (bin_index < n_bins)
    counts = np.bincount(bin_index[inside], minlength=n_bins)
    sums = np.bincount(bin_index[inside], weights=roi[inside], minlength=n_bins)
    filled = counts > 0
    empty_fraction = 1.0 - np.count_nonzero(filled) / n_bins
    if empty_fraction > DESKEW_MAX_EMPTY_FRACTION:
        LOGGER.debug(
            "Deskew %s: empty-bin fraction %.3f > %.2f; using pass-1 result",
            name, empty_fraction, DESKEW_MAX_EMPTY_FRACTION,
        )
        return None, "fallback:bins"

    bins = np.arange(n_bins, dtype=np.float64)
    profile = np.interp(bins, bins[filled], sums[filled] / counts[filled])
    LOGGER.debug("Deskew %s: slope=%.6g empty_fraction=%.3f", name, slope, empty_fraction)
    return profile, "deskew"


def _deskew_position(
    roi: np.ndarray,
    pos1: float,
    slope: float,
    *,
    oversample_step: float,
    fine_window: float,
    subpixel: bool,
    subpixel_neighbors: int,
    name: str,
    refine_search: float = DESKEW_SEARCH_PX,
    max_refine_shift: float = DESKEW_MAX_REFINE_SHIFT_PX,
) -> tuple[str, float | None]:
    """
    Pass 2 of the deskew profile mode for one ROI: center the ``_deskew_profile`` with
    ``_sdrm_core_detail`` (no quality limit) within +/-``refine_search`` px of ``pos1``.

    Returns (method, pos2). ``pos2`` is the pass-2 position in ROI column coordinates, or
    None when no profile could be built. The caller uses ``pos2`` only when ``method`` is
    "deskew"; otherwise it keeps the pass-1 result:
    - "fallback:tilt", "fallback:bins": see ``_deskew_profile`` (pos2 is None).
    - "fallback:edge": the integer fine minimum is at either end of the search range, so
      no real minimum was found inside it.
    - "fallback:shift": |pos2 - pos1| > ``max_refine_shift`` px.
    """
    profile, method = _deskew_profile(
        roi, pos1, slope, oversample_step=oversample_step, fine_window=fine_window, name=name
    )
    if profile is None:
        return method, None

    search_samples = int(round(refine_search / oversample_step))
    offset_samples, r1, k_int = _sdrm_core_detail(
        profile,
        search_samples,
        subpixel=subpixel,
        subpixel_neighbors=subpixel_neighbors,
    )
    pos2 = pos1 + offset_samples * oversample_step
    LOGGER.debug(
        "Deskew %s: r1=%.4g min_index=%+d/%d pass1=%.4f pass2=%.4f",
        name, r1, k_int, search_samples, pos1, pos2,
    )
    if abs(k_int) == search_samples:
        return "fallback:edge", pos2
    if abs(pos2 - pos1) > max_refine_shift:
        return "fallback:shift", pos2
    return "deskew", pos2


def sdrm_ssd_curve(values, search_size: int) -> np.ndarray:
    """
    SDRM sum of squared differences for offsets k = -search_size..+search_size (index
    k + search_size), with the same baseline padding as ``_sdrm_core``. For diagnostics.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size <= 2 * search_size:
        raise ValueError(
            f"values must be 1D with more than 2*search_size elements (got shape {arr.shape}, "
            f"search_size={search_size})"
        )
    baseline = float(np.average(arr))
    flipped = np.flip(arr)
    ssd = np.empty(2 * search_size + 1, dtype=np.float64)
    for i in range(2 * search_size + 1):
        j = 2 * search_size - i
        padded = np.concatenate([np.full(j, baseline), arr, np.full(i, baseline)])
        padded_flipped = np.concatenate([np.full(i, baseline), flipped, np.full(j, baseline)])
        diff = padded - padded_flipped
        ssd[i] = float(np.sum(diff * diff))
    return ssd


def _roi_point(name: str, x0: int, y0: int, roi: np.ndarray, pos: float) -> tuple[float, float]:
    """Full-image point of an ROI result; ``roi`` is in search orientation (transposed for left/right)."""
    if name in ("top", "bottom"):
        return (float(x0) + pos, float(y0) + roi.shape[0] / 2.0)
    # pos is along original y because of transpose
    return (float(x0) + (roi.shape[0] / 2.0), float(y0) + pos)


def find_cross_center(
    image: np.ndarray,
    crop_center: tuple[int, int],
    crop_size: tuple[int, int],
    roi_size: tuple[int, int],
    q_limit:float,
    *,
    slant: bool = False,
    debug: bool = False,
    debug_prefix: str = "dbg",
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
    profile_mode: str = "average",
    oversample_step: float = 0.1,
    fine_window: int = 40,
    refine_search: float = DESKEW_SEARCH_PX,
    max_refine_shift: float = DESKEW_MAX_REFINE_SHIFT_PX,
    gray_mode: str = "luma",
    fine_window_mode: str = "fixed",
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, np.ndarray, list[float]]:
    """
    Find the center of a crosshair and estimate tilt angles.

    Same as ``find_cross_center_detailed`` but returns a tuple and drops the per-ROI
    profile methods.

    Returns
    -------
    (center, angles, overlay, q_ratio_list)
        ``center`` and ``angles`` are None if any ROI fails the quality check.
        ``q_ratio_list`` holds one RSS ratio per ROI, in ``ROI_NAMES`` order.
    """
    result = find_cross_center_detailed(
        image,
        crop_center,
        crop_size,
        roi_size,
        q_limit,
        slant=slant,
        debug=debug,
        debug_prefix=debug_prefix,
        subpixel=subpixel,
        subpixel_neighbors=subpixel_neighbors,
        profile_mode=profile_mode,
        oversample_step=oversample_step,
        fine_window=fine_window,
        refine_search=refine_search,
        max_refine_shift=max_refine_shift,
        gray_mode=gray_mode,
        fine_window_mode=fine_window_mode,
    )
    return result.center, result.angles, result.overlay, result.q_ratio_list


def find_cross_center_detailed(
    image: np.ndarray,
    crop_center: tuple[int, int],
    crop_size: tuple[int, int],
    roi_size: tuple[int, int],
    q_limit:float,
    *,
    slant: bool = False,
    debug: bool = False,
    debug_prefix: str = "dbg",
    subpixel: bool = False,
    subpixel_neighbors: int = 2,
    profile_mode: str = "average",
    oversample_step: float = 0.1,
    fine_window: int = 40,
    refine_search: float = DESKEW_SEARCH_PX,
    max_refine_shift: float = DESKEW_MAX_REFINE_SHIFT_PX,
    gray_mode: str = "luma",
    fine_window_mode: str = "fixed",
) -> CrossCenterResult:
    """
    Find the center of a crosshair and estimate tilt angles.

    Parameters
    ----------
    subpixel, subpixel_neighbors:
        How the SDRM minimum is read (integer, or quadratic fit over ``subpixel_neighbors``
        samples on each side). Off by default (integer SDRM offsets). Applies to both passes.
    profile_mode:
        How each ROI's 1D profile is built. "average" (default): average the ROI along the
        line (pass 1 only). "deskew": after pass 1, project each pixel across the line using
        the slope measured in pass 1, bin at ``oversample_step`` px and center that profile
        (pass 2). A ROI falls back to its pass-1 result when the line shifts less than 1 px
        over the ROI depth ("fallback:tilt") or more than 20 % of the bins are empty
        ("fallback:bins"); see also ``refine_search`` and ``max_refine_shift``.
    oversample_step:
        Bin width of the deskewed profile in px, 0.02..0.5 (deskew only).
    fine_window:
        Half-width in px of the deskewed profile around the pass-1 estimate (deskew only); a
        minimum when ``fine_window_mode`` is "adaptive". The half-width used per ROI is
        recorded in ``fine_windows``.
    fine_window_mode:
        "fixed" (default): every ROI uses ``fine_window``. "adaptive": per ROI the half-width
        is max(fine_window, 1.5 * line FWHM), with the FWHM estimated from the pass-1 profile
        (see ``_estimate_line_fwhm``). Deskew only.
    refine_search:
        Pass-2 search half-range in px around the pass-1 estimate, 0.5..3.0 and smaller than
        ``fine_window`` (deskew only). A ROI whose fine minimum lies at either end of this
        range falls back to its pass-1 result ("fallback:edge").
    max_refine_shift:
        A ROI whose pass-2 position differs from pass 1 by more than this many px, 0.1..3.0,
        falls back to its pass-1 result ("fallback:shift") (deskew only).
    gray_mode:
        How a BGR image is reduced to one channel: "luma" (default, cv2 BGR2GRAY), "green"
        (G channel only) or "max" (per-pixel max of B, G, R). See ``ensure_gray_f64``.

    Returns
    -------
    CrossCenterResult
        ``center`` and ``angles`` are None if any ROI fails the quality check. The quality
        check is made in pass 1 only, so pass 2 never causes a failure.

    Notes
    -----
    This implementation uses four ROIs (top/right/bottom/left) around a crop region.
    Each ROI is reduced to a 1D profile and centered using ``find_center_pixel``.
    """

    LOGGER.info(
        "Running center-finding util with crop_center=%s crop_size=%s roi_size=%s debug=%s "
        "profile_mode=%s subpixel=%s gray_mode=%s",
        crop_center,
        crop_size,
        roi_size,
        debug,
        profile_mode,
        subpixel,
        gray_mode,
    )

    _validate_profile_options(profile_mode, oversample_step, fine_window, refine_search, max_refine_shift)
    if gray_mode not in GRAY_MODES:
        raise ValueError(f"gray_mode must be one of {GRAY_MODES}, got {gray_mode!r}")
    if fine_window_mode not in FINE_WINDOW_MODES:
        raise ValueError(f"fine_window_mode must be one of {FINE_WINDOW_MODES}, got {fine_window_mode!r}")

    if image is None:
        raise ValueError("image is None")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (BGR)")
    if crop_center is None or len(crop_center) != 2:
        raise ValueError("crop_center must be (x, y)")
    if crop_size is None or len(crop_size) != 2:
        raise ValueError("crop_size must be (w, h)")
    if roi_size is None or len(roi_size) != 2:
        raise ValueError("roi_size must be (w, h)")

    cx_crop, cy_crop = int(crop_center[0]), int(crop_center[1])
    crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    roi_w, roi_h = int(roi_size[0]), int(roi_size[1])
    roi_w  = min(crop_w,crop_h)
    roi_w_v = crop_h
    roi_w_h = crop_h


    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("crop_size values must be > 0")
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError("roi_size values must be > 0")

    gray = ensure_gray_f64(image, gray_mode)
    h_img, w_img = gray.shape[:2]

    # Crop bounds
    cx0 = max(0, cx_crop - crop_w // 2)
    cy0 = max(0, cy_crop - crop_h // 2)
    cx1 = min(w_img, cx0 + crop_w)
    cy1 = min(h_img, cy0 + crop_h)

    # Re-adjust to preserve size if clipped
    if (cx1 - cx0) < crop_w:
        cx0 = max(0, cx1 - crop_w)
    if (cy1 - cy0) < crop_h:
        cy0 = max(0, cy1 - crop_h)

    overlay = ensure_bgr_u8(gray)
    if debug:
        overlay = draw_box_on_image(overlay, (cx0, cy0), (cx1 - 1, cy1 - 1), width=3, color=(255, 0, 0))
        # LOGGER.debug(f"overlay file save location: {debug_prefix}_overlay.jpg")
        cv2.imwrite(f"{debug_prefix}_overlay.jpg", overlay)

    # ROI centers in full-image coordinates (approximate)
    roi_centers = {
        "top": (cx_crop, cy0),
        "bottom": (cx_crop, cy1 - 1),
        "right": (cx1 - 1, cy_crop),
        "left": (cx0, cy_crop),
    }  # keys in ROI_NAMES order

    centers_measured: dict[str, tuple[float, float]] = {}

    # LOGGER.debug(f"Debug value in find_cross_center: {debug}")
    # LOGGER.debug(f"ROI Centers: {roi_centers.items()}")

    # for name,(rx,ry) in roi_centers.items():
    #         LOGGER.debug(f"ROI Name: {name}")

    measure_fail = 0
    q_ratio_list = []
    # Per ROI, from pass 1: (x0, y0, x1, y1, roi in search orientation, position in roi)
    roi_data: dict[str, tuple[int, int, int, int, np.ndarray, float]] = {}
    for name, (rx, ry) in roi_centers.items():
        # LOGGER.debug(f"Inside the ROI Loop - Name of ROI:{name}")
        # LOGGER.debug(f"Find Cross Center - Debug prefix: {debug_prefix}_overlay.jpg")

        if name in ("top", "bottom"):
            # ROI is wide in x, short in y
            s = 0
            if name == "top":
                s = 1
            x0 = int(rx - roi_w /2 )
            x1 = int(rx + roi_w /2 )
            y0 = int(ry - roi_h /2 )
            y1 = int(ry + roi_h /2 )

            y0 = int(ry - roi_h * (1-s) )
            y1 = int(ry + roi_h * (s-0) )

            # print(f"name:{name}\trx:{rx}\try:{ry}\ty0:{y0}\ty1:{y1}")

            x0, y0, x1, y1 = _clip_roi(x0, y0, x1, y1, w_img, h_img)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                raise ValueError(f"Empty ROI for {name} after clipping")
            center_in_roi = (roi.shape[1] / 2.0, roi.shape[0] / 2.0)
            pos_x, q_ratio = find_center_pixel(
                roi,
                center_position=center_in_roi,
                search_size=max(1, roi.shape[1] // 2 - 2),
                q_limit=q_limit,
                search_method="sdrm",
                debug=False,
                subpixel=subpixel,
                subpixel_neighbors=subpixel_neighbors,
                debug_prefix=f"{debug_prefix}_{name}",
            )

            # LOGGER.debug(f"Quality ratio of center finding: {q_ratio}")
            if pos_x == None:
                LOGGER.warning(f"For ROI {name}, Quality too low for reliable center finding: {q_ratio}")
                measure_fail = True
                pos_x=0 # Note - will not be used for computation, because "measure_fail" should trigger

        else:
            # ROI is tall in y, short in x; transpose so we still search along x-axis
            s = 0
            if name == "right":
                s = 1
            x0 = int(rx - roi_h / 2)
            x1 = int(rx + roi_h / 2)
            y0 = int(ry - roi_w / 2)
            y1 = int(ry + roi_w / 2)

            x0 = int(rx - roi_h * (s-0) )
            x1 = int(rx + roi_h * (1-s) )

            x0, y0, x1, y1 = _clip_roi(x0, y0, x1, y1, w_img, h_img)
            roi0 = gray[y0:y1, x0:x1]
            if roi0.size == 0:
                raise ValueError(f"Empty ROI for {name} after clipping")
            roi = roi0.T
            center_in_roi = (roi.shape[1] / 2.0, roi.shape[0] / 2.0)
            pos_x, q_ratio = find_center_pixel(
                roi,
                center_position=center_in_roi,
                search_size=max(1, roi.shape[1] // 2 - 2),
                q_limit=q_limit,
                search_method="sdrm",
                debug=False,
                subpixel=subpixel,
                subpixel_neighbors=subpixel_neighbors,
                debug_prefix=f"{debug_prefix}_{name}",
            )

            if pos_x == None:
                LOGGER.warning(f"For ROI {name}, Quality too low for reliable center finding: {q_ratio}")
                measure_fail = True
                pos_x=0 # Note - will not be used for computation, because "measure_fail" should trigger

        q_ratio_list.append(float(q_ratio))
        roi_data[name] = (x0, y0, x1, y1, roi, pos_x)
        centers_measured[name] = _roi_point(name, x0, y0, roi, pos_x)

    # Saturation from the original image, not the reduced gray (a clipped channel may be
    # hidden by the luma weights). Reported only; never fails the measurement (R18).
    color_u8 = ensure_bgr_u8(image)
    saturation: dict[str, float] = {}
    for name in ROI_NAMES:
        x0, y0, x1, y1 = roi_data[name][:4]
        saturation[name] = roi_saturation_fraction(color_u8[y0:y1, x0:x1])
        if saturation[name] > SATURATION_WARN_FRACTION:
            LOGGER.warning(
                "ROI %s: %.2f %% of pixels saturated (any channel >= %d; warning above %.0f %%)",
                name, 100.0 * saturation[name], SATURATION_LEVEL, 100.0 * SATURATION_WARN_FRACTION,
            )

    roi_methods = {name: "average" for name in ROI_NAMES}
    roi_slopes: dict[str, float | None] = {name: None for name in ROI_NAMES}
    roi_pos2: dict[str, float | None] = {name: None for name in ROI_NAMES}
    roi_line_width: dict[str, float | None] = {name: None for name in ROI_NAMES}
    roi_window: dict[str, float | None] = {name: None for name in ROI_NAMES}
    if profile_mode == "deskew" and not measure_fail:
        # Pass 2: slopes from the pass-1 points; vertical line dx/dy, horizontal line dy/dx.
        top, bottom = centers_measured["top"], centers_measured["bottom"]
        right, left = centers_measured["right"], centers_measured["left"]
        dy_v = top[1] - bottom[1]
        dx_h = right[0] - left[0]
        slope_v = (top[0] - bottom[0]) / dy_v if dy_v != 0 else math.nan
        slope_h = (right[1] - left[1]) / dx_h if dx_h != 0 else math.nan
        for name in ROI_NAMES:
            x0, y0, _x1, _y1, roi, pos1 = roi_data[name]
            roi_slopes[name] = slope_v if name in ("top", "bottom") else slope_h
            if fine_window_mode == "adaptive":
                window, line_width = _adaptive_fine_window(roi.mean(axis=0), pos1, fine_window)
                LOGGER.debug(
                    "Deskew %s: pass-1 line FWHM %s px, fine_window %.4g px (minimum %d)",
                    name, "unknown" if line_width is None else f"{line_width:.4g}", window, fine_window,
                )
            else:
                window, line_width = fine_window, None
            roi_window[name], roi_line_width[name] = float(window), line_width
            method, pos2 = _deskew_position(
                roi,
                pos1,
                roi_slopes[name],
                oversample_step=oversample_step,
                fine_window=window,
                subpixel=subpixel,
                subpixel_neighbors=subpixel_neighbors,
                name=name,
                refine_search=refine_search,
                max_refine_shift=max_refine_shift,
            )
            roi_methods[name] = method
            roi_pos2[name] = pos2
            if method == "deskew":
                centers_measured[name] = _roi_point(name, x0, y0, roi, pos2)
            elif pos2 is not None:
                # Rejected pass-2 candidate (fallback:edge / fallback:shift).
                axis, origin = ("x", x0) if name in ("top", "bottom") else ("y", y0)
                LOGGER.warning(
                    "Deskew ROI %s: %s, keeping pass-1 result. pass1 %s=%.3f, pass2 %s=%.3f "
                    "(diff %+.3f px; refine_search %g px, max_refine_shift %g px)",
                    name, method, axis, origin + pos1, axis, origin + pos2, pos2 - pos1,
                    refine_search, max_refine_shift,
                )
        LOGGER.info(
            "Deskew profile methods: %s",
            ", ".join(f"{name}={roi_methods[name]}" for name in ROI_NAMES),
        )
    roi_methods_tuple = tuple(roi_methods[name] for name in ROI_NAMES)
    roi_traces = tuple(
        RoiTrace(
            name=name,
            x0=roi_data[name][0],
            y0=roi_data[name][1],
            roi=roi_data[name][4],
            slope=roi_slopes[name],
            pos1=roi_data[name][5],
            pos2=roi_pos2[name],
            saturation=saturation[name],
            line_width=roi_line_width[name],
            fine_window_used=roi_window[name],
        )
        for name in ROI_NAMES
    )
    saturation_tuple = tuple(saturation[name] for name in ROI_NAMES)
    windows_tuple = tuple(roi_window[name] for name in ROI_NAMES)

    for name in ROI_NAMES:
        x0, y0, x1, y1, _roi, _pos = roi_data[name]
        measured = centers_measured[name]
        if debug:
            overlay = draw_box_on_image(overlay, (x0, y0), (x1 - 1, y1 - 1), width=2, color=(0, 255, 255))
            overlay = draw_crosshair(overlay, measured, leg_length=10, width=2, color=(0, 255, 0))

        if debug:
            LOGGER.debug("%s ROI bounds: (%d,%d)-(%d,%d) measured=%s", name, x0, y0, x1, y1, measured)

    if debug:
        cv2.imwrite(f"{debug_prefix}_overlay.jpg", overlay)

    if measure_fail: 
        LOGGER.warning(f"Quality too low for reliable center finding: - No center computed")
        return CrossCenterResult(
            None, None, overlay, q_ratio_list, roi_methods_tuple, roi_traces, saturation_tuple, windows_tuple
        )

    # Combine centers
    top = centers_measured["top"]
    bottom = centers_measured["bottom"]
    right = centers_measured["right"]
    left = centers_measured["left"]

    # old center finding algoithm - not accurate. 
    # x_center = (top[0] + bottom[0]) / 2.0
    # y_center = (right[1] + left[1]) / 2.0
    # center = (x_center, y_center)
    
    l1 = (top,bottom)
    l2 = (left, right)
    center = line_intersection(l1, l2)

    # Angles: use atan2 for robustness
    # Vertical tilt (from vertical): dx over dy between top and bottom
    dx_v = top[0] - bottom[0]
    dy_v = top[1] - bottom[1]
    angle_v = float(np.degrees(np.arctan2(dx_v, dy_v)))

    # Horizontal tilt (from horizontal): (-dy) over dx between right and left
    dy_h = (right[1] - left[1])
    dx_h = (right[0] - left[0])
    angle_h = float(np.degrees(np.arctan2(-dy_h, dx_h)))

    if angle_h<0:angle_h = angle_h+180
    if angle_v<0:angle_v = angle_v+180

    if debug:
        overlay = draw_crosshair(overlay, center, leg_length=12, width=3, color=(0, 0, 255))
        overlay = draw_line_through_points(overlay, top, bottom, thickness=1, color=(255, 0, 255), extend=True)
        overlay = draw_line_through_points(overlay, right, left, thickness=1, color=(255, 0, 255), extend=True)
        LOGGER.debug(f"Find Cross Center - overlay file output: {debug_prefix}_overlay.jpg")
        LOGGER.debug(f"Quality ratio from boxes: {q_ratio_list}")
        cv2.imwrite(f"{debug_prefix}_overlay.jpg", overlay)

        # LOGGER.info("Measured angles (vertical, horizontal): (%.3f, %.3f)", angle_v, angle_h)

    _ = slant  # placeholder for future behavior

    return CrossCenterResult(
        center, (angle_v, angle_h), overlay, q_ratio_list, roi_methods_tuple, roi_traces, saturation_tuple,
        windows_tuple,
    )


# ----------------------------
# Test harness (optional)
# ----------------------------
def run_demo() -> None:
    """
    Example workflow used by the original file.

    This is intentionally not executed on import.
    """
    logging.basicConfig(level=logging.INFO)

    prefix_in = Path("./test_in")
    prefix_out = Path("./test_out")
    clear_folder(prefix_out, create=True)

    images = [
        "blob-05.jpg",
        "blob-06.jpg",
        "blob-08.jpg",
        "blob-09.jpg",
        "blob-11.jpg",
        "blob-12.jpg",
    ]

    # Approximate percent centers and crop sizes (original values)
    cx_list = [0.37, 0.6, 0.27, 0.6, 0.3, 0.6, 0.5]
    cy_list = [0.5, 0.7, 0.35, 0.4, 0.4, 0.4, 0.4]
    x_pct_list = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.6]
    y_pct_list = [0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

    first = cv2.imread(str(prefix_in / images[0]))
    if first is None:
        raise FileNotFoundError(f"Cannot read image: {prefix_in / images[0]}")
    height, width = first.shape[:2]

    cx_px = pct_list_to_int_list(cx_list, width)
    cy_px = pct_list_to_int_list(cy_list, height)
    x_crop_px = pct_list_to_int_list(x_pct_list, width)
    y_crop_px = pct_list_to_int_list(y_pct_list, height)

    s="Summary:\n"

    for i, fname in enumerate(images, start=1):
        img = cv2.imread(str(prefix_in / fname))
        if img is None:
            LOGGER.warning("Skipping unreadable image: %s", fname)
            continue

        debug_prefix = str(prefix_out / f"debug-{i}")
        position, angles = find_cross_center(
            image=img,
            crop_center=(cx_px[i - 1], cy_px[i - 1]),
            crop_size=(x_crop_px[i - 1], y_crop_px[i - 1]),
            roi_size=(500, 20),
            debug=True,
            debug_prefix=debug_prefix,
        )
        LOGGER.info("Initial position=%s angles=%s", position, angles)

        if position == None:
            LOGGER.warning(f"file ({fname}) failed to find reliable centerpoint.  Prefix:{prefix_in}")
            continue

        crop_pixels = min(x_crop_px[i - 1], y_crop_px[i - 1])
        position2, angles2 = find_cross_center(
            image=img,
            crop_center=(int(position[0]), int(position[1])),
            crop_size=(crop_pixels, crop_pixels),
            roi_size=(500, 20),
            debug=True,
            debug_prefix=f"{debug_prefix}_refined",
        )
        LOGGER.info("Refined position=%s angles=%s", position2, angles2)

        x0 = position[0]
        x1 = position2[0]
        y0 = position[1]
        y1 = position2[1]

        s+=f"fname:{fname}\tprefix:{debug_prefix}\tx0:{x0}\ty0:{y0}\tx1:{x1}\ty1:{y1}\t\n"

        if position == None:
            print(f"file ({fname}) failed.  Prefix:{prefix_in}")
            continue
        
    print(s)



if __name__ == "__main__":
    # Uncomment to run the demo workflow.
    run_demo()
    pass
