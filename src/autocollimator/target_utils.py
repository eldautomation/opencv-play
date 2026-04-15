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

from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)


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


def ensure_gray_f64(image: np.ndarray) -> np.ndarray:
    """
    Ensure a grayscale float64 image.

    Accepts:
      - grayscale (H,W)
      - BGR color (H,W,3): converted using cv2

    Returns
    -------
    np.ndarray float64 of shape (H,W)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image is empty")

    if image.ndim == 2:
        return np.asarray(image, dtype=np.float64)

    if image.ndim == 3 and image.shape[2] == 3:
        # Use OpenCV luminance conversion for stability
        u8 = ensure_bgr_u8(image)
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
) -> int:
    """
    Symmetric Difference RMS Minimum (SDRM) center offset estimator for 1D signals.

    This function preserves the original SDRM algorithm:
    - Reverse the signal
    - Pad both original and reversed signals with a baseline value (mean)
    - For each offset in [-search_size, +search_size], compute RSS between padded arrays
    - Return the offset (from center) that minimizes RSS

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

    Returns
    -------
    peak_from_center:
        Integer offset (negative/positive) from center pixel that minimizes RSS.
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

    if not isinstance(q_limit,(float,np.float32, np.float64)):
        raise TypeError(f"q_limit must be a float, got {type(q_limit).__name__}")
    if not 0 < q_limit < 1:
        raise ValueError(f"q_limit must be between 0 and 1, got {q_limit}")

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
        rss = math.sqrt(float(np.sum(diff * diff)))

        rss_by_i.append((i, rss))

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
    peak_from_center = int(min_k[0])
    if r1 > q_limit:
        peak_from_center = None

    return peak_from_center, r1

def find_center_pixel(
    image: np.ndarray,
    center_position: tuple[float, float],
    search_size: int,
    q_limit:float,
    *,
    search_method: str = "sdrm",
    debug: bool = False,
    debug_prefix: str = "dbg",
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
            best_k,q_ratio = sdrm_2(chunk,max_off,q_limit=q_limit,debug = debug, debug_prefix = f"{debug_prefix}")
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
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Find the center of a crosshair and estimate tilt angles.

    Notes
    -----
    This implementation uses four ROIs (top/right/bottom/left) around a crop region.
    Each ROI is reduced to a 1D profile and centered using ``find_center_pixel``.
    """

    LOGGER.info(
        "Running center-finding util with crop_center=%s crop_size=%s roi_size=%s debug=%s",
        crop_center,
        crop_size,
        roi_size,
        debug,
    )

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

    gray = ensure_gray_f64(image)
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
    }

    centers_measured: dict[str, tuple[float, float]] = {}

    # LOGGER.debug(f"Debug value in find_cross_center: {debug}")
    # LOGGER.debug(f"ROI Centers: {roi_centers.items()}")

    # for name,(rx,ry) in roi_centers.items():
    #         LOGGER.debug(f"ROI Name: {name}")

    measure_fail = 0
    q_ratio_list = []
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
                debug_prefix=f"{debug_prefix}_{name}",
            )

            # LOGGER.debug(f"Quality ratio of center finding: {q_ratio}")
            if pos_x == None:
                LOGGER.warning(f"For ROI {name}, Quality too low for reliable center finding: {q_ratio}")
                measure_fail = True
                pos_x=0 # Note - will not be used for computation, because "measure_fail" should trigger

            measured = (float(x0) + pos_x, float(y0) + center_in_roi[1])

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
                debug_prefix=f"{debug_prefix}_{name}",
            )

            if pos_x == None:
                LOGGER.warning(f"For ROI {name}, Quality too low for reliable center finding: {q_ratio}")
                measure_fail = True
                pos_x=0 # Note - will not be used for computation, because "measure_fail" should trigger

            # pos_x is along original y because of transpose
            measured = (float(x0) + (roi0.shape[1] / 2.0), float(y0) + pos_x)

        q_ratio_list.append(float(round(q_ratio,3)))
        centers_measured[name] = measured

        if debug:
            overlay = draw_box_on_image(overlay, (x0, y0), (x1 - 1, y1 - 1), width=2, color=(0, 255, 255))
            overlay = draw_crosshair(overlay, measured, leg_length=10, width=2, color=(0, 255, 0))

        if debug:
            LOGGER.debug("%s ROI bounds: (%d,%d)-(%d,%d) measured=%s", name, x0, y0, x1, y1, measured)

    if debug:
        cv2.imwrite(f"{debug_prefix}_overlay.jpg", overlay)

    if measure_fail: 
        LOGGER.warning(f"Quality too low for reliable center finding: - No center computed")
        return (None,None), (None,None),overlay,q_ratio_list

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

    return center, (angle_v, angle_h), overlay, q_ratio_list


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
