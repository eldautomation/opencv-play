import logging
import cv2
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import warnings
import os
import shutil
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
            path.mkdir(parents=True, existSyok=True)
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


def pct_list_to_int_list(pct_list,scale):
    """
    Inputs
    ------
    pct_list: list of floats
        Input a list of floating point numbers, as percentages
    scale: Integer
        A scaling factor. 
    """

    if type(pct_list) != type ([]):
        raise ValueError("pct_list must be entered as a list")
    if type(pct_list[0]) != type (0.1):
        raise ValueError("Entries of pct_lsit must be floats")
    if type(scale) != type (3):
        raise ValueError("Scaling factor must be an integer")
    
    int_list = []
    for entry in pct_list: 
        val = math.floor(entry*scale/2)*2
        val = int(val)
        int_list.append(val)
    
    return int_list

def draw_box_on_image(
    image: np.ndarray,
    corner1: tuple[int, int],
    corner2: tuple[int, int],
    width: int,
    color: tuple[int, int, int] = (0, 255, 255)  # yellow in BGR
) -> np.ndarray:
    """
    Draw a rectangle on an image. If the input is grayscale, convert to BGR first.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale (H,W) or color (H,W,3).
    corner1 : (int, int)
        (x, y) corner.
    corner2 : (int, int)
        (x, y) opposite corner.
    width : int
        Rectangle line thickness in pixels.
    color : (int, int, int)
        BGR color tuple. Default is yellow (0,255,255).

    Returns
    -------
    out : np.ndarray
        Color (BGR) image with rectangle drawn.
    """
    if image is None:
        raise ValueError("image is None")
    if width <= 0:
        raise ValueError("width must be > 0")

    # Convert to BGR if grayscale
    if image.ndim == 2:
        gray_u8 = image.astype(np.uint8)
        out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        # If color image is float64, convert to uint8 similarly (optional but safer)
        if image.dtype == np.uint8:
            out = image.copy()
        else:
            # normalize each channel to uint8 conservatively
            out = np.dstack([_to_uint8_gray(image[:, :, c]) for c in range(3)])
    else:
        raise ValueError("image must be grayscale (H,W) or BGR color (H,W,3)")

    if image.ndim == 2:
        if image.dtype != np.uint8:
            gray_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            gray_u8 = image
        out = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


    H, W = out.shape[:2]
    x1, y1 = int(corner1[0]), int(corner1[1])
    x2, y2 = int(corner2[0]), int(corner2[1])

    # Correct ordering
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    # Clip to bounds (OpenCV rectangle uses inclusive end coords)
    x_min = max(0, min(W - 1, x_min))
    x_max = max(0, min(W - 1, x_max))
    y_min = max(0, min(H - 1, y_min))
    y_max = max(0, min(H - 1, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Rectangle corners produce an empty/degenerate box after clipping.")

    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, thickness=width)
    return out

def sdrm(values, search_size, debug, debug_prefix):
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
        plot_xy_positions(padded_values,x_label = "Pixel Value", y_label = "Pixel Intensity", title = "Padded Values", save_path = debug_prefix+"23.jpg",dpi = 300)
        plot_xy_positions(flipped_values,x_label = "Pixel Value", y_label = "PIxel Intensity", title = "Flipped Values", save_path = debug_prefix+"22.jpg",dpi = 300)
        plot_xy_positions(sums2,x_label = "Pixel Value", y_label = "Rss Values", title = "Rss value", save_path = debug_prefix+"24.jpg",dpi = 300)
        plot_xy_positions(sums3,x_label = "Pixel Value, corrected", y_label = "Rss Values", title = "Rss value", save_path = debug_prefix+"25.jpg",dpi = 300)

        print(f"search_size:\t{search_size}")
        # print(f"min point1:{min_point}")
        # print(f"min point2:{min_point2}")
        print(f"min point3:{min_point3}")

        # print(f"sums:\n{sums}")
        # print(f"sums2:\n{sums2}")

    peak_from_left  = min_point[0]
    peak_from_center = min_point3[0]

    return peak_from_left, peak_from_center


def plot_xy_positions(positions: np.ndarray,
                      x_label: str = "Index",
                      y_label: str = "Value",
                      title: str = "Position Plot",
                      save_path: str = "example.jpg",
                      dpi: int = 300):
    """
    Plot an array of (X, Y) positions and save as a JPG.

    Parameters
    ----------
    positions : np.ndarray (N, 2)
        Array of floating-point (X, Y) values.
    x_label : str
        Label for the X axis.
    y_label : str
        Label for the Y axis.
    title : str
        Title of the graph.
    save_path : str
        Output filename.
    dpi : int
        Resolution of saved image.
    """

    if positions is None:
        raise ValueError("positions is None")

    positions = np.asarray(positions, dtype=np.float64)

    # if positions.ndim != 1 or positions.shape[1] != 1:
    #     raise ValueError("positions must be an array of shape (N, 1)")

    # print(f"shape\t{positions.shape}")
    # print(f"size\t{positions.size}")
    # print(f"num dims\t{positions.ndim}")

    if positions.ndim == 1:
        y = positions
        x = range(len(positions))
    elif positions.ndim == 2:
        x = positions[:,0]
        y = positions[:,1]

        # a = positions[:,0]
        # b = positions[:,1]

        # print("positions")
        # print(positions)

        # print(f"a\t{a}")
        # print(f"b\t{b}")
    else: 
        raise ValueError("Array must have either 1 or two series of data")

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, format="jpg")
    plt.close()

def find_center_pixel(
    image: np.ndarray,
    center_position: tuple[float, float],
    search_size: int,
    slant: bool = False,
    search_method: str = "sdrm",
    debug: bool = False,
    debug_prefix: str = "dbg"
):
    """
    Estimate the center of a 1D intensity distribution - the source is assumed to be a symmetric narrow slit. 
    inside a narrow strip image (e.g., 5 px tall x 50 px wide).

    Intended use:
      - You have a narrow strip whose long axis defines the 1D coordinate (x by default).
      - The intensity across that axis forms an approximately Gaussian peak.
      - Return the peak center position along the 1D axis, and mapped back to 2D.

    Inputs
    ------
    image : np.ndarray
        Strip image, shape (H,W) or (H,W,C). Typically small H, larger W.
    center_position : (float, float)
        Approximate (x, y) in *strip coordinates* (not full-image unless you pass a cropped strip).
    search_size : int
        Half-width of search region (pixels) along the 1D axis, centered at center_position.
        Example: search_size=20 searches [cx-20, cx+20].
    slant : bool
        If True, allow/expect the peak to be slanted across rows and account for it (placeholder).
    search_method : str
        Method name (placeholder). Common options:
          - "argmax": max of the 1D profile
          - "moments": center-of-mass on the 1D profile
          - "gaussian_fit": fit a 1D Gaussian (requires optimizer)
    debug : bool
        If True, save/print intermediate arrays (placeholder hooks).
    debug_prefix : str
        Prefix for debug artifacts.

    Outputs
    -------
    pos_1d : float
        Center position along the strip's 1D axis (x-axis by default).
    pos_2d : (float, float)
        Corresponding (x, y) position in strip coordinates.
        Typically y is center_position[1] unless slant logic is implemented.
    """

    # -------------------------
    # 0) Validate inputs
    # -------------------------
    if image is None:
        raise ValueError("image is None")
    if image.ndim not in (2,-1):
        raise ValueError("image must be 2D (grayscale)")
    if search_size <= 0:
        raise ValueError("search_size must be > 0")
    if (not isinstance(center_position, tuple)) or len(center_position) != 2:
        raise ValueError("center_position must be a tuple (x, y)")
    if not isinstance(search_method,str):
        raise ValueError(f"search method must be a string")


    H, W = image.shape[:2]
    cx, cy = float(center_position[0]), float(center_position[1])


    # -------------------------
    # 2) Define 1D axis and extract 1D profile
    # -------------------------
    # Convention: 1D axis is x (columns). Average along rows to get a 1D profile.
    # For a narrow strip (small H), this is typical.
    profile_full = image.mean(axis=0)  # shape (W,)

    if debug:
        pass
        # print(f"cx:{cx}")
        # print(f"search_size:{search_size}")
        # print(f"W:{W}")
        # print(f"H:{H}")

    # Define search window around cx
    x0 = int(np.floor(cx - search_size))
    x1 = int(np.ceil(cx + search_size)) + 1  # inclusive -> exclusive
    # x0 = max(0, min(W, x0))
    # x1 = max(0, min(W, x1))

    if x1 <= x0:
        raise ValueError("Search window is empty after clipping.")

    profile = profile_full[x0:x1]
    xs = np.arange(x0, x1, dtype=np.float64)

    # -------------------------
    # 3) Estimate peak center (placeholder implementations)
    # -------------------------

    if search_method.lower() == "argmax":
        # Peak at maximum
        idx = int(np.argmax(profile))
        pos_1d = float(xs[idx])

    elif search_method.lower() == "moments":
        # Center-of-mass (works well for roughly Gaussian peaks)
        # Shift baseline to reduce bias from DC offset:
        baseline = float(np.min(profile))
        wts = profile - baseline
        wts[wts < 0] = 0.0

        if float(np.sum(wts)) <= 1e-12:
            # Fallback if weights are all zero
            idx = int(np.argmax(profile))
            pos_1d = float(xs[idx])
        else:
            pos_1d = float(np.sum(xs * wts) / np.sum(wts))

    elif search_method.lower() == "gaussian_fit":
        # Placeholder: implement a 1D Gaussian fit (A*exp(-(x-mu)^2/(2*sigma^2))+b)
        # Typically requires scipy.optimize.curve_fit or a custom optimizer.
        raise NotImplementedError("gaussian_fit not implemented in skeleton.")

    elif search_method.lower() == "sdrm":
        # Placeholder: implement a 1D Gaussian fit (A*exp(-(x-mu)^2/(2*sigma^2))+b)
        # Typically requires scipy.optimize.curve_fit or a custom optimizer.
        # raise NotImplementedError("symmetric diff rms minimum (sdrm) not implemented in skeleton.")
        warnings.warn("symmetric diff rms minimum (sdrm) not implemented yet")
        plot_xy_positions(profile_full,x_label = "Pixel Index", y_label = "Pixel Intensity", title = "Position Plot", save_path = debug_prefix+"21.jpg",dpi = 300)
        peak_from_left,peak_from_center = sdrm(profile_full,search_size, debug = True, debug_prefix = debug_prefix)
        pos_left = peak_from_left
        pos_center = peak_from_center

        if debug: 
            print(f"pos_left is: {pos_left}")
            print(f"pos_center is: {pos_center}")

    else:
        raise ValueError(f"Unknown search_method: {search_method}")


    # -------------------------
    # 5) Debug hooks
    # -------------------------
    if debug:
        pass
        # print("debug mode active - pixel centerline search")
        # Keep this lightweight; do not print huge arrays in real use.
        # printf(f"[{debug_prefix}] strip size: (H={H}, W={W})")
        # print(f"[{debug_prefix}] search window: x0={x0}, x1={x1}, method={search_method}")
        # Optionally save a plot/overlay outside this skeleton.

    return pos_center, pos_left

def find_cross_center(
    image: np.ndarray,
    crop_center: tuple[int, int],
    crop_size: tuple[int, int],
    roi_size: tuple[int, int],
    slant: bool = False,
    debug: bool = False,
    debug_prefix: str = "dbg"
):
    """
    Finds the center of a crosshair and estimate angles.

    Inputs
    ------
    image : np.ndarray
        Input image (grayscale or BGR), shape (H,W) or (H,W,3).
    crop_center : (int, int)
        (x, y) center of the crop region in pixel coordinates (full image coords).
    crop_size : (int, int)
        (width, height) of the crop region in pixels.
    roi_size : (int, int)
        (width, height) of the ROI used inside the crop (pixels).
    slant : bool
        If True, expect slanted crosshair arms and estimate both H and V angles.
        If False, assume near-orthogonal / near-horizontal+vertical.
    debug : bool
        If True, save intermediate debug images and logs.
    debug_prefix : str
        Prefix for debug outputs (filenames, etc.).

    Outputs
    -------
    position : (float, float)
        (x, y) center in full-image pixel coordinates.
    angle : (float, float)
        (H, V) angles in degrees.
    """

    # -------------------------
    # 0) Basic input validation
    # -------------------------
    if image is None:
        raise ValueError("No image given")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (BGR)")
    if (not isinstance(crop_center, tuple)) or len(crop_center) != 2:
        raise ValueError("crop_center must be a tuple (x, y)")
    if (not isinstance(crop_size, tuple)) or len(crop_size) != 2:
        raise ValueError("crop_size must be a tuple (w, h)")
    if (not isinstance(roi_size, tuple)) or len(roi_size) != 2:
        raise ValueError("roi_size must be a tuple (w, h)")

    cx_crop, cy_crop = int(crop_center[0]), int(crop_center[1])
    crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    roi_w, roi_h   = int(roi_size[0]), int(roi_size[1])

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("crop_size values must be > 0")
    if roi_w <= 0 or roi_h <= 0:
        raise ValueError("roi_size values must be > 0")

    H_img, W_img = image.shape[:2]

    # Convert image to greyscale before analysis.
    if image.ndim == 3:
        # Placeholder: caller can provide already-grayscale strips for speed
        # If needed, implement BGR->gray with cv2.cvtColor outside this skeleton.
        image = image.mean(axis=2).astype(np.float64)
    else:
        image = image.astype(np.float64)
    overlay = image.copy()

    # -------------------------
    # 1) Crop around crop_center
    # -------------------------
    # Compute crop bounds in full-image coords
    x0 = max(0, cx_crop - crop_w // 2)
    y0 = max(0, cy_crop - crop_h // 2)
    x1 = min(W_img, x0 + crop_w)
    y1 = min(H_img, y0 + crop_h)

    overlay = draw_box_on_image(overlay,corner1=(x0,y0),corner2=(x1,y1),width=3,color=(255,0,0))

    # Optionally re-adjust to preserve requested size if clipped
    if (x1 - x0) < crop_w:
        x0 = max(0, x1 - crop_w)
    if (y1 - y0) < crop_h:
        y0 = max(0, y1 - crop_h)

    crop = image[y0:y1, x0:x1]

    # -------------------------
    # 2) Define ROIs for line measurement
    # -------------------------
    # Default: center ROI within the crop

    center_t = (cx_crop,y0)
    center_r = (x0,cy_crop)
    center_b = (cx_crop,y1)
    center_l = (x1,cy_crop)

    roi_w = roi_size[0]
    roi_h = roi_size[1]

    roi_list = ["top","right","bottom","left"]
    roi_loc = [center_t,center_r,center_b,center_l]

    print(f"roi_w:{roi_w}")
    print(f"roi_h:{roi_h}")
    i = 0

    for sub_roi in roi_list: 
        debug_img = "0"
        if sub_roi == "top":
            x0 = int(roi_loc[i][0] - roi_w/2)
            x1 = int(roi_loc[i][0] + roi_w/2)
            y0 = int(roi_loc[i][1] - roi_h/2)
            y1 = int(roi_loc[i][1] + roi_h/2)
            roi_x = roi_loc[i][0]
            roi_y = roi_loc[i][1]
            debug_img = "1"
            roi_img = image[y0:y1,x0:x1]

        if sub_roi == "bottom":
            x0 = int(roi_loc[i][0] - roi_w/2)
            x1 = int(roi_loc[i][0] + roi_w/2)
            y0 = int(roi_loc[i][1] - roi_h/2)
            y1 = int(roi_loc[i][1] + roi_h/2)
            roi_x = roi_loc[i][0]
            roi_y = roi_loc[i][1]
            debug_img = "3"
            roi_img = image[y0:y1,x0:x1]

        if sub_roi == "right":
            x0 = int(roi_loc[i][0] - roi_h/2)
            x1 = int(roi_loc[i][0] + roi_h/2)
            y0 = int(roi_loc[i][1] - roi_w/2)
            y1 = int(roi_loc[i][1] + roi_w/2)
            roi_x = roi_loc[i][0]
            roi_y = roi_loc[i][1]
            debug_img = "2"
            roi_img = image[y0:y1,x0:x1]
            roi_img = roi_img.T

        if sub_roi == "left":
            x0 = int(roi_loc[i][0] - roi_h/2)
            x1 = int(roi_loc[i][0] + roi_h/2)
            y0 = int(roi_loc[i][1] - roi_w/2)
            y1 = int(roi_loc[i][1] + roi_w/2)
            roi_x = roi_loc[i][0]
            roi_y = roi_loc[i][1]
            debug_img = "4"
            roi_img = image[y0:y1,x0:x1]
            roi_img = roi_img.T

        # print(f"Roi corners: {x0}\t{x1}\t{y0}\t{y1}")


        i+=1

        roi = roi_img

        if debug & (sub_roi in ("top","bottom","right","left")): 
            print(f"\nWhich ROI:{sub_roi}")
            print(f"\t{x0}\t{y0}\t{x1}\t{y1}\t")
            h, w = image.shape[:2]
            print(f"\tImage Size:\theight:{h}\twidth:{w}")
            h, w = crop.shape[:2]
            print(f"\tCrop Size:\theight:{h}\twidth:{w}")
            print(f"\troi_center_x:{roi_x}")
            print(f"\roi_center_y:{roi_y}")

            overlay = draw_box_on_image(overlay,corner1=(x0,y0),corner2=(x1,y1),width=3,color=(0,255,255))


            cv2.imwrite(f"{debug_prefix}05.jpg",image)
            cv2.imwrite(f"{debug_prefix}10-{debug_img}.jpg",crop)
            cv2.imwrite(f"{debug_prefix}11.jpg",overlay)
            cv2.imwrite(f"{debug_prefix}12-{debug_img}.jpg",roi_img)


            # print(f"roi size:{roi.size}")
            # print(f"roi shape:{roi.shape}")
            # print(f"roi[:,0] size:{roi[:,0].size}")
            # print(f"roi[0,:] size:{roi[0,:].size}")
            # print("max",max(roi.shape))

        pos_center, pos_left = find_center_pixel(
            image = roi,
            center_position = (roi_x,roi_y),
            search_size = int(max(roi.shape)/2-2),
            slant = False,
            search_method = "sdrm",
            debug = True,
            debug_prefix = debug_prefix
        )
        
        measured_center = roi_x + pos_center
        print(f"measured_center is:{measured_center}")

    # -------------------------
    # 3) Crosshair detection (placeholder)
    # -------------------------
    # Suggested stages:
    #   a) preprocess (channel selection / grayscale / blur)
    #   b) segment crosshair pixels (threshold on "redness" or intensity)
    #   c) skeletonize OR edge detect
    #   d) estimate center:
    #        - fit two line models and intersect, OR
    #        - compute symmetry score, OR
    #        - template match
    #   e) estimate angles:
    #        - from fitted line orientations
    #
    # Return center in ROI coords first, then translate back:
    #
    # center_roi_x, center_roi_y = ...
    # angle_h, angle_v = ...

    # raise NotImplementedError(
    #     "find_cross_center: detection pipeline not implemented yet. "
    #     "Add preprocessing, segmentation, line fitting, and intersection."
    # )


    # -------------------------
    # 4) Translate ROI result to full-image coords (example)
    # -------------------------
    # center_in_crop_x = rx0 + center_roi_x
    # center_in_crop_y = ry0 + center_roi_y
    # center_full_x = x0 + center_in_crop_x
    # center_full_y = y0 + center_in_crop_y
    #
    # position = (float(center_full_x), float(center_full_y))
    # angle = (float(angle_h), float(angle_v))
    #
    position = "lol"
    angle = "jk"
    return position, angle

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

        crop_pixels = min(x_crop_px[i - 1], y_crop_px[i - 1])
        print("crop pixelx:",crop_pixels)
        print("position:",position)
        position2, angles2 = find_cross_center(
            image=img,
            crop_center=(int(position[0]), int(position[1])),
            crop_size=(crop_pixels, crop_pixels),
            roi_size=(500, 20),
            debug=True,
            debug_prefix=f"{debug_prefix}_refined",
        )
        LOGGER.info("Refined position=%s angles=%s", position2, angles2)


if __name__ == "__main__":
    # Uncomment to run the demo workflow.
    run_demo()
    pass