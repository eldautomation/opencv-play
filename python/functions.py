import cv2
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import warnings


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

def sdrm(values, search_size, debug, debug_prefix):
    """

    Inputs: 
    Array: 
        A 1D array of floating point number. 
        This is the data who's center we want to find. 
    search_size: int 
        This is the size of the window we want to search. 
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
        print(f"min point1:{min_point}")
        print(f"min point2:{min_point2}")
        print(f"min point3:{min_point3}")

        # print(f"sums:\n{sums}")
        # print(f"sums2:\n{sums2}")


    return min_point3


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

    print(f"shape\t{positions.shape}")
    print(f"size\t{positions.size}")
    print(f"num dims\t{positions.ndim}")

    if positions.ndim == 1:
        y = positions
        x = range(len(positions))
    elif positions.ndim == 2:
        pass
        x = positions[:,0]
        y = positions[:,1]

        a = positions[:,0]
        b = positions[:,1]

        print("positions")
        print(positions)

        print(f"a\t{a}")
        print(f"b\t{b}")


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

    # Define search window around cx
    x0 = int(np.floor(cx - search_size))
    x1 = int(np.ceil(cx + search_size)) + 1  # inclusive -> exclusive
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))

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
        mid_pos = sdrm(profile_full,search_size, debug = True, debug_prefix = debug_prefix)
        pos_1d = "Fake"
        if debug: 
            print(f"mis_pos is: {mid_pos}")

    else:
        raise ValueError(f"Unknown search_method: {search_method}")



    # -------------------------
    # 4) Map 1D position back to 2D position (placeholder)
    # -------------------------
    # For non-slanted distribution, y can be taken as cy (or strip center).
    if not slant:
        pos_2d = (pos_1d, cy)
    else:
        # Placeholder: if the peak is slanted across rows, you could:
        #   - compute per-row peak centers, fit a line, then evaluate at desired y
        #   - or rotate the strip to deskew before measuring
        # For now, return same y.
        pos_2d = (pos_1d, cy)

    # -------------------------
    # 5) Debug hooks
    # -------------------------
    if debug:
        print("debug mode active - pixel centerline search")
        # Keep this lightweight; do not print huge arrays in real use.
        # print(f"[{debug_prefix}] strip size: (H={H}, W={W})")
        # print(f"[{debug_prefix}] search window: x0={x0}, x1={x1}, method={search_method}")
        # print(f"[{debug_prefix}] pos_1d={pos_1d:.3f}, pos_2d=({pos_2d[0]:.3f},{pos_2d[1]:.3f})")
        # Optionally save a plot/overlay outside this skeleton.

    return pos_1d, pos_2d

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

    # -------------------------
    # 1) Crop around crop_center
    # -------------------------
    # Compute crop bounds in full-image coords
    x0 = max(0, cx_crop - crop_w // 2)
    y0 = max(0, cy_crop - crop_h // 2)
    x1 = min(W_img, x0 + crop_w)
    y1 = min(H_img, y0 + crop_h)

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
    center_r = (x0,cx_crop)
    center_b = (cx_crop,y1)
    center_l = (x1,cx_crop)

    roi_w = roi_size[0]
    roi_h = roi_size[1]
    roi_x = 100
    roi_y = 150

    roi = crop # stand-in

    if debug: 
        h, w = image.shape[:2]
        print(f"Image Size:\theight:{h}\twidth:{w}")
        h, w = crop.shape[:2]
        print(f"Crop Size:\theight:{h}\twidth:{w}")


    find_center_pixel(
        image = roi,
        center_position = (roi_x,roi_y),
        search_size = 500,
        slant = False,
        search_method = "sdrm",
        debug = True,
        debug_prefix = debug_prefix
    )


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

def run():
    pass
    # cleanup folders
    
    # Generate list of images to process. 
    images = []
    # images.extend(["blob-5.jpg","blob-6.jpg","blob-11.jpg","blob-12.jpg"])
    prefix_in = "./test_in/"
    prefix_out = "./test_out/"
    images.extend(["blob-5.jpg"])

    cx_list=[0.4,0.6,0.25,0.6]
    cy_list=[0.35,0.7,0.35,0.4]
    x_pct_list=[0.6,0.6,0.6,0.6]
    y_pct_list=[0.6,0.6,0.6,0.6]
    line_width_list=[3,3,3,3]


    # Convert percentages to pixels
    img = cv2.imread(prefix_in+images[0])
    height, width, channels = img.shape
    cx_list = pct_list_to_int_list(cx_list,width)
    cy_list = pct_list_to_int_list(cy_list,height)
    x_crop_list = pct_list_to_int_list(x_pct_list,width)        
    y_crop_list = pct_list_to_int_list(y_pct_list,height)

    # Begin running for each image.
    i=0
    for n in images: 
        i+=1

        img = cv2.imread(prefix_in+n)
        debug_image_prefix = f"{prefix_out}debug-{i}-"

        find_cross_center(
            image=img,
            crop_center = (cx_list[i],cy_list[i]),
            crop_size = (x_crop_list[i],y_crop_list[i]),
            roi_size = (50,10),
            slant = False,
            debug = True,
            debug_prefix = debug_image_prefix
        )



run()
