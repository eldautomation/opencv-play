import cv2
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt

def _auto_binary_crosshair(gray):
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ensure crosshair is white in center region
    h, w = gray.shape
    y0, y1 = int(0.3*h), int(0.7*h)
    x0, x1 = int(0.3*w), int(0.7*w)
    if np.sum(bw[y0:y1, x0:x1] == 255) < np.sum((255-bw)[y0:y1, x0:x1] == 255):
        bw = 255 - bw
    return bw

def _normalize_rho_theta(rho, theta):
    # enforce rho >= 0 for consistent clustering
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % (2*np.pi)
    # fold theta into [0, pi)
    theta = theta % np.pi
    return rho, theta

def _intersect_rho_theta(rho1, th1, rho2, th2):
    A = np.array([[math.cos(th1), math.sin(th1)],
                  [math.cos(th2), math.sin(th2)]], dtype=np.float64)
    b = np.array([rho1, rho2], dtype=np.float64)
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)

def skeletonize(binary_255: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization.
    Input:  binary uint8 image (0/255).
    Output: skeleton uint8 image (0/255).
    """
    img = (binary_255 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded

        if cv2.countNonZero(img) == 0:
            break

    return skel


def detect_crosshair_centerline_hough(img, debug_out=None):
    """
    Returns (cx, cy, theta_deg, overlay)
    where (cx, cy) is the intersection of the *centerlines* of the two thick bars.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overlay = img.copy()
    else:
        gray = img
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    bw = _auto_binary_crosshair(gray)

    # optional cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    edges = cv2.Canny(bw, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 140)
    if lines is None or len(lines) < 4:
        return None

    # collect and normalize candidates
    cand = []
    for i in range(min(80, len(lines))):
        rho, th = float(lines[i][0][0]), float(lines[i][0][1])
        rho, th = _normalize_rho_theta(rho, th)
        cand.append((rho, th))

    # cluster into two angle groups (two bars)
    # use k-means on angle represented on unit circle with period pi
    ang = np.array([[math.cos(2*t), math.sin(2*t)] for _, t in cand], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, _ = cv2.kmeans(ang, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    groups = [[], []]
    for (rho, th), lab in zip(cand, labels.ravel()):
        groups[int(lab)].append((rho, th))

    # For each group: estimate centerline by taking extreme rhos (two edges)
    center_lines = []
    for g in groups:
        if len(g) < 2:
            continue
        # Use median theta (robust) then find min/max rho within that group
        thetas = np.array([t for _, t in g], dtype=np.float64)
        th_med = float(np.median(thetas))

        # keep only lines close in angle to th_med (avoid contamination)
        g2 = [(rho, th) for rho, th in g if abs(((th - th_med + np.pi/2) % np.pi) - np.pi/2) < (8*np.pi/180)]
        if len(g2) < 2:
            g2 = g

        rhos = np.array([rho for rho, _ in g2], dtype=np.float64)
        rho_min = float(np.min(rhos))
        rho_max = float(np.max(rhos))
        rho_center = 0.5 * (rho_min + rho_max)

        center_lines.append((rho_center, th_med))

    if len(center_lines) != 2:
        return None

    (rho1, th1), (rho2, th2) = center_lines
    p = _intersect_rho_theta(rho1, th1, rho2, th2)
    if p is None:
        return None
    cx, cy = p

    # angle output: convert normal angle to line direction
    a1 = (math.degrees(th1) - 90.0)
    a2 = (math.degrees(th2) - 90.0)
    def norm(a): return (a + 90) % 180 - 90
    a1, a2 = norm(a1), norm(a2)
    theta_deg = a1 if abs(a1) <= abs(a2) else a2

    # debug overlay
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    def draw_line(rho, th, color):
        ct, st = math.cos(th), math.sin(th)
        x0, y0 = ct * rho, st * rho
        dx, dy = -st, ct
        L = 2 * max(h, w)
        p1 = (int(round(x0 + dx*L)), int(round(y0 + dy*L)))
        p2 = (int(round(x0 - dx*L)), int(round(y0 - dy*L)))
        cv2.line(overlay, p1, p2, color, 1)

    draw_line(rho1, th1, (0, 255, 0))
    draw_line(rho2, th2, (0, 255, 0))

    if debug_out:
        cv2.imwrite(debug_out, overlay)

    # print(f"type cy = {type(cy)}")
    # print(f"type cx = {type(cx)}")
    # print(f"type theta = {type(theta_deg)}")
    # print(f"type overlay = {type(overlay)}")

    # print(f"cy is:{cy}")
    # print(f"cx is:{cx}")
    # print(f"theta_deg is:{theta_deg}")
    # print(f"overlay is:{overlay}")

    return cx, cy, theta_deg, overlay

def red_only(img):

    img_float = img.astype(np.float32)

    blue = img_float[:, :, 0]
    red  = img_float[:, :, 2]

    # Average of red and blue
    avg_rb = (red + blue) / 2.0

    # Subtract average from red
    new_red = red - avg_rb

    # Floor negative values to 0
    new_red = np.maximum(new_red, 0)

    # Clip to valid 8-bit range
    new_red = np.clip(new_red, 0, 255).astype(np.uint8)

    # Create output image
    output = img.copy()
    output[:, :, 0] = 0
    output[:, :, 1] = 0    
    output[:, :, 2] = new_red

    return output

def rectangle_mid_subregions(
    corners,
    width:int,
    height:int,
):
    """
    return sub-region boxes, based on the input box.
    """
    midpoints = rectangle_edge_midpoints(corners)
    top_mid = midpoints[0]
    right_mid = midpoints[1]
    bottom_mid = midpoints[2]
    left_mid = midpoints[3]
    
    #Note: Corners are specified as upper-left, then bottom-right

    #right
    x0 = right_mid[0]-width/2
    x1 = right_mid[0]+width/2

    y0 = right_mid[1]-height/2
    y1 = right_mid[1]+height/2
    right_corners = ((x0,y0),(x1,y1))

    #top
    x0 = top_mid[0]-height/2
    x1 = top_mid[0]+height/2

    y0 = top_mid[1]-width/2
    y1 = top_mid[1]+width/2
    top_corners = ((x0,y0),(x1,y1))

    #left
    x0 = left_mid[0]-width/2
    x1 = left_mid[0]+width/2

    y0 = left_mid[1]-height/2
    y1 = left_mid[1]+height/2
    left_corners = ((x0,y0),(x1,y1))


    #bottom
    x0 = bottom_mid[0]-height/2
    x1 = bottom_mid[0]+height/2

    y0 = bottom_mid[1]-width/2
    y1 = bottom_mid[1]+width/2
    bottom_corners = ((x0,y0),(x1,y1))

    return (right_corners,top_corners,left_corners,bottom_corners)



def visualize_roi_percent(
    img: np.ndarray,
    center_x_pct: float,
    center_y_pct: float,
    size_x_pct: float,
    size_y_pct: float,
    line_width: int
):
    """
    Draw a yellow ROI box on the image using percentage-based specification.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or BGR).
    center_x_pct : float
        Center X location as fraction of width (0.0–1.0).
    center_y_pct : float
        Center Y location as fraction of height (0.0–1.0).
    size_x_pct : float
        Box width as fraction of image width (0.0–1.0).
    size_y_pct : float
        Box height as fraction of image height (0.0–1.0).
    line_width : int
        Border thickness in pixels.

    Returns
    -------
    out_img : np.ndarray
        Image with yellow ROI border drawn.
    (top_left, bottom_right) : tuple
        Each corner as (x, y) in pixel coordinates.
        bottom_right is inclusive (OpenCV drawing convention).
    """

    if img is None:
        raise ValueError("Input image is None.")

    H, W = img.shape[:2]

    # Convert to BGR if grayscale
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    # Clamp percentages to [0,1]
    center_x_pct = np.clip(center_x_pct, 0.0, 1.0)
    center_y_pct = np.clip(center_y_pct, 0.0, 1.0)
    size_x_pct   = np.clip(size_x_pct,   0.0, 1.0)
    size_y_pct   = np.clip(size_y_pct,   0.0, 1.0)

    # Convert to pixel units
    cx = center_x_pct * W
    cy = center_y_pct * H
    box_w = size_x_pct * W
    box_h = size_y_pct * H

    # Compute corners
    x0 = int(round(cx - box_w / 2))
    y0 = int(round(cy - box_h / 2))
    x1 = int(round(cx + box_w / 2))
    y1 = int(round(cy + box_h / 2))

    # Clip to image bounds
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))

    # Draw yellow rectangle (BGR: 0,255,255)
    cv2.rectangle(
        out,
        (x0, y0),
        (x1, y1),
        (0, 255, 255),
        thickness=line_width
    )

    return out, ((x0, y0), (x1, y1))

def crop_subimage(img: np.ndarray, corners):
    """
    Crop a subimage from img using corner coordinates.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or color).
    corners : tuple
        ((x1, y1), (x2, y2))
        Intended as top-left and bottom-right, but order is corrected if needed.

    Returns
    -------
    subimg : np.ndarray
        Cropped image region.
    ((x_min, y_min), (x_max, y_max)) : tuple
        Corrected corner coordinates actually used.
        x_max, y_max are exclusive (Python slicing convention).
    """

    if img is None:
        raise ValueError("Input image is None.")

    (x1, y1), (x2, y2) = corners

    # Ensure numeric
    x1, y1 = float(x1), float(y1)
    x2, y2 = float(x2), float(y2)

    H, W = img.shape[:2]

    # Correct ordering
    x_min = int(round(min(x1, x2)))
    x_max = int(round(max(x1, x2)))
    y_min = int(round(min(y1, y2)))
    y_max = int(round(max(y1, y2)))

    # Clip to image bounds
    x_min = max(0, min(W, x_min))
    x_max = max(0, min(W, x_max))
    y_min = max(0, min(H, y_min))
    y_max = max(0, min(H, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid crop region after correction/clipping.")

    subimg = img[y_min:y_max, x_min:x_max]

    return subimg, ((x_min, y_min), (x_max, y_max))

def draw_crosshair_at_point(img: np.ndarray, point, length: int):
    """
    Draw a 1-pixel wide crosshair centered at a given point.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or BGR).
    point : tuple
        (x, y) pixel coordinates of the center.
    length : int
        Total length of each crosshair arm (in pixels).

    Returns
    -------
    out : np.ndarray
        Image with crosshair drawn.
    """

    if img is None:
        raise ValueError("Input image is None.")
    if length <= 0:
        raise ValueError("Length must be positive.")

    H, W = img.shape[:2]
    cx, cy = int(round(point[0])), int(round(point[1]))

    # Convert to BGR if grayscale
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    half = length // 2

    # Horizontal line
    x0 = max(0, cx - half)
    x1 = min(W - 1, cx + half)
    if 0 <= cy < H:
        cv2.line(out, (x0, cy), (x1, cy), (0, 255, 255), thickness=1)

    # Vertical line
    y0 = max(0, cy - half)
    y1 = min(H - 1, cy + half)
    if 0 <= cx < W:
        cv2.line(out, (cx, y0), (cx, y1), (0, 255, 255), thickness=1)

    return out

def draw_box_at_corners(img: np.ndarray, boxes, line_width: int, color=(0, 255, 255)):
    """
    Draw one or more rectangles on an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or BGR).
    boxes : list
        List of boxes, each in the form ((x0, y0), (x1, y1)),
        intended as upper-left and lower-right but corrected if needed.
    line_width : int
        Rectangle border thickness in pixels.
    color : tuple
        BGR color for the rectangle (default: yellow).

    Returns
    -------
    out : np.ndarray
        Output image with rectangles drawn.
    """

    if img is None:
        raise ValueError("Input image is None.")
    if line_width <= 0:
        raise ValueError("line_width must be a positive integer.")

    H, W = img.shape[:2]

    # Convert to BGR if grayscale
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    if boxes is None:
        return out

    for box in boxes:
        # print("box is:",box)
        (x0, y0), (x1, y1) = box

        # Coerce to ints and correct ordering
        x0 = int(round(x0)); y0 = int(round(y0))
        x1 = int(round(x1)); y1 = int(round(y1))

        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))

        # Clip to bounds
        x_min = max(0, min(W - 1, x_min))
        x_max = max(0, min(W - 1, x_max))
        y_min = max(0, min(H - 1, y_min))
        y_max = max(0, min(H - 1, y_max))

        # Skip degenerate boxes
        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, thickness=line_width)

    return out


def rectangle_edge_midpoints(corners):
    """
    Given two rectangle corners, return the midpoints of the four edges.

    Parameters
    ----------
    corners : tuple
        ((x1, y1), (x2, y2))
        Intended as upper-left and lower-right, but ordering is corrected if needed.

    Returns
    -------
    midpoints : list of tuples
        [top_mid, right_mid, bottom_mid, left_mid]
        Each as (x, y) in integer pixel coordinates.
    """

    (x1, y1), (x2, y2) = corners

    # Ensure numeric
    x1, y1 = float(x1), float(y1)
    x2, y2 = float(x2), float(y2)

    # Correct ordering if needed
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    # Compute and round midpoints
    top_mid = (
        int(round((x_min + x_max) / 2.0)),
        int(round(y_min))
    )

    right_mid = (
        int(round(x_max)),
        int(round((y_min + y_max) / 2.0))
    )

    bottom_mid = (
        int(round((x_min + x_max) / 2.0)),
        int(round(y_max))
    )

    left_mid = (
        int(round(x_min)),
        int(round((y_min + y_max) / 2.0))
    )

    return [top_mid, right_mid, bottom_mid, left_mid]


def crop_to_box(img, box):
    """
    Function 1:
    Crop image to the rectangle defined by corners.

    Inputs
    ------
    img : np.ndarray
    box : ((x0,y0),(x1,y1))  OR  (x0,y0,x1,y1)

    Returns
    -------
    cropped : np.ndarray
    box_used : ((x_min,y_min),(x_max,y_max))  where x_max,y_max are exclusive (Python slicing)
    """
    if img is None:
        raise ValueError("img is None")

    H, W = img.shape[:2]

    if len(box) == 4:
        x0, y0, x1, y1 = box
    else:
        (x0, y0), (x1, y1) = box

    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)

    x_min = int(round(min(x0, x1)))
    x_max = int(round(max(x0, x1)))
    y_min = int(round(min(y0, y1)))
    y_max = int(round(max(y0, y1)))

    # clip (x_max/y_max treated as exclusive)
    x_min = max(0, min(W, x_min))
    x_max = max(0, min(W, x_max))
    y_min = max(0, min(H, y_min))
    y_max = max(0, min(H, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid crop after ordering/clipping.")

    cropped = img[y_min:y_max, x_min:x_max]
    return cropped, ((x_min, y_min), (x_max, y_max))

def row_average_profile(img):
    """
    Function 2:
    Average pixels along each row, producing a tall 1D list (length = image height).

    For color images, it first converts to grayscale, then averages across columns.

    Returns
    -------
    profile : np.ndarray shape (H,)
        Average intensity per row in [0,255] (float64).
    """
    if img is None:
        raise ValueError("img is None")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # mean across x (columns)
    profile = gray.astype(np.float64).mean(axis=1)
    return profile

def profile_to_greyscale_and_plot(profile, title="Row average profile"):
    """
    Function 3:
    Convert the tall list (profile) to a grayscale image representation and plot it.

    - Grayscale "image": height = len(profile), width = 20 (for visibility)
    - Graph: X axis is the row index (y coordinate), Y axis is average value
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("profile must be a non-empty 1D array")

    # Normalize to 0..255 for grayscale visualization
    p_min, p_max = float(profile.min()), float(profile.max())
    if abs(p_max - p_min) < 1e-12:
        norm = np.zeros_like(profile, dtype=np.uint8)
    else:
        norm = np.clip((profile - p_min) * 255.0 / (p_max - p_min), 0, 255).astype(np.uint8)

    # Make a visible grayscale strip image (H x Wstrip)
    strip_w = 20
    gray_strip = np.repeat(norm[:, None], strip_w, axis=1)  # (H, strip_w)

    # Plot grayscale strip and the profile
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gray_strip, cmap="gray", aspect="auto", origin="upper")
    ax0.set_title("Grayscale\n(strip)")
    ax0.set_xlabel("")
    ax0.set_ylabel("y (row)")
    ax0.set_xticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(profile.size)  # y index
    ax1.plot(y, profile)
    ax1.set_title(title)
    ax1.set_xlabel("y (row index)")
    ax1.set_ylabel("average intensity")
    ax1.grid(True)

    plt.tight_layout()
    # plt.show()

    return gray_strip

def plot_grey_strip_profile(gray_strip,
                            title="Gray Strip Profile",
                            save_path="profile.jpg",
                            dpi=300):
    """
    Plot the pixel intensity values from a grayscale strip image
    and save the figure as a JPG.

    Parameters
    ----------
    gray_strip : np.ndarray (H, W)
        Grayscale strip image (uint8).
    title : str
        Plot title.
    save_path : str
        Output JPG file path.
    dpi : int
        Resolution of saved image.
    """

    if gray_strip is None:
        raise ValueError("gray_strip is None")

    if gray_strip.ndim != 2:
        raise ValueError("gray_strip must be a 2D grayscale image")

    profile = gray_strip[:, 0].astype(np.float64)
    y = np.arange(len(profile))

    plt.figure(figsize=(6, 4))
    plt.plot(y, profile)
    plt.xlabel("y (row index)")
    plt.ylabel("Pixel intensity (0-255)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, format="jpg")
    # plt.show()


def pad_grey_strip_symmetric(gray_strip: np.ndarray, N: int) -> np.ndarray:
    """
    Function 4:
    Pad a gray strip symmetrically (top and bottom) with black pixels (value = 0).

    Parameters
    ----------
    gray_strip : np.ndarray (H, W)
        2D grayscale array.
    N : int
        Number of black pixels to add to BOTH top and bottom.

    Returns
    -------
    padded : np.ndarray
        Padded array with shape (H + 2N, W), same dtype as input.
    """

    if gray_strip is None:
        raise ValueError("gray_strip is None")
    if gray_strip.ndim != 2:
        raise ValueError("gray_strip must be a 2D array")
    if N < 0:
        raise ValueError("N must be >= 0")

    padded = np.pad(
        gray_strip,
        pad_width=((N, N), (0, 0)),  # (top,bottom), (left,right)
        mode='constant',
        constant_values=0
    )

    return padded

def flip_gray_strip(gray_strip: np.ndarray) -> np.ndarray:
    """
    Function 1:
    Flip the gray strip top-to-bottom.

    Input
    -----
    gray_strip : np.ndarray (H, W), uint8 or float

    Output
    ------
    gray_flipped : np.ndarray (H, W), same dtype as input
    """
    if gray_strip is None:
        raise ValueError("gray_strip is None")
    if gray_strip.ndim != 2:
        raise ValueError("gray_strip must be a 2D array (H, W)")

    return np.flipud(gray_strip)

def strip_squared_difference_sum(grey_strip: np.ndarray,flipped_strip: np.ndarray) -> float:
    """
    Function 2:
    Compute sum over all pixels of (gray_strip - flipped_strip(gray_strip))^2.

    Output is a scalar (float).
    """
    if grey_strip is None:
        raise ValueError("gray_strip is None")
    if grey_strip.ndim != 2:
        raise ValueError("gray_strip must be a 2D array (H, W)")

    if flipped_strip is None:
        raise ValueError("flipped_strip is None")
    if flipped_strip.ndim != 2:
        raise ValueError("flipped_strip must be a 2D array (H, W)")

    if grey_strip.shape != flipped_strip.shape:
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got {grey_strip.shape} and {flipped_strip.shape}."
        )

    g = grey_strip.astype(np.float64)
    gf = flipped_strip.astype(np.float64)

    # print(g)
    # print(gf)

    diff = g - gf
    sse = float(np.sum(diff * diff))
    return sse

def difference_rss(strip:np.array,n:int):
    """
   The function takes a 1-dimensional aray in, and compares it to itself, trying to find the central point. 
    input: 1-D strip, 
    """
    strip_flipped = np.flipud(strip)

    sums = []
    for i in range(n*2):
        j = 2*n-i
        print(f"i is: {i}")
        print(f"j is: {j}")
        strip_pad = np.pad(
            strip,
            pad_width = ((i,j),(0,0)), # (top,bottom,),(eft,right)
            constant_values = 0
        )
    
        flipped_pad = np.pad(
            strip_flipped,
            pad_width = ((j,i),(0,0)), # (top,bottom,),(eft,right)
            constant_values = 0
        )
        s = (i,strip_squared_difference_sum(strip_pad,flipped_pad))
        print(f"position,sum is: {s}")
        sums.append(s)
    
    sums_array = np.array(sums,dtype=np.float64)
    return sums_array

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

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be an array of shape (N, 2)")

    x = positions[:, 0]
    y = positions[:, 1]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=dpi, format="jpg")
    plt.close()



subprocess.run(["bash","./cleanup.sh"]) # Clean up the folder. 

images = []
#images.append(["blob-1.jpg",blob-2.jpg","blob-3.jpg"])
#images.extend(["blob-4.jpg","blob-5.jpg","blob-6.jpg","blob-7.jpg","blob-8.jpg","blob-9.jpg","blob-10.jpg","blob-11.jpg","blob-12.jpg"])

# images.extend(["blob-5.jpg","blob-6.jpg","blob-11.jpg","blob-12.jpg"])
images.extend(["blob-5.jpg"])

# cx_list=[0.4,0.6,0.25,0.6]
# cy_list=[0.35,0.7,0.35,0.4]
# x_pct_list=[0.4,0.4,0.4,0.4]
# y_pct_list=[0.4,0.4,0.4,0.4]
# line_width_list=[3,3,3,3]

cx_list=[0.4,0.6,0.25,0.6]
cy_list=[0.35,0.7,0.35,0.4]
x_pct_list=[0.6,0.6,0.6,0.6]
y_pct_list=[0.6,0.6,0.6,0.6]
line_width_list=[3,3,3,3]


i=0
for n in images: 
    i+=1

    img = cv2.imread(n)

    color = False
    if color == True: # Do some Visualization of the color channels
        out1 = img.copy()
        # out1[:,:,0]=0
        out1[:,:,1]=0
        out1[:,:,2]=0
        cv2.imwrite(f"blue-{i}.jpg",out1)

        out2 = img.copy()
        out2[:,:,0]=0
        # out2[:,:,1]=1
        out2[:,:,2]=0
        cv2.imwrite(f"green-{i}.jpg", out2)

        out3 = img.copy()
        out3[:,:,0]=0
        out3[:,:,1]=0
        # out3[:,:,2]=1
        cv2.imwrite(f"red-{i}.jpg",out3)

        img_r = red_only(img.copy())
        cv2.imwrite(f"red_only_{i}.png",img_r)

    debug_name = f"debug-{i}-10.png"
    # print(f"debug name is: {debug_name}")
    # detect_crosshair_centerline_hough(img,debug_name)
    cx,cy,theta_deg,overlay = detect_crosshair_centerline_hough(img,debug_name)
    print(f"cx,cy: {cx},{cy}")

    # Test the crosshair drawing code.
    overlay = draw_crosshair_at_point(overlay, (cx+10,cy+10), length=10)
    cv2.imwrite(f"debug-{i}-12.png", overlay)

    # try: 
    #     skel = skeletonize(img)
    #     cv2.imwrite(f"debug-{i}-50.png", skel)
    # except: 
    #     print(f"Skeletonization failed on image: {i}")



    overlay, corners = visualize_roi_percent(
        img,
        center_x_pct=cx_list[i-1],
        center_y_pct=cy_list[i-1],
        size_x_pct=x_pct_list[i-1],
        size_y_pct=y_pct_list[i-1],
        line_width=line_width_list[i-1]
    )

    # img = cv2.imread("test_square.jpg")
    # debug_name = f"debug_{i}-1-square.png"
    # cy,cy,theta_deg,overlay = detect_crosshair_centerline_hough(img, debug_out=debug_name)

    cv2.imwrite(f"debug-{i}-15.png", overlay)
    # print("Corners:", corners)

    subimg,corners_clipped = crop_subimage(img,corners)
    cv2.imwrite(f"debug-{i}-17.png", subimg)

    try: 
        cv2.imwrite(debug_name, subimg)
        cx,cy,theta_deg,overlay = detect_crosshair_centerline_hough(subimg,debug_name)
        cv2.imwrite(f"debug-{i}-20.png", overlay)
    except: 
        print(f"crosshair detection faield on subimage {i}")

    # try: 
    #     skel = skeletonize(subimg)
    #     cv2.imwrite(f"debug-{i}-40.png", skel)
    # except: 
    #     print(f"Skeletonization failed on sub image: {i}")

    # Begin finding sub-areas
    img1 = img.copy()
    points = rectangle_edge_midpoints(corners)
    for p in points: 
        img1 = draw_crosshair_at_point(img1,p,length=10)
    # cv2.imwrite(f"debug-{i}-50.png", img1)

    # Height and width are assumed to be the height at the vertical (Right & Left) edges of the box.  
    # The are swapped on the top and bottom.
    h = 300
    w = 30

    img2 = img1.copy()
    mid_subregions = rectangle_mid_subregions(corners,width=w,height=h)

    # print("mid subregions are: ",mid_subregions)
    img3 = draw_box_at_corners(img2,mid_subregions,line_width = 3)
    cv2.imwrite(f"debug-{i}-60.png",img3)


    box = mid_subregions[0]
    cropped, ((x_min, y_min), (x_max, y_max)) = crop_to_box(img, box)

    profile = row_average_profile(cropped)
    grey_strip = profile_to_greyscale_and_plot(profile, title="Row average profile")
    cv2.imwrite(f"debug-{i}-70.png",grey_strip)
    
    plot_grey_strip_profile(grey_strip,title="Gray Strip Profile",save_path=f"debug-{i}-75.jpg",dpi=300)

    padded_grey = pad_grey_strip_symmetric(grey_strip, N=50)
    cv2.imwrite(f"debug-{i}-71.png",padded_grey)

    print(f"padded grey is: \n{padded_grey}")

    flipped_strip = flip_gray_strip(padded_grey)
    print(f"flipped_strip is: \n{flipped_strip}")

    score = strip_squared_difference_sum(padded_grey,flipped_strip)
    print(f"rss score is: {score}")

    rss_vals = difference_rss(grey_strip,50)
    plot_xy_positions(rss_vals,save_path = f"debug-{i}-76.jpg")
