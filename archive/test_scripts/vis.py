import cv2
import numpy as np

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


import numpy as np

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


img = cv2.imread("blob-11.jpg")

overlay, corners = visualize_roi_percent(
    img,
    center_x_pct=0.25,
    center_y_pct=0.35,
    size_x_pct=0.4,
    size_y_pct=0.4,
    line_width=3
)

cv2.imwrite("roi_overlay.png", overlay)
print("Corners:", corners)

subimg,corners_clipped = crop_subimage(img,corners)
cv2.imwrite("roi_subimage.png", subimg)


