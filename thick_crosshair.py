import cv2
import numpy as np
import math

def pca_angle_and_center(binary):
    ys, xs = np.where(binary > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    if len(pts) < 50:
        return None

    # Center (centroid)
    cx, cy = pts.mean(axis=0)

    # PCA
    pts0 = pts - np.array([cx, cy], dtype=np.float32)
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # principal direction
    theta = math.degrees(math.atan2(v[1], v[0]))  # angle of principal axis

    return cx, cy, theta

def detect_crosshair_pca(path, debug_out="overlay.png"):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to Grayscale

    # cv2.imwrite("out_1.png",gray)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Apply Threshhold

    cv2.imwrite("out_1.png",gray)
    cv2.imwrite("out_2.png",bw)
    bw = ensure_white_foreground_center_roi(gray, bw, roi_w_frac=0.2, roi_h_frac=0.2)
    cv2.imwrite("out_3.png",bw)

    # cv2.imwrite("out_2.png",bw)
    gray = cv2.GaussianBlur(bw, (5,5), 0)
    # cv2.imwrite("out_3.png",gray)

    # Binary segmentation: choose one
    # _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 5)
    # cv2.imwrite("out_4.png",bw)

    # If crosshair is dark on bright background, invert:
    # bw = 255 - bw
    bw = ensure_white_foreground(gray, bw) # Convert to white cross

    # Clean up
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw2 = cv2.morphologyEx(bw2, cv2.MORPH_CLOSE, k, iterations=1)

    # Keep largest connected component (or closest to image center)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw2, connectivity=8)
    if num <= 1:
        return None

    # Ignore background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + np.argmax(areas)
    mask = (labels == best).astype(np.uint8) * 255

    res = pca_angle_and_center(mask)
    if res is None:
        return None
    cx, cy, theta = res

    # Debug overlay
    overlay = img.copy()
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0,0,255),
                   markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)

    # draw principal axis line
    L = 200
    rad = math.radians(theta)
    x2 = int(round(cx + L * math.cos(rad)))
    y2 = int(round(cy + L * math.sin(rad)))
    x1 = int(round(cx - L * math.cos(rad)))
    y1 = int(round(cy - L * math.sin(rad)))
    cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imwrite(debug_out, overlay)

    return {"x": cx, "y": cy, "theta_deg": theta, "debug_image": debug_out}
 
def ensure_white_foreground(gray, bw):
    """
    Ensures that the crosshair (foreground) is white (255) in bw.
    If the detected foreground is brighter than the background in the
    grayscale image, the binary image is inverted.

    Parameters
    ----------
    gray : uint8 grayscale image
    bw   : uint8 binary image (0 or 255)

    Returns
    -------
    bw_out : uint8 binary image with white foreground
    """

    # Make sure bw is 0/255
    bw = (bw > 0).astype(np.uint8) * 255

    # Foreground mask (assume largest non-zero region is crosshair)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    if num <= 1:
        # Nothing detected; return as-is
        return bw

    # Ignore label 0 (background), choose largest component
    areas = stats[1:, cv2.CC_STAT_AREA]
    fg_label = 1 + np.argmax(areas)
    fg_mask = (labels == fg_label)

    bg_mask = ~fg_mask

    # Mean grayscale values
    fg_mean = float(gray[fg_mask].mean())
    bg_mean = float(gray[bg_mask].mean())

    # If foreground is brighter than background, invert
    if fg_mean > bg_mean:
        bw = 255 - bw

    return bw

def crop_center_roi(img, roi_w_frac=0.5, roi_h_frac=0.5):
    """
    Crop a centered ROI from an image.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or color).
    roi_w_frac : float
        Fraction of image width to keep (0 < roi_w_frac <= 1).
    roi_h_frac : float
        Fraction of image height to keep (0 < roi_h_frac <= 1).

    Returns
    -------
    roi : np.ndarray
        Cropped ROI view of the image.
    bbox : tuple
        (x0, y0, x1, y1) ROI bounds in the original image coordinates.
    """
    h, w = img.shape[:2]
    print(f"image shape before crop: - height:{h}\twidth:{w}",)
    roi_w = max(1, int(round(w * roi_w_frac)))
    roi_h = max(1, int(round(h * roi_h_frac)))
    print(f"image shape after crop: - height:{roi_h}\twidth:{roi_w}",)

    x0 = (w - roi_w) // 2
    y0 = (h - roi_h) // 2
    x1 = x0 + roi_w
    y1 = y0 + roi_h

    roi = img[y0:y1, x0:x1]
    return roi, (x0, y0, x1, y1)


def choose_component_nearest_center(binary_roi):
    """
    Given a binary ROI (0/255), return a mask (0/255) of the connected component
    whose centroid is closest to the ROI center. Returns None if no components.
    """
    bw = (binary_roi > 0).astype(np.uint8) * 255
    h, w = bw.shape[:2]
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return None

    # labels: 0 is background, 1..n-1 are components
    best_label = None
    best_d2 = None

    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 20:
            continue  # ignore tiny specks

        cxi, cyi = centroids[label]
        d2 = (cxi - cx0) ** 2 + (cyi - cy0) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_label = label

    if best_label is None:
        return None

    mask = (labels == best_label).astype(np.uint8) * 255
    return mask


def ensure_white_foreground_center_roi(gray, bw, roi_w_frac=0.5, roi_h_frac=0.5):
    """
    ROI-based polarity check:
    - crops a centered ROI
    - finds the component nearest ROI center
    - compares mean gray of that component vs ROI background
    - inverts bw if needed so the crosshair becomes white (255)

    Returns
    -------
    bw_out : uint8 binary image (0/255) in full image size
    """
    bw = (bw > 0).astype(np.uint8) * 255

    gray_roi, (x0, y0, x1, y1) = crop_center_roi(gray, roi_w_frac, roi_h_frac)

    h,w = bw.shape[:2]
    print(f"xx.image shape before crop: - height:{h}\twidth:{w}",)

    bw_roi, _ = crop_center_roi(bw, roi_w_frac, roi_h_frac)
    h,w = bw_roi.shape[:2]
    print(f"xx.image shape after crop: - height:{h}\twidth:{w}",)

    fg_mask_roi = choose_component_nearest_center(bw_roi)
    if fg_mask_roi is None:
        print("Nothing useable detected")
        return bw  # nothing usable detected

    fg = fg_mask_roi > 0
    bg = ~fg

    # If ROI background mask is empty (unlikely), skip inversion logic
    if bg.sum() == 0 or fg.sum() == 0:
        print("Nothing useable detected")
        return bw

    fg_mean = float(gray_roi[fg].mean())
    bg_mean = float(gray_roi[bg].mean())

    # If foreground is brighter than background, invert full binary image
    if fg_mean > bg_mean:
        bw = 255 - bw

    h,w = bw.shape[:2]
    print(f"xxx.image shape after crop: - height:{h}\twidth:{w}",)

    return bw


detect_crosshair_pca("test_square.jpg", debug_out="overlay.png")

