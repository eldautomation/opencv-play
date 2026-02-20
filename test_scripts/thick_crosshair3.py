import cv2
import numpy as np
import math

# ----------------------------
# 1) "Redness" extraction
# ----------------------------
def redness_image(bgr: np.ndarray) -> np.ndarray:
    """
    Returns a single-channel float32 image emphasizing red features:
        redness = R - 0.5*(G + B)
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    red = r - 0.5 * (g + b)
    return red

# ----------------------------
# 2) Threshold + cleanup
# ----------------------------
def segment_crosshair_from_redness(bgr: np.ndarray, roi_frac=0.8):
    """
    Produces a binary mask (0/255) of likely crosshair pixels.
    Threshold is chosen from a centered ROI to reduce background influence.
    """
    H, W = bgr.shape[:2]
    red = redness_image(bgr)

    # Normalize to 8-bit for thresholding
    red_norm = cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Center ROI for threshold selection
    rw = int(W * roi_frac)
    rh = int(H * roi_frac)
    x0 = (W - rw) // 2
    y0 = (H - rh) // 2
    roi = red_norm[y0:y0+rh, x0:x0+rw]

    # Otsu threshold on ROI, then apply globally
    _, t = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (red_norm >= t).astype(np.uint8) * 255

    # Morphological cleanup: close to fill bright cores, open to remove specks
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k_open,  iterations=1)

    return bw, red_norm

# ----------------------------
# 3) Skeletonization (thinning)
#    (pure OpenCV morphology loop)
# ----------------------------
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

# ----------------------------
# 4) Fit two lines (RANSAC-ish) using Hough on skeleton
#    Then centerline correction by rho averaging inside clusters
# ----------------------------
def _normalize_rho_theta(rho, theta):
    # enforce rho >= 0 and theta in [0, pi)
    if rho < 0:
        rho = -rho
        theta = (theta + np.pi) % (2 * np.pi)
    theta = theta % np.pi
    return float(rho), float(theta)

def _intersect_rho_theta(rho1, th1, rho2, th2):
    A = np.array([[math.cos(th1), math.sin(th1)],
                  [math.cos(th2), math.sin(th2)]], dtype=np.float64)
    b = np.array([rho1, rho2], dtype=np.float64)
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)

def fit_crosshair_lines_from_skeleton(skel_255: np.ndarray):
    """
    Uses HoughLines on the skeleton to find the two dominant line families,
    then estimates each bar centerline as the mean rho of that family.

    Returns:
      (rho1, th1), (rho2, th2)  or None
    """
    edges = skel_255  # already 1px structures; treat as edge map

    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    if lines is None or len(lines) < 2:
        return None

    cand = []
    for i in range(min(80, len(lines))):
        rho, th = lines[i][0]
        rho, th = _normalize_rho_theta(rho, th)
        cand.append((rho, th))

    # Cluster by angle into 2 groups using 2*theta embedding (period pi)
    ang = np.array([[math.cos(2*t), math.sin(2*t)] for _, t in cand], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, _ = cv2.kmeans(ang, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    groups = [[], []]
    for (rho, th), lab in zip(cand, labels.ravel()):
        groups[int(lab)].append((rho, th))

    center_lines = []
    for g in groups:
        if len(g) < 1:
            continue
        thetas = np.array([t for _, t in g], dtype=np.float64)
        th_med = float(np.median(thetas))

        # keep only near that theta (reduce contamination)
        keep = []
        for rho, th in g:
            d = abs((th - th_med) % np.pi)
            d = min(d, np.pi - d)
            if d < (8 * np.pi/180):
                keep.append((rho, th))
        if len(keep) >= 3:
            g = keep

        rhos = np.array([rho for rho, _ in g], dtype=np.float64)

        # For skeleton, rhos should cluster tightly around the centerline.
        # Use median rho (robust).
        rho_c = float(np.median(rhos))
        center_lines.append((rho_c, th_med))

    if len(center_lines) != 2:
        return None

    return center_lines[0], center_lines[1]

def crosshair_angle_from_theta(th1, th2):
    # Convert normal angle to line direction angle: dir = theta - 90°
    a1 = math.degrees(th1) - 90.0
    a2 = math.degrees(th2) - 90.0
    def norm(a): return (a + 90) % 180 - 90
    a1, a2 = norm(a1), norm(a2)
    # report the more "horizontal" arm consistently
    return a1 if abs(a1) <= abs(a2) else a2

# ----------------------------
# 5) Full pipeline
# ----------------------------
def detect_crosshair_pipeline(
    jpg_path: str,
    roi_frac_for_thresh: float = 0.8,
    debug_prefix: str = "dbg"
):
    """
    End-to-end:
      JPG -> redness -> threshold -> cleanup -> skeleton -> Hough -> intersection

    Returns dict with x,y,theta_deg and writes debug images.
    """
    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(jpg_path)

    bw, red_norm = segment_crosshair_from_redness(img, roi_frac=roi_frac_for_thresh)
    skel = skeletonize(bw)

    fitted = fit_crosshair_lines_from_skeleton(skel)
    if fitted is None:
        return None

    (rho1, th1), (rho2, th2) = fitted
    p = _intersect_rho_theta(rho1, th1, rho2, th2)
    if p is None:
        return None
    cx, cy = p
    theta_deg = crosshair_angle_from_theta(th1, th2)

    # Debug overlay
    overlay = img.copy()
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    H, W = img.shape[:2]
    def draw_line(rho, th, color):
        ct, st = math.cos(th), math.sin(th)
        x0, y0 = ct * rho, st * rho
        dx, dy = -st, ct
        L = 2 * max(H, W)
        p1 = (int(round(x0 + dx*L)), int(round(y0 + dy*L)))
        p2 = (int(round(x0 - dx*L)), int(round(y0 - dy*L)))
        cv2.line(overlay, p1, p2, color, 1)

    draw_line(rho1, th1, (0, 255, 0))
    draw_line(rho2, th2, (0, 255, 0))

    cv2.imwrite(f"{debug_prefix}_rednorm.png", red_norm)
    cv2.imwrite(f"{debug_prefix}_mask.png", bw)
    cv2.imwrite(f"{debug_prefix}_skel.png", skel)
    cv2.imwrite(f"{debug_prefix}_overlay.png", overlay)

    return {
        "x": cx,
        "y": cy,
        "theta_deg": theta_deg,
        "debug": {
            "rednorm": f"{debug_prefix}_rednorm.png",
            "mask": f"{debug_prefix}_mask.png",
            "skeleton": f"{debug_prefix}_skel.png",
            "overlay": f"{debug_prefix}_overlay.png",
        }
    }

# Example:
res = detect_crosshair_pipeline("blob-12.jpg", debug_prefix="blob12-debug")
print(res)
 
