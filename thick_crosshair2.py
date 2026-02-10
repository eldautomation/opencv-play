import cv2
import numpy as np
import math

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

    return cx, cy, theta_deg, overlay


img = cv2.imread("test.jpg")
cy,cy,theta_deg,overlay = detect_crosshair_centerline_hough(img, debug_out="debug_1.png")

img = cv2.imread("test_square.jpg")
cy,cy,theta_deg,overlay = detect_crosshair_centerline_hough(img, debug_out="debug_2.png")

