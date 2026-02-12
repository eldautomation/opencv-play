import cv2
import numpy as np
import math

def _auto_binary(gray):
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Make foreground be white (crosshair)
    # Choose polarity that yields more white pixels in the central region
    h, w = gray.shape
    y0, y1 = int(0.25*h), int(0.75*h)
    x0, x1 = int(0.25*w), int(0.75*w)
    if np.sum(bw[y0:y1, x0:x1] == 255) < np.sum((255-bw)[y0:y1, x0:x1] == 255):
        bw = 255 - bw
    return bw

def _line_intersection_rho_theta(rho1, th1, rho2, th2):
    # Solve:
    # x cos(th) + y sin(th) = rho
    A = np.array([[math.cos(th1), math.sin(th1)],
                  [math.cos(th2), math.sin(th2)]], dtype=np.float64)
    b = np.array([rho1, rho2], dtype=np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)

def detect_crosshair_intersection_hough(img_bgr_or_gray, debug_out=None):
    """
    Returns (cx, cy, theta_deg, overlay_bgr)

    cx, cy: intersection in pixel coordinates
    theta_deg: angle of one arm (the more "horizontal" arm) in degrees in [-90, 90)
    """
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        base = img_bgr_or_gray.copy()
    else:
        gray = img_bgr_or_gray
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    bw = _auto_binary(gray)

    # Clean small noise; keep bars
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    # Edges for Hough
    edges = cv2.Canny(bw, 50, 150)

    # Hough lines (infinite lines)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=140)
    if lines is None or len(lines) < 2:
        return None

    # Consider top N strongest lines
    N = min(30, len(lines))
    cand = [(float(lines[i][0][0]), float(lines[i][0][1])) for i in range(N)]  # (rho, theta)

    # Find best near-perpendicular pair whose intersection is plausible (near image)
    best = None
    best_score = None

    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            rho1, th1 = cand[i]
            rho2, th2 = cand[j]

            # angle difference modulo pi
            d = abs(th1 - th2) % np.pi
            d = min(d, np.pi - d)
            perp_err = abs(d - np.pi/2)

            if perp_err > (10 * np.pi/180):  # within 10 degrees of perpendicular
                continue

            p = _line_intersection_rho_theta(rho1, th1, rho2, th2)
            if p is None:
                continue
            x, y = p

            # Prefer intersections inside/near the image and near the center
            cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
            dist2 = (x - cx0)**2 + (y - cy0)**2

            # Penalize being far outside the image
            outside_pen = 0.0
            if x < -0.1*w or x > 1.1*w or y < -0.1*h or y > 1.1*h:
                outside_pen = 1e12

            score = dist2 + outside_pen + (perp_err * 1e6)

            if best_score is None or score < best_score:
                best_score = score
                best = (rho1, th1, rho2, th2, x, y)

    if best is None:
        return None

    rho1, th1, rho2, th2, cx, cy = best

    # Convert Hough theta (normal angle) to line direction angle.
    # Line direction = theta - 90°.
    a1 = (math.degrees(th1) - 90.0)
    a2 = (math.degrees(th2) - 90.0)

    # Normalize to [-90, 90)
    def norm(a):
        return (a + 90) % 180 - 90

    a1, a2 = norm(a1), norm(a2)

    # Choose the "more horizontal" one for reporting theta (consistent)
    theta_deg = a1 if abs(a1) <= abs(a2) else a2

    overlay = base.copy()
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # Optional: draw the two infinite lines for debugging
    def draw_infinite_line(rho, th, img, color):
        ct, st = math.cos(th), math.sin(th)
        x0, y0 = ct * rho, st * rho
        # direction vector along the line
        dx, dy = -st, ct
        # pick two far points
        L = max(h, w) * 2
        x1, y1 = int(round(x0 + dx * L)), int(round(y0 + dy * L))
        x2, y2 = int(round(x0 - dx * L)), int(round(y0 - dy * L))
        cv2.line(img, (x1, y1), (x2, y2), color, 1)

    draw_infinite_line(rho1, th1, overlay, (0, 255, 0))
    draw_infinite_line(rho2, th2, overlay, (0, 255, 0))

    if debug_out is not None:
        cv2.imwrite(debug_out, overlay)

    return (cx, cy, theta_deg, overlay)


img = cv2.imread("test.jpg")
res = detect_crosshair_intersection_hough(img, debug_out="hough_debug.png")
print(res[0], res[1], res[2])  # cx, cy, theta_deg

img = cv2.imread("test_square.jpg")
res = detect_crosshair_intersection_hough(img, debug_out="hough_debug_square.png")
print(res[0], res[1], res[2])  # cx, cy, theta_deg

