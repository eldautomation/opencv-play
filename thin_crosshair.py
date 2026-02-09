import cv2
import numpy as np
import math

def line_angle_deg(x1, y1, x2, y2):
    # angle of direction vector in degrees, range (-180, 180]
    return math.degrees(math.atan2((y2 - y1), (x2 - x1)))

def intersect_lines(l1, l2):
    # l = (x1, y1, x2, y2) infinite lines intersection
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    # Solve using determinant form
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # parallel or nearly parallel

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    return (px, py)

def detect_crosshair(jpg_path, debug_out="debug_overlay.png"):
    img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {jpg_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast normalization (helps with uneven illumination)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Canny thresholds may need adjustment depending on your images
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # Hough line segments
    # Tune: minLineLength and maxLineGap depend on your crosshair size
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=80, minLineLength=int(min(h, w)*0.2), maxLineGap=10)

    if lines is None or len(lines) < 2:
        return None

    # Flatten
    segs = [tuple(l[0]) for l in lines]

    # Score segments by length
    def seg_len(s):
        x1, y1, x2, y2 = s
        return math.hypot(x2 - x1, y2 - y1)

    segs = sorted(segs, key=seg_len, reverse=True)

    # Pick best perpendicular pair among top K segments
    K = min(30, len(segs))
    best = None
    best_score = -1

    for i in range(K):
        for j in range(i+1, K):
            a1 = line_angle_deg(*segs[i])
            a2 = line_angle_deg(*segs[j])
            # angle difference modulo 180
            d = abs(a1 - a2) % 180
            d = min(d, 180 - d)
            # want close to 90
            perp_err = abs(d - 90)

            if perp_err < 10:  # 10° tolerance for MVP; tighten later
                score = seg_len(segs[i]) + seg_len(segs[j]) - 5 * perp_err
                if score > best_score:
                    best_score = score
                    best = (segs[i], segs[j], a1, a2, d)

    if best is None:
        return None

    l1, l2, a1, a2, d = best
    p = intersect_lines(l1, l2)
    if p is None:
        return None

    cx, cy = p

    # Choose a consistent theta:
    # Here: take the line closer to horizontal as "theta"
    # Normalize angle to [-90, 90)
    def norm_angle(a):
        a = (a + 90) % 180 - 90
        return a

    a1n, a2n = norm_angle(a1), norm_angle(a2)
    theta = a1n if abs(a1n) <= abs(a2n) else a2n

    # Debug overlay
    overlay = img.copy()
    cv2.line(overlay, (l1[0], l1[1]), (l1[2], l1[3]), (0, 255, 0), 2)
    cv2.line(overlay, (l2[0], l2[1]), (l2[2], l2[3]), (0, 255, 0), 2)
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.imwrite(debug_out, overlay)

    return {
        "x": cx,
        "y": cy,
        "theta_deg": theta,
        "debug_image": debug_out
    }

result = detect_crosshair("test.jpg", debug_out="overlay.png")
print(result)
print(result["debug_image"])


# Example usage:
# result = detect_crosshair("test.jpg", debug_out="overlay.png")
# print(result)
