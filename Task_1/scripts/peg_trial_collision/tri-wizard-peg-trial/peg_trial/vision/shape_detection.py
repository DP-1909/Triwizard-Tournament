"""Step 2 of the vision stack: decide if an outline is a square, triangle,
circle or hexagon, and find its rotation (yaw).

Method (classic OpenCV, no training needed). We measure 3 numbers:
  f_tri  = area / area of the smallest triangle around it
  f_rect = area / area of the smallest rotated rectangle around it
  rr     = (10th percentile of centre-to-edge distance) / (90th percentile)
           -> 1.0 for a circle, smaller the "pointier" the corners

                 f_tri   f_rect   rr
    triangle     1.00    0.50    0.55
    square       0.50    1.00    0.75
    hexagon      0.67    0.75    0.89
    circle       0.60    0.79    1.00

and pick the closest row (rr counts double: it separates circle/hexagon).
This is much more robust than counting corners with approxPolyDP, because
real pegs have slightly rounded corners and noisy edges.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SHAPES = ("square", "triangle", "circle", "hexagon")
# rotational symmetry: a square looks the same every 90 deg, etc.
SYMMETRY = {"square": 4, "triangle": 3, "hexagon": 6, "circle": 0}
IDEAL = {
    "triangle": (1.00, 0.50, 0.548),
    "square": (0.50, 1.00, 0.747),
    "hexagon": (0.667, 0.75, 0.889),
    "circle": (0.605, 0.785, 1.00),
}
WEIGHTS = (1.0, 1.0, 2.0)


def circularity(contour) -> float:
    a = cv2.contourArea(contour)
    p = cv2.arcLength(contour, True)
    return 0.0 if p == 0 else 4 * math.pi * a / (p * p)


def approx_vertices(contour, eps_frac: float = 0.03):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, eps_frac * peri, True).reshape(-1, 2)


def _resample(closed_pts, step=1.0):
    """Points evenly spaced along a closed polygon (so every edge counts equally)."""
    p = np.vstack([closed_pts, closed_pts[:1]]).astype(float)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * t for t in np.arange(n) / n]
    return np.array(out)


def fill_features(contour):
    """(f_tri, f_rect, rr) — see the module docstring."""
    hull = cv2.convexHull(contour.astype(np.float32))
    a = cv2.contourArea(hull)
    if a <= 0:
        return None, hull
    tri_area, _ = cv2.minEnclosingTriangle(hull)
    (w, h) = cv2.minAreaRect(hull)[1]
    m = cv2.moments(hull)
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    pts = _resample(hull.reshape(-1, 2), step=max(0.5, math.sqrt(a) / 100))
    r = np.linalg.norm(pts - c, axis=1)
    rr = float(np.percentile(r, 10) / max(np.percentile(r, 90), 1e-9))
    return (a / max(tri_area, 1e-9), a / max(w * h, 1e-9), rr), hull


def corner_points(shape, hull, eps_frac=0.03):
    """Corner points used to measure yaw."""
    if shape == "triangle":
        return cv2.minEnclosingTriangle(hull)[1].reshape(-1, 2)
    if shape == "square":
        return cv2.boxPoints(cv2.minAreaRect(hull))
    if shape == "hexagon":
        for e in (eps_frac, 0.02, 0.04, 0.05, 0.015):
            v = approx_vertices(hull, e)
            if len(v) == 6:
                return v
    return hull.reshape(-1, 2)


def classify_shape(contour, eps_frac: float = 0.03):
    """Return (shape_name, corner points Nx2, confidence 0..1)."""
    feats, hull = fill_features(contour)
    if feats is None:
        return "circle", hull.reshape(-1, 2), 0.0
    d = {s: math.sqrt(sum(w * (f - i) ** 2 for w, f, i in zip(WEIGHTS, feats, IDEAL[s]))) for s in IDEAL}
    ranked = sorted(d, key=d.get)
    shape = ranked[0]
    # confidence: how much closer the winner is than the runner-up
    conf = max(0.0, min(1.0, (d[ranked[1]] - d[shape]) / (d[ranked[1]] + 1e-9) * 1.5))
    return shape, np.asarray(corner_points(shape, hull, eps_frac), float), conf


def polygon_yaw(points_xy: np.ndarray, center_xy, n: int) -> float:
    """Yaw of a regular n-gon from its corner points (radians, in [0, 2pi/n)).

    Trick: multiply every corner angle by n so all corners land on the same
    angle, average them on the unit circle, then divide by n again.
    Works in any 2-D frame (image pixels or robot x-y metres).
    """
    if n == 0:
        return 0.0
    d = np.asarray(points_xy, float) - np.asarray(center_xy, float)
    ang = np.arctan2(d[:, 1], d[:, 0]) * n
    yaw = math.atan2(np.sin(ang).sum(), np.cos(ang).sum()) / n
    return yaw % (2 * math.pi / n)
