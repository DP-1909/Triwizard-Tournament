"""Camera -> robot base calibration with ONE printed ArUco marker.

How it works:
  * You lay a marker FLAT on the table at a spot you measured in the robot
    base frame (centre at x=mx, y=my), with the marker's TOP edge pointing
    AWAY from the robot (+x direction).
  * OpenCV finds the 4 marker corners in the image and solvePnP gives the
    marker pose in the camera frame:          T_cam_marker
  * We know the marker pose in the base frame: T_base_marker
  * So the camera pose in the base frame is:  T_base_cam = T_base_marker @ inv(T_cam_marker)
"""
from __future__ import annotations

import cv2
import numpy as np


def _dictionary(name="DICT_4X4_50"):
    d = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(d)


def detect_markers(gray, dict_name="DICT_4X4_50"):
    d = _dictionary(dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):                       # OpenCV >= 4.7
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:                                                          # OpenCV 4.5 (Ubuntu 22.04 apt)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=cv2.aruco.DetectorParameters_create())
    return corners, ids


def marker_image(marker_id=0, px=400, dict_name="DICT_4X4_50"):
    d = _dictionary(dict_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(d, marker_id, px)
    return cv2.aruco.drawMarker(d, marker_id, px)


def T_base_marker(mx, my, mz):
    """Marker flat on the table, top edge pointing +x (away from robot)."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)   # columns: x_m=-y, y_m=+x, z_m=+z
    T[:3, 3] = [mx, my, mz]
    return T


def solve_T_base_cam(bgr, K, marker_len, mx, my, mz, marker_id=0, dist=None, dict_name="DICT_4X4_50"):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    corners, ids = detect_markers(gray, dict_name)
    if ids is None or marker_id not in ids.flatten():
        raise RuntimeError(f"marker id {marker_id} not found (found: {None if ids is None else ids.flatten().tolist()})")
    c = corners[list(ids.flatten()).index(marker_id)].reshape(4, 2).astype(np.float64)
    h = marker_len / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)   # TL, TR, BR, BL
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(obj, c, np.asarray(K, float), dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(K, float), dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - c, axis=1)))
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_marker[:3, 3] = tvec.ravel()
    return T_base_marker(mx, my, mz) @ np.linalg.inv(T_cam_marker), err
