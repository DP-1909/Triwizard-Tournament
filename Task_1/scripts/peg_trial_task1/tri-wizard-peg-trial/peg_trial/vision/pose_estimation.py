"""Step 3: turn pixels + depth into 3-D points in the ROBOT BASE frame.

Pinhole camera model (intrinsics fx, fy, cx, cy from /camera_info):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth at pixel (u, v)          [metres]
That point is in the CAMERA frame. The 4x4 matrix T_base_cam from
config/camera_calib.yaml then moves it into the robot base frame:
    p_base = T_base_cam @ [X, Y, Z, 1]
"""
from __future__ import annotations

import cv2
import numpy as np


def depth_in_mask(depth_m: np.ndarray, mask: np.ndarray, erode_px: int = 5):
    """Robust depth (median) of the pixels inside the mask, ignoring zeros/NaN.

    Eroding first keeps us away from the blob edge where depth is noisy.
    """
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
    vals = depth_m[mask > 0]
    vals = vals[np.isfinite(vals) & (vals > 0.05)]
    if vals.size < 10:
        return None
    return float(np.median(vals))


def deproject(u, v, z, K):
    """Pixel(s) + depth -> 3-D point(s) in the camera optical frame."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, np.full_like(u, z)], axis=-1)


def to_base(points_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(points_cam)
    ph = np.hstack([p, np.ones((len(p), 1))])
    return (np.asarray(T_base_cam) @ ph.T).T[:, :3]


def depth_to_m(depth_img: np.ndarray, scale: float = 0.001) -> np.ndarray:
    """RealSense publishes 16UC1 in millimetres; Gazebo publishes 32FC1 in metres."""
    if depth_img.dtype == np.uint16:
        return depth_img.astype(np.float32) * scale
    return depth_img.astype(np.float32)
