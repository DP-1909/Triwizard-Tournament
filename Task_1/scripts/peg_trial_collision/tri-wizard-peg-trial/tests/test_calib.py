"""ArUco extrinsic calibration recovers a known camera pose (synthetic image)."""
import cv2
import numpy as np

from peg_trial.utils.calib import T_base_marker, marker_image, solve_T_base_cam
from peg_trial.utils.synthetic import default_K


def lookat(eye, tgt):
    z = np.array(tgt, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, eye
    return T


def test_calibration_recovers_pose():
    K = default_K()
    T_true = lookat(np.array([0.05, 0.25, 0.5]), (0.2, 0.0, -0.05))
    L, mx, my, mz = 0.08, 0.2, 0.0, -0.05
    h = L / 2
    corners_m = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]])   # TL TR BR BL
    pts_base = (T_base_marker(mx, my, mz) @ corners_m.T).T
    pc = (np.linalg.inv(T_true) @ pts_base.T).T[:, :3]
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    px = 400
    mk = marker_image(0, px)
    src = np.float32([[0, 0], [px, 0], [px, px], [0, px]])
    H = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    img = np.full((480, 640), 255, np.uint8)
    warped = cv2.warpPerspective(mk, H, (640, 480), borderValue=255)
    img = np.minimum(img, warped)
    img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    T_est, err = solve_T_base_cam(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), K, L, mx, my, mz)
    assert err < 1.0
    assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=0.01)
    assert np.allclose(T_est[:3, :3], T_true[:3, :3], atol=0.03)
