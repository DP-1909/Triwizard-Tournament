#!/usr/bin/env python3
"""Compute T_base_cam for one camera from a saved frame with an ArUco marker.

1. Print marker id 0 (DICT_4X4_50):  python3 scripts/calibrate_extrinsics.py --make-marker
   -> marker_0.png. Print at 100 % scale, MEASURE the black square side (e.g. 0.08 m).
2. Lay it flat on the table, top edge pointing AWAY from the robot, centre at a
   measured point, e.g. x=0.20 m in front of the base centre, y=0.00.
3. Save frames:  python3 scripts/save_frames.py
4. Run:
   python3 scripts/calibrate_extrinsics.py --camera cam1 --size 0.08 --x 0.20 --y 0.0 --z -0.05
   (--z = table height in base frame = task.yaml table_z) -> writes config/camera_calib.yaml
5. Repeat for cam2 (marker may stay at the same spot if both cameras see it).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from peg_trial.utils.calib import marker_image, solve_T_base_cam  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-marker", action="store_true")
    ap.add_argument("--camera", default="cam1")
    ap.add_argument("--size", type=float, default=0.045, help="marker black-square side in metres (default: 0.045m / 4.5cm)")
    ap.add_argument("--x", type=float, default=0.093, help="marker centre x in base frame (default: 0.093m)")
    ap.add_argument("--y", type=float, default=0.128, help="marker centre y in base frame (default: 0.128m)")
    ap.add_argument("--z", type=float, default=0.0, help="table height (default: 0.0m)")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name")
    ap.add_argument("--data", default="data")
    ap.add_argument("--calib", default=os.path.join(ROOT, "config", "camera_calib.yaml"))
    a = ap.parse_args()

    if a.make_marker:
        img = cv2.copyMakeBorder(marker_image(a.id, 600, a.dict), 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(f"marker_{a.id}.png", img)
        print(f"wrote marker_{a.id}.png")
        return

    bgr = cv2.imread(os.path.join(a.data, f"{a.camera}_color.png"))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {os.path.join(a.data, f'{a.camera}_color.png')}. Run save_frames.py first!")
    K = np.load(os.path.join(a.data, f"{a.camera}_K.npy"))
    T, err = solve_T_base_cam(bgr, K, a.size, a.x, a.y, a.z, a.id, dict_name=a.dict)
    print(f"reprojection error: {err:.2f} px  (good: < 1 px)")
    print(f"camera position in base frame: {np.round(T[:3, 3], 3).tolist()} m")
    print(np.round(T, 4))

    with open(a.calib) as f:
        cfg = yaml.safe_load(f)
    cfg["cameras"][a.camera]["T_base_cam"] = np.round(T, 5).tolist()
    with open(a.calib, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    print(f"saved to {a.calib}")


if __name__ == "__main__":
    main()
