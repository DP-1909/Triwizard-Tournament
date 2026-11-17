#!/usr/bin/env python3
"""Save one colour + depth frame and the intrinsics of every camera to data/.

    python3 scripts/save_frames.py            (cameras must be running)

Output per camera:  data/<cam>_color.png, data/<cam>_depth.npy (metres), data/<cam>_K.npy
Use them for:  tune_hsv.py,  calibrate_extrinsics.py --offline,  offline_vision_test.py
"""
import os
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peg_trial.utils.config import load  # noqa: E402
from peg_trial.vision.pose_estimation import depth_to_m  # noqa: E402


def main():
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=None, help="Specific camera to save, e.g. cam1 (default: all)")
    args = parser.parse_args()

    calib = load("camera_calib.yaml")
    cams = [args.camera] if args.camera else list(calib["cameras"].keys())
    cams = [c for c in cams if c in calib["cameras"]]

    rclpy.init()
    node = rclpy.create_node("peg_save_frames")
    br = CvBridge()
    got = {}
    for cam in cams:
        c = calib["cameras"][cam]
        node.create_subscription(Image, c["color_topic"], lambda m, cam=cam: got.__setitem__((cam, "c"), m), 1)
        node.create_subscription(Image, c["depth_topic"], lambda m, cam=cam: got.__setitem__((cam, "d"), m), 1)
        node.create_subscription(CameraInfo, c["info_topic"], lambda m, cam=cam: got.__setitem__((cam, "k"), m), 1)
    need = {(cam, t) for cam in cams for t in "cdk"}
    t0 = time.time()
    while not need <= set(got) and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    os.makedirs("data", exist_ok=True)
    for cam in cams:
        if not {(cam, t) for t in "cdk"} <= set(got):
            print(f"{cam}: missing topics, skipped")
            continue
        cv2.imwrite(f"data/{cam}_color.png", br.imgmsg_to_cv2(got[(cam, "c")], "bgr8"))
        np.save(f"data/{cam}_depth.npy", depth_to_m(br.imgmsg_to_cv2(got[(cam, "d")], "passthrough"),
                                                   calib.get("depth_scale", 0.001)))
        np.save(f"data/{cam}_K.npy", np.array(got[(cam, "k")].k).reshape(3, 3))
        print(f"{cam}: saved successfully to data/")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
