#!/usr/bin/env python3
"""Live RealSense Viewer with Real-Time Detection Boundaries & Reachability Overlays.

Features:
  - Real-time contour boundary of each peg (Orange, Green, Blue, Yellow).
  - Real-time contour boundary of each socket cutout.
  - Live distance (r in cm) from robot base for each socket.
  - Live reachability indicator: CYAN = OK, BRIGHT RED = TOO FAR (>24cm).
  - Press 'q' or ESC in window to exit.

Usage:
    python3 scripts/view_camera.py
    python3 scripts/view_camera.py --raw   # show plain video without overlay
"""
import argparse
import math
import os
import sys

# Force X11 backend to bypass Wayland / GNOME schema antialiasing bug
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from peg_trial.utils.config import load  # noqa: E402
from peg_trial.vision.pose_estimation import depth_to_m  # noqa: E402
from peg_trial.vision.scene import draw, process_frame  # noqa: E402


class LiveDetectorViewer(Node):
    def __init__(self, camera_name="cam1", raw_only=False):
        super().__init__("live_detector_viewer")
        self.br = CvBridge()
        self.raw_only = raw_only
        self.camera_name = camera_name

        self.colors = load("colors.yaml")
        self.calib = load("camera_calib.yaml")
        self.task = load("task.yaml")

        cam_cfg = self.calib["cameras"].get(camera_name, {})
        color_topic = cam_cfg.get("color_topic", "/cam1/camera/color/image_raw")
        depth_topic = cam_cfg.get("depth_topic", "/cam1/camera/aligned_depth_to_color/image_raw")
        info_topic = cam_cfg.get("info_topic", "/cam1/camera/color/camera_info")

        self.T = np.array(cam_cfg.get("T_base_cam", np.eye(4)), float)
        self.depth_scale = self.calib.get("depth_scale", 0.001)

        self.last_depth = None
        self.K = None

        self.create_subscription(CameraInfo, info_topic, self.on_info, 1)
        self.create_subscription(Image, depth_topic, self.on_depth, 1)
        self.create_subscription(Image, color_topic, self.on_color, 1)

        self.get_logger().info(f"Live Detection Viewer listening to {color_topic}")
        self.get_logger().info("Press 'q' or ESC in window to exit.")

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def on_depth(self, msg: Image):
        try:
            raw = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.last_depth = depth_to_m(raw, self.depth_scale)
        except Exception:
            pass

    def on_color(self, msg: Image):
        try:
            bgr = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            display_img = bgr

            if not self.raw_only and self.K is not None and self.last_depth is not None:
                try:
                    result = process_frame(bgr, self.last_depth, self.K, self.T, self.colors, self.task, self.camera_name)
                    display_img = draw(bgr, result, self.K, self.T)
                except Exception as e:
                    cv2.putText(display_img, f"Vision: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("RealSense Live Detection Feed", display_img)
            key = cv2.waitKey(1)
            if key in (27, ord("q")):
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Render error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--raw", action="store_true", help="Show raw stream without detection overlay")
    args = parser.parse_args()

    rclpy.init()
    node = LiveDetectorViewer(camera_name=args.camera, raw_only=args.raw)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
