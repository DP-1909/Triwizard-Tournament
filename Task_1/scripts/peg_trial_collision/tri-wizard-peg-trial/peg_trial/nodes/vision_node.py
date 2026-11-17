"""ROS 2 node: reads both RGB-D cameras, runs the vision pipeline, publishes the scene.

Publishes
  /peg_trial/scene            std_msgs/String  (JSON: {"stamp", "pegs": [...], "sockets": [...]})
  /peg_trial/debug/<cam>      sensor_msgs/Image (overlay, view in rqt_image_view)

JSON is used (instead of a custom .msg) so the package stays pure Python.
"""
from __future__ import annotations

import json

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from peg_trial.utils.config import load_all
from peg_trial.vision.pose_estimation import depth_to_m
from peg_trial.vision.scene import draw, fuse, process_frame


class VisionNode(Node):
    def __init__(self):
        super().__init__("peg_vision")
        self.declare_parameter("colors_file", "")
        self.declare_parameter("calib_file", "")
        self.declare_parameter("task_file", "")
        g = lambda n: self.get_parameter(n).value or None
        self.colors, self.calib, self.task = load_all(g("colors_file"), g("calib_file"), g("task_file"))

        self.bridge = CvBridge()
        self.frames = {}      # cam -> (bgr, depth_m)
        self.K = {}
        self.debug_pub = {}
        self.subs = []
        for cam, c in self.calib["cameras"].items():
            self.create_subscription(CameraInfo, c["info_topic"], lambda m, cam=cam: self._on_info(cam, m),
                                     qos_profile_sensor_data)
            # sensor-data QoS (best effort) works with both reliable and best-effort cameras
            col = message_filters.Subscriber(self, Image, c["color_topic"], qos_profile=qos_profile_sensor_data)
            dep = message_filters.Subscriber(self, Image, c["depth_topic"], qos_profile=qos_profile_sensor_data)
            sync = message_filters.ApproximateTimeSynchronizer([col, dep], 5, 0.1)
            sync.registerCallback(lambda a, b, cam=cam: self._on_rgbd(cam, a, b))
            self.subs += [col, dep, sync]
            self.debug_pub[cam] = self.create_publisher(Image, f"/peg_trial/debug/{cam}", 1)

        self.pub = self.create_publisher(String, "/peg_trial/scene", 10)
        self.create_timer(1.0 / self.task["vision"].get("rate_hz", 5.0), self._tick)
        self.get_logger().info(f"vision up, cameras: {list(self.calib['cameras'])}")

    def _on_info(self, cam, msg: CameraInfo):
        self.K[cam] = np.array(msg.k).reshape(3, 3)

    def _on_rgbd(self, cam, color_msg, depth_msg):
        bgr = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = depth_to_m(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"), self.calib.get("depth_scale", 0.001))
        self.frames[cam] = (bgr, depth)

    def _tick(self):
        results = []
        for cam, (bgr, depth) in list(self.frames.items()):
            if cam not in self.K:
                continue
            T = np.array(self.calib["cameras"][cam]["T_base_cam"], float)
            try:
                r = process_frame(bgr, depth, self.K[cam], T, self.colors, self.task, cam)
            except Exception as e:  # never crash the node on a bad frame
                self.get_logger().warn(f"{cam}: {e}")
                continue
            results.append(r)
            if self.debug_pub[cam].get_subscription_count() > 0:
                self.debug_pub[cam].publish(self.bridge.cv2_to_imgmsg(draw(bgr, r, self.K[cam], T), "bgr8"))
        if not results:
            return
        scene = fuse(results, self.task["vision"].get("fuse_merge_dist", 0.03))
        scene["stamp"] = self.get_clock().now().nanoseconds * 1e-9

        def _json_serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return str(obj)

        self.pub.publish(String(data=json.dumps(scene, default=_json_serialize)))


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
