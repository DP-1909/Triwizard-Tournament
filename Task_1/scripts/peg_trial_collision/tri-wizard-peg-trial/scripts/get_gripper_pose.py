#!/usr/bin/env python3
"""Print the current gripper Cartesian coordinates (x, y, z) in the robot base frame.

Usage:
  1. Live from ROS 2 (when hardware.launch.py is running):
       python3 scripts/get_gripper_pose.py

  2. From Dynamixel Wizard angles (degrees):
       python3 scripts/get_gripper_pose.py --deg 229.2 174.6 199.2 248.5
"""
import argparse
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from peg_trial.planning.kinematics import fk  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Get gripper (x, y, z) in robot base frame")
    parser.add_argument("--deg", nargs=4, type=float, help="Joint 1-4 angles in degrees from Dynamixel Wizard")
    args = parser.parse_args()

    if args.deg:
        # Dynamixel center is 180 degrees (0 rad)
        q1 = math.radians(args.deg[0] - 180.0)
        q2 = math.radians(args.deg[1] - 180.0)
        q3 = math.radians(args.deg[2] - 180.0)
        q4 = math.radians(args.deg[3] - 180.0)
        q = [q1, q2, q3, q4]
        x, y, z, pitch = fk(q)
        print("\n--- FROM DYNAMIXEL WIZARD ANGLES ---")
        print(f"Raw Wizard Degrees: {args.deg}")
        print(f"Joint Angles (rad): [q1={q1:.4f}, q2={q2:.4f}, q3={q3:.4f}, q4={q4:.4f}]")
        print(f"Gripper Center Position:")
        print(f"  X = {x:.4f} m  ({x*100:.1f} cm)")
        print(f"  Y = {y:.4f} m  ({y*100:.1f} cm)")
        print(f"  Z = {z:.4f} m  ({z*100:.1f} cm)")
        print(f"  Pitch = {math.degrees(pitch):.1f}° (90° = straight down)")
        print("\nUse this for calibration:")
        print(f"python3 scripts/calibrate_extrinsics.py --camera cam1 --size 0.037 --x {x:.3f} --y {y:.3f} --z {z:.3f}\n")
        return

    # If no angles provided, read live from ROS 2
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("get_gripper_pose")
    got = []

    def cb(msg: JointState):
        name_map = {n: p for n, p in zip(msg.name, msg.position)}
        if all(f"joint{i}" in name_map for i in range(1, 5)):
            got.append([name_map[f"joint{i}"] for i in range(1, 5)])

    node.create_subscription(JointState, "/joint_states", cb, 10)
    print("Waiting for /joint_states from robot...")
    import time
    t0 = time.time()
    while not got and time.time() - t0 < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if not got:
        print("ERROR: Did not receive /joint_states within 3s. Is hardware.launch.py running?")
        print("Tip: You can also pass angles directly: python3 scripts/get_gripper_pose.py --deg 229.2 174.6 199.2 248.5")
        return

    q = got[-1]
    x, y, z, pitch = fk(q)
    print("\n--- LIVE ROBOT GRIPPER POSE ---")
    print(f"Joint Angles (rad): [q1={q[0]:.4f}, q2={q[1]:.4f}, q3={q[2]:.4f}, q4={q[3]:.4f}]")
    print(f"Gripper Position:")
    print(f"  X = {x:.4f} m  ({x*100:.1f} cm)")
    print(f"  Y = {y:.4f} m  ({y*100:.1f} cm)")
    print(f"  Z = {z:.4f} m  ({z*100:.1f} cm)")
    print(f"  Pitch = {math.degrees(pitch):.1f}°")
    print("\nUse this for calibration:")
    print(f"python3 scripts/calibrate_extrinsics.py --camera cam1 --size 0.037 --x {x:.3f} --y {y:.3f} --z {z:.3f}\n")


if __name__ == "__main__":
    main()
