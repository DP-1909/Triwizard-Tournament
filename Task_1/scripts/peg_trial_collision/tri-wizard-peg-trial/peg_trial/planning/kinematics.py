"""Forward / inverse kinematics for ROBOTIS OpenMANIPULATOR-X (4-DOF + gripper).

Frame used everywhere in this repo = robot BASE frame (OMX `link1` / `world`):
    x forward, y left, z up, origin at the centre of the base plate (table level).

Link dimensions come from the official OMX URDF:
    joint1 at (0.012, 0, 0.017)      base yaw   (about z)
    joint2 at +0.0595 in z           shoulder   (about y)
    joint3 at (+0.024, 0, +0.128)    elbow      (about y)
    joint4 at (+0.124, 0, 0)         wrist      (about y)
    tool   at (+0.126, 0, 0)         gripper centre (end_effector_link)

Sign convention (same as ROS): positive joint2/3/4 tilt the arm DOWN/forward.
Pitch `phi` = angle of the gripper BELOW horizontal = q2 + q3 + q4.
    phi = 0      -> gripper points straight forward
    phi = pi/2   -> gripper points straight down  (TOP GRASP)

The OMX has NO wrist-roll joint, so in a top grasp the gripper's yaw
is always atan2(y, x) of the target. See planner for what that means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

J1_X = 0.012                      # joint1 axis offset in x
J2_Z = 0.017 + 0.0595             # shoulder height above base
A_X, A_Z = 0.024, 0.128           # link2 vector at q2 = 0
LA = math.hypot(A_X, A_Z)         # 0.1302 m
OFF = math.atan2(A_X, A_Z)        # 0.1853 rad, the "bent" link2 offset
LB = 0.124                        # link3
LC = 0.126                        # link4 -> gripper centre

# URDF joint limits (rad)
LIMITS = [(-math.pi, math.pi), (-2.05, 1.57), (-1.67, 1.53), (-1.80, 2.00)]


class IKError(Exception):
    """Raised when a target is outside the robot's reachable workspace."""


@dataclass
class Pose:
    x: float
    y: float
    z: float
    pitch: float = math.pi / 2    # default: gripper pointing down


def fk(q):
    """Joint angles [q1..q4] -> (x, y, z, pitch) of the gripper centre."""
    q1, q2, q3, q4 = q
    # work in the vertical plane of the arm: r = radial distance, h = height
    r = A_X * math.cos(q2) + A_Z * math.sin(q2)
    h = -A_X * math.sin(q2) + A_Z * math.cos(q2)
    r += LB * math.cos(q2 + q3)
    h += -LB * math.sin(q2 + q3)
    r += LC * math.cos(q2 + q3 + q4)
    h += -LC * math.sin(q2 + q3 + q4)
    x = J1_X + r * math.cos(q1)
    y = r * math.sin(q1)
    z = J2_Z + h
    return x, y, z, q2 + q3 + q4


def ik(pose: Pose, elbow_up: bool = True, check_limits: bool = True):
    """(x, y, z, pitch) -> [q1, q2, q3, q4]. Raises IKError if unreachable."""
    dx = pose.x - J1_X
    q1 = math.atan2(pose.y, dx)
    r = math.hypot(dx, pose.y)

    # 1) step back from the gripper centre to the wrist (joint4)
    rw = r - LC * math.cos(pose.pitch)
    hw = (pose.z + LC * math.sin(pose.pitch)) - J2_Z

    # 2) solve the 2-link (LA, LB) planar problem for the wrist point
    d2 = rw * rw + hw * hw
    cos_b = (d2 - LA * LA - LB * LB) / (2 * LA * LB)
    if abs(cos_b) > 1.0:
        raise IKError(f"target out of reach (r={r:.3f}, z={pose.z:.3f})")
    beta = -math.acos(cos_b) if elbow_up else math.acos(cos_b)
    # elevation (angle above horizontal) of link2 and link3
    e2 = math.atan2(hw, rw) - math.atan2(LB * math.sin(beta), LA + LB * math.cos(beta))
    e3 = e2 + beta

    q2 = math.pi / 2 - OFF - e2
    q3 = -e3 - q2
    q4 = pose.pitch - q2 - q3
    q = [q1, q2, q3, q4]
    q = [math.atan2(math.sin(a), math.cos(a)) for a in q]   # wrap to [-pi, pi]

    if check_limits:
        for i, (a, (lo, hi)) in enumerate(zip(q, LIMITS)):
            if not lo <= a <= hi:
                raise IKError(f"joint{i + 1}={a:.2f} rad outside [{lo}, {hi}]")
    return q


def ik_best(pose: Pose, pitch_tolerance: float = 0.0, step: float = 0.05):
    """Try the requested pitch first, then nearby pitches (helps far targets)."""
    tries = [0.0]
    k = step
    while k <= pitch_tolerance + 1e-9:
        tries += [-k, k]
        k += step
    last = None
    for dp in tries:
        try:
            return ik(Pose(pose.x, pose.y, pose.z, pose.pitch + dp))
        except IKError as e:
            last = e
    raise last


def interpolate_line(p0: Pose, p1: Pose, max_step: float = 0.01):
    """Straight Cartesian line p0 -> p1 as a list of poses (for slow, straight moves)."""
    a = np.array([p0.x, p0.y, p0.z, p0.pitch])
    b = np.array([p1.x, p1.y, p1.z, p1.pitch])
    n = max(1, int(math.ceil(np.linalg.norm(b[:3] - a[:3]) / max_step)))
    return [Pose(*(a + (b - a) * i / n)) for i in range(1, n + 1)]
