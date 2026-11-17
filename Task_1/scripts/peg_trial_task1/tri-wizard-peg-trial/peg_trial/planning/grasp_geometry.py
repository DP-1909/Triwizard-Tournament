"""Grasping a peg that lies at a RANDOM rotation (yaw).

Facts for the OMX (4 joints, no wrist roll):
  * With the gripper pointing down, the fingers always close along the TANGENTIAL
    direction (perpendicular to the line robot-base -> gripper). We cannot turn them.
  * So the gripper must open wider than the peg is wide IN THAT DIRECTION, which depends
    on the peg's random yaw (a 45 mm square seen corner-first is 64 mm wide!).
  * When parallel fingers squeeze a polygon, the peg rotates until a flat face lies
    against a finger ("squeeze grasp"). After that, the peg's yaw relative to the arm
    is KNOWN, whatever it was before. That is what we use to predict the yaw at the socket.

Yaw convention = vision's polygon_yaw: angle (rad) of one corner, seen from above, in the
robot base frame, defined modulo 360°/n.
"""
from __future__ import annotations

import math

from .kinematics import J1_X

SIDES = {"triangle": 3, "square": 4, "hexagon": 6, "circle": 0}


def heading(x, y):
    """Direction from the robot base axis to the point (= gripper heading, joint1 angle)."""
    return math.atan2(y, x - J1_X)


def _corners(shape, yaw):
    n = SIDES[shape]
    return [yaw + 2 * math.pi * k / n for k in range(n)]


def width_along(shape, size, yaw, direction):
    """Width (m) of the peg's top outline measured along `direction` (rad). size = corner-to-corner."""
    R = size / 2
    if SIDES.get(shape, 0) == 0:
        return 2 * R
    c = [math.cos(a - direction) for a in _corners(shape, yaw)]
    return R * (max(c) - min(c))


def open_width_needed(peg):
    """Finger gap needed to come down AROUND the peg at its current (random) yaw."""
    t = heading(peg["x"], peg["y"]) + math.pi / 2           # finger closing direction
    return width_along(peg["shape"], peg.get("size", 0.05), peg.get("yaw", 0.0), t)


def held_width(shape, size):
    """Finger gap after the squeeze (face against finger)."""
    R = size / 2
    return {"square": R * math.sqrt(2), "hexagon": R * math.sqrt(3), "triangle": 1.5 * R}.get(shape, 2 * R)


def _wrap(a, period):
    a = a % period
    return a - period if a > period / 2 else a


def yaw_after_squeeze(peg):
    """Peg yaw (world) right after the fingers close, from the squeeze-grasp model."""
    n = SIDES.get(peg["shape"], 0)
    if n == 0:
        return peg.get("yaw", 0.0)
    psi = heading(peg["x"], peg["y"])
    period = 2 * math.pi / n
    if n == 4:          # faces end up perpendicular to the finger axis -> corners at +45 deg
        cands = [psi + math.pi / 4]
    elif n == 6:        # flats against the fingers -> corners along the arm heading
        cands = [psi]
    else:               # triangle: a face lands on the left OR right finger
        cands = [psi - math.pi / 6, psi + math.pi / 6]
    y0 = peg.get("yaw", 0.0)
    # the peg turns the shorter way -> pick the outcome closest to where it started
    return min(cands, key=lambda c: abs(_wrap(c - y0, period))) % period


def predicted_socket_error(peg, sock):
    """Yaw misalignment (rad) between the carried peg and the socket when it arrives.

    The peg turns with joint1 while carried: + (heading(socket) - heading(peg)).
    """
    n = SIDES.get(peg["shape"], 0)
    if n == 0:
        return 0.0
    period = 2 * math.pi / n
    arrive = yaw_after_squeeze(peg) + heading(sock["x"], sock["y"]) - heading(peg["x"], peg["y"])
    return _wrap(arrive - sock.get("yaw", 0.0), period)


def known_size(shape, cfg):
    """Corner-to-corner size (m) from the measured peg dimensions in task.yaml, or None."""
    g = (cfg or {}).get("peg_geometry", {}).get(shape)
    if not g:
        return None
    if shape == "circle":
        return g["diameter"]
    a = g["side"]
    return {"square": a * math.sqrt(2), "triangle": 2 * a / math.sqrt(3), "hexagon": 2 * a}[shape]


def with_known_size(peg, cfg):
    s = known_size(peg.get("shape"), cfg)
    return dict(peg, size=s) if s else peg
