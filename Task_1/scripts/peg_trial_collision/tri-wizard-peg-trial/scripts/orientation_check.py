#!/usr/bin/env python3
"""ORIENTATION: will each peg arrive turned the same way as its socket?  Where should the robot stand?

Why this matters (OMX = 4 joints, no wrist roll):
  * the fingers always close sideways (perpendicular to the line robot-base -> gripper);
  * closing turns the peg until a flat face lies on a finger ("squeeze grasp"), so the peg's
    rotation relative to the arm after grasping is FIXED by its shape, not by how it was lying;
  * so the peg's rotation at a socket depends only on WHERE the socket is relative to the robot base.
For zero error each socket must be turned like this (seen from above):
    square   : one side facing the robot base
    hexagon  : one corner pointing at the robot base (two flats parallel to the line to the base)
    triangle : one side parallel to the line to the base (two mirror options; which one the
               peg ends up in depends on how it was lying -> 50 % chance unless the socket has clearance)
    circle   : never matters

Usage
  ros2 topic echo --once --full-length /peg_trial/scene --field data > scene.json   # board EMPTY
  python3 scripts/orientation_check.py --scene scene.json
  python3 scripts/orientation_check.py --demo
Prints the predicted error per socket now, and the best place to move the robot base.
After moving the robot you MUST re-calibrate the camera (T_base_cam changes).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peg_trial.planning.grasp_geometry import SIDES  # noqa: E402
from peg_trial.planning.kinematics import J1_X  # noqa: E402


def wrap(a, period):
    a = a % period
    return a - period if a > period / 2 else a


def errors_for(sock, dx=0.0, dy=0.0):
    """Predicted peg-vs-socket yaw error(s) in degrees if the robot base is moved by (dx, dy)."""
    n = SIDES.get(sock["shape"], 0)
    if n == 0:
        return [0.0]
    x, y = sock["x"] - dx, sock["y"] - dy
    hd = math.atan2(y, x - J1_X)
    period = 2 * math.pi / n
    arrive = {4: [hd + math.pi / 4], 6: [hd], 3: [hd - math.pi / 6, hd + math.pi / 6]}[n]
    return [math.degrees(wrap(a - sock.get("yaw", 0.0), period)) for a in arrive]


def score(sockets, dx, dy, rmin=0.10, rmax=0.21):
    worst = 0.0
    for s in sockets:
        r = math.hypot(s["x"] - dx - J1_X, s["y"] - dy)
        if not rmin <= r <= rmax:
            return None
        e = errors_for(s, dx, dy)
        worst = max(worst, min(abs(v) for v in e))       # triangle: the better of its two outcomes
    return worst


def demo():
    return {"sockets": [
        dict(shape="square", x=0.09, y=-0.06, yaw=math.radians(45)),
        dict(shape="triangle", x=0.17, y=-0.06, yaw=math.radians(10)),
        dict(shape="hexagon", x=0.09, y=-0.14, yaw=math.radians(0)),
        dict(shape="circle", x=0.17, y=-0.14, yaw=0.0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--range", type=float, default=0.10, help="how far (m) the robot may be moved")
    a = ap.parse_args()
    scene = json.loads(open(a.scene).read().strip().strip("'\"")) if a.scene else demo()
    socks = scene["sockets"]

    print("Now (robot where it is):")
    for s in socks:
        e = errors_for(s)
        txt = " or ".join(f"{v:+.0f}°" for v in e)
        print(f"  {s['shape']:8s} socket at ({s['x']:.3f},{s['y']:.3f}): peg arrives {txt} off")

    best = None
    st = 0.0025
    k = int(a.range / st)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            c = score(socks, i * st, j * st)
            if c is not None and (best is None or c < best[0] - 1e-9):
                best = (c, i * st, j * st)
    if best is None:
        print("\nNo robot position keeps every socket in reach.")
        return
    c, dx, dy = best
    print(f"\nBest: move the robot base {dx * 100:+.1f} cm along its x (forward) and {dy * 100:+.1f} cm along "
          f"its y (left)\n      -> worst remaining error {c:.1f}° (triangle counted at its better outcome)")
    for s in socks:
        e = errors_for(s, dx, dy)
        print(f"  {s['shape']:8s}: " + " or ".join(f"{v:+.1f}°" for v in e))
    print("\nThen: re-calibrate the camera, and compare the errors with your socket clearance:")
    print("  allowed error ≈ atan(clearance / peg half-width), e.g. 1 mm clearance on a 40 mm square ≈ 3°.")


if __name__ == "__main__":
    main()
