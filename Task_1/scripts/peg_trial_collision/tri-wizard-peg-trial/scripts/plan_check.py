#!/usr/bin/env python3
"""Check the collision-free plan BEFORE moving the robot, and draw it (top view).

  # 1) save what vision sees right now (vision node running):
  ros2 topic echo --once --full-length /peg_trial/scene --field data > scene.json
  # 2) plan + picture (no ROS needed):
  python3 scripts/plan_check.py --scene scene.json            # -> out/plan_top_view.png
  python3 scripts/plan_check.py --demo                         # built-in example layout

Prints the placement order, grasp height, carry height and route ("over" / "around")
for each peg, or the exact reason a peg cannot be done safely.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from peg_trial.planning.kinematics import fk  # noqa: E402
from peg_trial.planning.pick_place_planner import match_pegs_to_sockets, plan_order, plan_pick_place  # noqa: E402
from peg_trial.utils.config import load  # noqa: E402

COL = {"blue": (255, 120, 0), "yellow": (0, 230, 255), "orange": (0, 140, 255), "green": (0, 200, 0),
       "red": (0, 0, 220)}


def demo_scene(tz):
    names = [("blue", "square"), ("yellow", "triangle"), ("orange", "hexagon"), ("green", "circle")]
    P = [(0.09, 0.06), (0.17, 0.06), (0.09, 0.14), (0.165, 0.135)]
    S = [(0.09, -0.06), (0.17, -0.06), (0.09, -0.14), (0.17, -0.14)]
    return {"pegs": [dict(color=n[0], shape=n[1], x=x, y=y, z_top=tz + 0.098, yaw=0, size=0.045)
                     for n, (x, y) in zip(names, P)],
            "sockets": [dict(shape=n[1], x=x, y=y, z_top=tz + 0.012, yaw=0, size=0.05)
                        for n, (x, y) in zip(names, S)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="JSON from /peg_trial/scene")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--table-z", type=float, help="override task.yaml table_z")
    a = ap.parse_args()
    cfg = load("task.yaml")
    if a.table_z is not None:
        cfg["table_z"] = a.table_z
    if a.scene:
        txt = open(a.scene).read().strip()
        if txt.startswith("'") or txt.startswith('"'):
            txt = txt.strip("'\"")
        scene = json.loads(txt)
    else:
        scene = demo_scene(cfg["table_z"])

    pairs = match_pegs_to_sockets(scene["pegs"], scene["sockets"])
    order, errs = plan_order(pairs, cfg, scene["sockets"], preferred=cfg["execution"]["order"])
    print(f"table_z = {cfg['table_z']}  |  collision-free for {len(order)}/{len(pairs)} pegs")
    for c, e in errs.items():
        if c not in [p["color"] for p, _ in order]:
            print(f"  ✗ {c}: {e}")

    # draw top view: 1 px = 1 mm, robot base at the left middle
    W, H, ox, oy = 420, 520, 60, 260
    img = np.full((H, W, 3), 255, np.uint8)
    to_px = lambda x, y: (int(ox + x * 1000), int(oy - y * 1000))   # noqa: E731
    for r in (0.10, 0.15, 0.20, 0.25):
        cv2.circle(img, to_px(0.012, 0), int(r * 1000), (230, 230, 230), 1)
    cv2.circle(img, to_px(0, 0), 25, (80, 80, 80), -1)
    cv2.putText(img, "robot", (ox - 25, oy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    for s in scene["sockets"]:
        cv2.circle(img, to_px(s["x"], s["y"]), int(s.get("size", 0.05) * 500), (180, 180, 180), 2)
        cv2.putText(img, s["shape"][:3], to_px(s["x"] - 0.012, s["y"] - 0.035), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1)
    for p in scene["pegs"]:
        cv2.circle(img, to_px(p["x"], p["y"]), int(p.get("size", 0.045) * 500), COL.get(p["color"], (0, 0, 0)), -1)

    done = []
    for i, (peg, sock) in enumerate(order):
        others = [q for q, _ in pairs if q is not peg and q["color"] not in {d["color"] for d, _ in done}]
        steps, info = plan_pick_place(peg, sock, cfg, None, others, done, scene["sockets"])
        done.append((peg, sock))
        print(f"  {i + 1}. {peg['color']:7s} grip {info['grasp_depth'] * 1000:.0f} mm below top | "
              f"carry z {info['carry_z']:.3f} | route {info['route']:6s} | yaw err {info['yaw_error_deg']:+.0f}°")
        pts = [fk(q)[:2] for s in steps if s["type"] == "move" for q, _ in s["points"]]
        colr = COL.get(peg["color"], (0, 0, 0))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(img, to_px(*p0), to_px(*p1), colr, 1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), to_px(peg["x"] - 0.004, peg["y"] - 0.005), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)
    os.makedirs("out", exist_ok=True)
    cv2.imwrite("out/plan_top_view.png", img)
    print("picture: out/plan_top_view.png  (x forward = right, y left = up, rings every 5 cm)")


if __name__ == "__main__":
    main()
