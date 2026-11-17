#!/usr/bin/env python3
"""Run the vision pipeline WITHOUT ROS.

  python3 scripts/offline_vision_test.py --synthetic        # fake scene, prints errors vs truth
  python3 scripts/offline_vision_test.py --data data        # frames saved by save_frames.py

Writes debug overlays to out/<cam>_debug.png.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from peg_trial.utils.config import load  # noqa: E402
from peg_trial.utils.synthetic import make_scene  # noqa: E402
from peg_trial.vision.scene import draw, fuse, process_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    colors, calib, task = load("colors.yaml"), load("camera_calib.yaml"), load("task.yaml")
    os.makedirs("out", exist_ok=True)

    results, truth = [], None
    for cam, c in calib["cameras"].items():
        T = np.array(c["T_base_cam"], float)
        if a.synthetic:
            bgr, depth, K, truth = make_scene(T, table_z=task["table_z"], seed=a.seed)
        else:
            p = os.path.join(a.data, f"{cam}_color.png")
            if not os.path.exists(p):
                print(f"{cam}: no saved frame, skipped")
                continue
            bgr = cv2.imread(p)
            depth = np.load(os.path.join(a.data, f"{cam}_depth.npy"))
            K = np.load(os.path.join(a.data, f"{cam}_K.npy"))
        r = process_frame(bgr, depth, K, T, colors, task, cam)
        results.append(r)
        cv2.imwrite(f"out/{cam}_debug.png", draw(bgr, r, K, T))
        print(f"{cam}: {len(r['pegs'])} pegs, {len(r['sockets'])} sockets")

    scene = fuse(results, task["vision"]["fuse_merge_dist"])
    print(json.dumps(scene, indent=1))

    if truth:
        print("\n--- error vs ground truth ---")
        for t in truth["pegs"]:
            d = next((p for p in scene["pegs"] if p["color"] == t["color"]), None)
            if d is None:
                print(f"peg {t['color']}: MISSING")
                continue
            print(f"peg {t['color']:6s} true {t['shape']:8s} got {d['shape']:8s} "
                  f"xy err {1000 * math.hypot(d['x'] - t['x'], d['y'] - t['y']):5.1f} mm  "
                  f"z err {1000 * (d['z_top'] - t['z_top']):5.1f} mm")
        for t in truth["sockets"]:
            d = min(scene["sockets"], key=lambda s: math.hypot(s["x"] - t["x"], s["y"] - t["y"]), default=None)
            if d is None:
                print("socket MISSING")
                continue
            print(f"socket true {t['shape']:8s} got {d['shape']:8s} "
                  f"xy err {1000 * math.hypot(d['x'] - t['x'], d['y'] - t['y']):5.1f} mm")


if __name__ == "__main__":
    main()
