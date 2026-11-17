#!/usr/bin/env python3
"""Find HSV thresholds for one colour with sliders (needs a desktop / display).

    python3 scripts/tune_hsv.py --image data/cam1_color.png

Move the sliders until ONLY the peg is white in the "mask" window, then copy
the printed [[h,s,v],[h,s,v]] into config/colors.yaml. Press q to quit.
Tip: click on a pixel in the "image" window to print its HSV value.
"""
import argparse
import os

os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("mask")
    for n, v, mx in (("H lo", 0, 179), ("H hi", 179, 179), ("S lo", 100, 255), ("S hi", 255, 255),
                     ("V lo", 60, 255), ("V hi", 255, 255)):
        cv2.createTrackbar(n, "mask", v, mx, lambda _: None)

    def click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            print("HSV at", (x, y), "=", hsv[y, x].tolist())
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click)

    last = None
    while True:
        g = [cv2.getTrackbarPos(n, "mask") for n in ("H lo", "S lo", "V lo", "H hi", "S hi", "V hi")]
        lo, hi = g[:3], g[3:]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cv2.imshow("image", img)
        cv2.imshow("mask", mask)
        if (lo, hi) != last:
            print(f"hsv: [[{lo}, {hi}]]")
            last = (lo, hi)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
