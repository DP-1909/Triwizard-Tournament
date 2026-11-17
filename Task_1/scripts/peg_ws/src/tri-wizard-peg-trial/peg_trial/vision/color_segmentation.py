"""Step 1 of the vision stack: find each coloured peg with an HSV threshold.

HSV is used instead of RGB because "hue" (the colour itself) changes much
less than RGB values when the lighting gets brighter or darker.
OpenCV ranges: H 0-179, S 0-255, V 0-255. Red wraps around 0, so a colour
may have several [lower, upper] ranges which are OR-ed together.
"""
from __future__ import annotations

import cv2
import numpy as np


def hsv_mask(bgr: np.ndarray, ranges, open_k: int = 5, close_k: int = 7) -> np.ndarray:
    """Binary mask (0/255) of pixels inside any of the HSV `ranges`.

    ranges: [[[h,s,v], [h,s,v]], ...]  list of (lower, upper) pairs
    open_k / close_k: kernel sizes for morphology
        opening  = erode then dilate -> removes small noise specks
        closing  = dilate then erode -> fills small holes inside the blob
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return mask


def blobs(mask: np.ndarray, min_area: float = 300.0, max_count: int = 1):
    """Largest external contours in the mask (biggest first)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours[:max_count]


def segment_pegs(bgr: np.ndarray, color_cfg: dict, min_area: float = 300.0):
    """Return {color_name: (contour, mask)} for every peg colour that was found."""
    morph = color_cfg.get("morphology", {})
    out = {}
    for name, spec in color_cfg["pegs"].items():
        m = hsv_mask(bgr, spec["hsv"], morph.get("open", 5), morph.get("close", 7))
        found = blobs(m, min_area, 1)
        if found:
            single = np.zeros_like(m)
            cv2.drawContours(single, found, -1, 255, cv2.FILLED)
            out[name] = (found[0], single)
    return out
