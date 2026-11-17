"""Step 4: find the sockets (holes) in the board and their shapes.

Idea: threshold the BOARD colour -> take the board's outer outline and fill
it -> subtract the board mask. Whatever is left inside the board outline
is "not board" = a hole. Each hole is then classified exactly like a peg.

Run this when the board is EMPTY (at start-up). The autonomy node caches
the socket poses because the board does not move during the trial.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color_segmentation import hsv_mask
from .shape_detection import classify_shape


def find_holes(bgr: np.ndarray, board_hsv_ranges, min_area: float = 200.0, max_area_frac: float = 0.3, peg_contours=None):
    """Return (board_mask, [hole_contour, ...])."""
    board = hsv_mask(bgr, board_hsv_ranges, open_k=3, close_k=3)

    # Bridge vertical cuts/glares across the long cardboard strip
    k_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
    closed = cv2.morphologyEx(board, cv2.MORPH_CLOSE, k_bridge)

    outer, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outer:
        return board, []

    # Prefer the tall vertical strip (board)
    candidates = [c for c in outer if cv2.boundingRect(c)[3] > 160]
    outline = max(candidates if candidates else outer, key=cv2.contourArea)
    board_area = cv2.contourArea(outline)

    filled = np.zeros_like(board)
    cv2.drawContours(filled, [outline], -1, 255, cv2.FILLED)
    holes_mask = cv2.subtract(filled, board)

    # If pegs are on the board (e.g. already placed in sockets), mask them out
    # so they don't merge with adjacent empty sockets or cause false socket detections.
    if peg_contours:
        peg_cutout = np.zeros_like(board)
        for cnt in peg_contours:
            if cnt is None or len(cnt) < 3:
                continue
            pm = cv2.moments(cnt)
            if pm["m00"] > 0:
                cx, cy = int(pm["m10"] / pm["m00"]), int(pm["m01"] / pm["m00"])
                if cv2.pointPolygonTest(outline, (cx, cy), False) >= 0:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    # Cover the placed peg region (expanded slightly in x and bottom, but not top
                    # so we don't encroach into an adjacent socket above)
                    cv2.rectangle(peg_cutout, (bx - 5, by - 2), (bx + bw + 5, by + bh + 5), 255, cv2.FILLED)
        holes_mask = cv2.subtract(holes_mask, peg_cutout)

    holes_mask = cv2.morphologyEx(holes_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    holes, _ = cv2.findContours(holes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = [h for h in holes if min_area <= cv2.contourArea(h) <= max_area_frac * board_area]
    return board, holes


def ring_mask(shape_hw, contour, inner_px: int = 6, outer_px: int = 18):
    """Pixels in a ring AROUND the hole -> used to read the board-top depth."""
    m_out = np.zeros(shape_hw, np.uint8)
    cv2.drawContours(m_out, [contour], -1, 255, cv2.FILLED)
    big = cv2.dilate(m_out, np.ones((outer_px, outer_px), np.uint8))
    small = cv2.dilate(m_out, np.ones((inner_px, inner_px), np.uint8))
    return cv2.subtract(big, small)


def classify_holes(holes, eps_frac=0.03):
    return [(h,) + classify_shape(h, eps_frac) for h in holes]   # (contour, shape, verts, conf)
