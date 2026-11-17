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


def find_holes(bgr: np.ndarray, board_hsv_ranges, min_area: float = 200.0, max_area_frac: float = 0.3):
    """Return (board_mask, [hole_contour, ...])."""
    board = hsv_mask(bgr, board_hsv_ranges, open_k=3, close_k=3)
    outer, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outer:
        return board, []
    outline = max(outer, key=cv2.contourArea)
    board_area = cv2.contourArea(outline)

    filled = np.zeros_like(board)
    cv2.drawContours(filled, [outline], -1, 255, cv2.FILLED)
    holes_mask = cv2.subtract(filled, board)
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
