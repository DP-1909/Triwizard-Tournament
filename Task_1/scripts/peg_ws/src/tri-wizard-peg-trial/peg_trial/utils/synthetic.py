"""Render a fake RGB-D frame of the peg scene (for testing WITHOUT a robot/camera).

It draws the table, a white board with 4 shaped holes, and 4 coloured prism
pegs (top face + side walls) as seen by a pinhole camera at T_base_cam.
Ground truth is returned so tests can check the vision accuracy.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

COLORS_BGR = {"red": (0, 0, 200), "green": (0, 170, 0), "blue": (200, 60, 0), "yellow": (0, 220, 230)}
SIDES = {"triangle": 3, "square": 4, "hexagon": 6, "circle": 48}


def polygon(shape, cx, cy, size, yaw):
    """Corner points (x, y) of a regular shape; size = circum-diameter."""
    n = SIDES[shape]
    r = size / 2
    return np.array([[cx + r * math.cos(yaw + 2 * math.pi * i / n), cy + r * math.sin(yaw + 2 * math.pi * i / n)]
                     for i in range(n)])


def default_K(w=640, h=480, f=600.0):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


class Renderer:
    def __init__(self, K, T_base_cam, w=640, h=480):
        self.K, self.w, self.h = np.asarray(K, float), w, h
        self.T_cam_base = np.linalg.inv(np.asarray(T_base_cam, float))
        self.rgb = np.zeros((h, w, 3), np.uint8)
        self.depth = np.full((h, w), np.inf, np.float32)

    def project(self, pts_xyz):
        p = np.hstack([pts_xyz, np.ones((len(pts_xyz), 1))]) @ self.T_cam_base.T
        uv = (p[:, :3] @ self.K.T)
        return uv[:, :2] / uv[:, 2:3], p[:, 2]

    def fill(self, pts_xyz, color):
        """Z-buffered flat polygon; per-pixel depth from the polygon's plane."""
        uv, z = self.project(pts_xyz)
        if np.any(z <= 0.01):
            return
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
        vs, us = np.nonzero(m)
        if us.size == 0:
            return
        pc = (np.c_[pts_xyz, np.ones(len(pts_xyz))] @ self.T_cam_base.T)[:, :3]
        n = np.cross(pc[1] - pc[0], pc[2] - pc[0])
        rays = np.linalg.inv(self.K) @ np.stack([us, vs, np.ones_like(us)]).astype(float)
        denom = n @ rays
        d = np.where(np.abs(denom) > 1e-9, (n @ pc[0]) / np.where(denom == 0, 1, denom), np.inf)
        sel = d < self.depth[vs, us]
        self.rgb[vs[sel], us[sel]] = color
        self.depth[vs[sel], us[sel]] = d[sel]

    def plane_depth(self, sel, z_plane):
        """Camera depth of the horizontal plane z = z_plane at the selected pixels."""
        vs, us = np.nonzero(sel)
        T = np.linalg.inv(self.T_cam_base)
        rays = T[:3, :3] @ (np.linalg.inv(self.K) @ np.stack([us, vs, np.ones_like(us)]).astype(float))
        return (z_plane - T[2, 3]) / rays[2]          # ray z-component of K^-1 is 1 -> t = depth

    def prism(self, xy, z0, z1, color):
        side = tuple(int(c * 0.7) for c in color)
        n = len(xy)
        for i in range(n):          # side walls first, top face last
            a, b = xy[i], xy[(i + 1) % n]
            self.fill(np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]]), side)
        self.fill(np.c_[xy, np.full(n, z1)], color)


def make_scene(T_base_cam, K=None, table_z=-0.05, seed=0, peg_size=0.05, board_th=0.02, noise_mm=1.0):
    rng = np.random.default_rng(seed)
    K = default_K() if K is None else K
    R = Renderer(K, T_base_cam)
    R.rgb[:] = (90, 110, 130)
    # table: a big quad far away covers everything
    R.depth[:] = 10.0
    R.fill(np.array([[-0.5, -1, table_z], [1, -1, table_z], [1, 1, table_z], [-0.5, 1, table_z]]), (90, 110, 130))

    truth = {"pegs": [], "sockets": []}
    shapes = ["triangle", "square", "hexagon", "circle"]
    board_top = table_z + board_th
    # board (white) with 4 holes: draw board, then holes as table-depth dark polygons
    # layout kept inside the OMX reach (r = 0.12 ... 0.23 m from the base)
    R.fill(np.array([[0.075, -0.18, board_top], [0.235, -0.18, board_top], [0.235, -0.025, board_top],
                     [0.075, -0.025, board_top]]), (235, 235, 235))
    for i, sh in enumerate(shapes):
        cx, cy = [(0.12, -0.065), (0.19, -0.065), (0.12, -0.14), (0.19, -0.14)][i]
        yaw = float(rng.uniform(0, 2 * math.pi / (SIDES[sh] if sh != "circle" else 1)))
        xy = polygon(sh, cx, cy, peg_size * 1.1, yaw)
        uv, _ = R.project(np.c_[xy, np.full(len(xy), board_top)])
        m = np.zeros((R.h, R.w), np.uint8)
        cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
        R.rgb[m > 0] = (40, 40, 40)
        R.depth[m > 0] = R.plane_depth(m > 0, table_z)      # see through the hole to the table
        truth["sockets"].append(dict(shape=sh, x=cx, y=cy, z_top=board_top, yaw=yaw))

    peg_xy = [(0.13, 0.07), (0.19, 0.06), (0.10, 0.14), (0.16, 0.13)]
    colors = ["red", "green", "blue", "yellow"]
    order = rng.permutation(4)
    for k, i in enumerate(order):
        sh, col = shapes[i], colors[k]
        cx, cy = peg_xy[k]
        yaw = float(rng.uniform(0, 2 * math.pi))
        R.prism(polygon(sh, cx, cy, peg_size, yaw), table_z, table_z + 0.10, COLORS_BGR[col])
        truth["pegs"].append(dict(color=col, shape=sh, x=cx, y=cy, z_top=table_z + 0.10, yaw=yaw))

    depth = R.depth + rng.normal(0, noise_mm / 1000, R.depth.shape).astype(np.float32)
    rgb = np.clip(R.rgb.astype(np.int16) + rng.normal(0, 4, R.rgb.shape), 0, 255).astype(np.uint8)
    return rgb, depth, K, truth
