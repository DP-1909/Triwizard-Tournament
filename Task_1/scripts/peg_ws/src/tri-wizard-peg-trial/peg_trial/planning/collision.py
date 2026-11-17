"""Simple 3-D collision model: gripper + held peg  vs  other pegs, placed pegs, board, table.

Why so simple? With a top-down grasp only the GRIPPER and the PEG it carries
come down near the other pegs (the wrist and upper links stay above them),
so we model:

  * every other peg       -> vertical cylinder  (from vision: x, y, size, z_top)
  * a peg already placed  -> vertical cylinder standing in its socket
  * the board             -> flat slab (top = socket z_top)
  * each gripper finger   -> box   (sizes from the OMX CAD model, see task.yaml `collision:`)
  * the wrist servo/link5 -> box above the fingers
  * the held peg          -> vertical cylinder hanging below the gripper centre (TCP)

Boxes are oriented with the arm: the fingers open/close along the TANGENTIAL
direction (perpendicular to the line from the robot base to the gripper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kinematics import J1_X


@dataclass
class Cylinder:
    name: str
    x: float
    y: float
    r: float
    z0: float
    z1: float


@dataclass
class Slab:
    """Axis-aligned box (used for the board)."""
    name: str
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float


@dataclass
class OBox:
    """Box rotated about z: half-sizes hx (radial) and hy (tangential)."""
    name: str
    cx: float
    cy: float
    yaw: float
    hx: float
    hy: float
    z0: float
    z1: float


@dataclass
class World:
    cylinders: list = field(default_factory=list)
    slabs: list = field(default_factory=list)
    table_z: float = 0.0


def _z_overlap(a0, a1, b0, b1, m):
    return a0 < b1 + m and a1 > b0 - m


def obox_circle_dist(b: OBox, x, y):
    """2-D distance from point (x, y) to the rotated rectangle (0 if inside)."""
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    dx, dy = x - b.cx, y - b.cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy          # point in box frame
    ex, ey = max(abs(lx) - b.hx, 0.0), max(abs(ly) - b.hy, 0.0)
    return math.hypot(ex, ey)


def obox_hits_cyl(b: OBox, cy: Cylinder, m):
    return _z_overlap(b.z0, b.z1, cy.z0, cy.z1, m) and obox_circle_dist(b, cy.x, cy.y) < cy.r + m


def obox_hits_slab(b: OBox, sl: Slab, m):
    if not _z_overlap(b.z0, b.z1, sl.z0, sl.z1, m):
        return False
    # conservative: compare the box's bounding circle with the slab rectangle
    rad = math.hypot(b.hx, b.hy)
    ex = max(sl.x0 - b.cx, 0.0, b.cx - sl.x1)
    ey = max(sl.y0 - b.cy, 0.0, b.cy - sl.y1)
    return math.hypot(ex, ey) < rad + m


def cyl_hits_cyl(a: Cylinder, b: Cylinder, m):
    return _z_overlap(a.z0, a.z1, b.z0, b.z1, m) and math.hypot(a.x - b.x, a.y - b.y) < a.r + b.r + m


def cyl_hits_slab(a: Cylinder, sl: Slab, m):
    if not _z_overlap(a.z0, a.z1, sl.z0, sl.z1, m):
        return False
    ex = max(sl.x0 - a.x, 0.0, a.x - sl.x1)
    ey = max(sl.y0 - a.y, 0.0, a.y - sl.y1)
    return math.hypot(ex, ey) < a.r + m


class GripperModel:
    """Gripper geometry (metres). Defaults measured from the ROBOTIS OMX CAD/MuJoCo model."""

    def __init__(self, ccfg: dict):
        self.tip_below = ccfg.get("finger_tip_below_tcp", 0.021)    # fingertip below TCP
        self.finger_top = ccfg.get("finger_top_above_tcp", 0.034)    # where the thin finger meets its slider
        self.finger_hx = ccfg.get("finger_half_depth", 0.0175)       # finger half-width (radial)
        self.finger_t = ccfg.get("finger_thickness", 0.012)          # finger thickness (tangential)
        # wide slider block each finger is mounted on (moves with the finger)
        self.mount_z0 = ccfg.get("mount_bottom_above_tcp", 0.034)
        self.mount_z1 = ccfg.get("mount_top_above_tcp", 0.064)
        self.mount_hx = ccfg.get("mount_half_depth", 0.029)
        self.mount_in = ccfg.get("mount_inner_offset", -0.014)       # relative to the finger's inner face
        self.mount_out = ccfg.get("mount_outer_offset", 0.026)
        self.wrist_z0 = ccfg.get("wrist_bottom_above_tcp", 0.058)    # link5 / servo box
        self.wrist_z1 = ccfg.get("wrist_top_above_tcp", 0.150)
        self.wrist_hx = ccfg.get("wrist_half_depth", 0.032)
        self.wrist_hy = ccfg.get("wrist_half_width", 0.064)
        self.margin = ccfg.get("safety_margin", 0.006)

    def boxes(self, x, y, z, gap):
        """Oriented boxes of the two fingers + wrist for gripper centre (x, y, z) and finger gap."""
        yaw = math.atan2(y, x - J1_X)                 # radial direction = gripper heading
        tx, ty = -math.sin(yaw), math.cos(yaw)        # tangential unit vector (finger axis)
        off = gap / 2 + self.finger_t / 2
        out = []
        moff = gap / 2 + (self.mount_in + self.mount_out) / 2
        mhy = (self.mount_out - self.mount_in) / 2
        for sgn, nm in ((1, "finger_L"), (-1, "finger_R")):
            out.append(OBox(nm, x + sgn * off * tx, y + sgn * off * ty, yaw,
                            self.finger_hx, self.finger_t / 2, z - self.tip_below, z + self.finger_top))
            out.append(OBox(nm + "_slider", x + sgn * moff * tx, y + sgn * moff * ty, yaw,
                            self.mount_hx, mhy, z + self.mount_z0, z + self.mount_z1))
        out.append(OBox("wrist", x, y, yaw, self.wrist_hx, self.wrist_hy, z + self.wrist_z0, z + self.wrist_z1))
        return out

    def outer_half_width(self, gap):
        return gap / 2 + self.finger_t


def held_peg(x, y, z, peg_h, grasp_depth, radius):
    """The carried peg: top is `grasp_depth` above the TCP."""
    return Cylinder("held_peg", x, y, radius, z - (peg_h - grasp_depth), z + grasp_depth)


def check_state(x, y, z, gap, world: World, model: GripperModel, holding=None, ignore_board=False):
    """Return None if collision-free, else a short text naming what is hit."""
    m = model.margin
    for b in model.boxes(x, y, z, gap):
        if b.z0 < world.table_z + 0.002:
            return f"{b.name} hits table"
        for c in world.cylinders:
            if obox_hits_cyl(b, c, m):
                return f"{b.name} hits {c.name}"
        if not ignore_board:
            for sl in world.slabs:
                if obox_hits_slab(b, sl, m * 0.5):
                    return f"{b.name} hits {sl.name}"
    if holding is not None:
        hp = held_peg(x, y, z, *holding)
        if hp.z0 < world.table_z + 0.002 and not ignore_board:
            return "held peg hits table"
        for c in world.cylinders:
            if cyl_hits_cyl(hp, c, m):
                return f"held peg hits {c.name}"
        if not ignore_board:
            for sl in world.slabs:
                if cyl_hits_slab(hp, sl, m * 0.5):
                    return f"held peg hits {sl.name}"
    return None


def build_world(cfg, other_pegs=(), placed=(), sockets=(), peg_h=None):
    """World from vision output.

    other_pegs : pegs still standing on the table (not the one being moved)
    placed     : [(peg_dict, socket_dict), ...] pegs already inserted
    sockets    : all detected sockets (used to approximate the board slab)
    """
    ccfg = cfg.get("collision", {})
    peg_h = peg_h or cfg.get("peg_height", 0.098)
    w = World(table_z=cfg["table_z"])
    for p in other_pegs:
        r = max(0.015, p.get("size", 0.05) / 2)            # size = circum-diameter from vision
        w.cylinders.append(Cylinder(f"{p.get('color', 'peg')} peg", p["x"], p["y"], r, cfg["table_z"], p["z_top"]))
    ins = cfg["place"]["insert_depth"]
    for p, s in placed:
        r = max(0.015, p.get("size", 0.05) / 2)
        w.cylinders.append(Cylinder(f"placed {p.get('color', 'peg')}", s["x"], s["y"], r,
                                    s["z_top"] - ins, s["z_top"] - ins + peg_h))
    if sockets and ccfg.get("model_board", True):
        xs = [s["x"] for s in sockets]
        ys = [s["y"] for s in sockets]
        mg = ccfg.get("board_margin", 0.035)
        top = max(sorted(s["z_top"] for s in sockets)[len(sockets) // 2], cfg["table_z"] + cfg.get("place", {}).get("board_height", 0.008))
        w.slabs.append(Slab("board", min(xs) - mg, max(xs) + mg, min(ys) - mg, max(ys) + mg, cfg["table_z"], top))
    return w
