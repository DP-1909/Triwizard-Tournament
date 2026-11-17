"""Planner without ROS: synthetic scene -> vision -> match -> collision-free plan."""
import math

import numpy as np

from peg_trial.planning.kinematics import fk
from peg_trial.planning.pick_place_planner import match_pegs_to_sockets, plan_order, plan_pick_place, yaw_error
from peg_trial.utils.synthetic import make_scene
from peg_trial.vision.scene import process_frame
from synthetic_cfg import CAMERAS, COLORS, task


def test_full_plan_from_synthetic_scene():
    cfg = task(table_z=-0.05)
    T = np.array(CAMERAS["cam1"], float)
    bgr, depth, K, _ = make_scene(T, table_z=cfg["table_z"], seed=3)
    scene = process_frame(bgr, depth, K, T, COLORS, cfg, "cam1")
    pairs = match_pegs_to_sockets(scene["pegs"], scene["sockets"])
    assert len(pairs) == 4
    order, errors = plan_order(pairs, cfg, scene["sockets"])
    assert order, errors
    peg, sock = order[0]
    others = [p for p, _ in pairs if p is not peg]
    steps, info = plan_pick_place(peg, sock, cfg, None, others, [], scene["sockets"])
    insert = next(s for s in steps if s["name"] == "insert")
    x, y, z, pitch = fk(insert["points"][-1][0])
    assert math.hypot(x - sock["x"], y - sock["y"]) < 1e-3          # ends over the socket
    assert abs(pitch - math.pi / 2) < 1e-6                           # gripper straight down
    times = [t for _, t in insert["points"]]
    assert times == sorted(times)


def test_plan_with_missing_socket():
    """Verify that if one socket (e.g. triangle) is undetected, planning still succeeds for other shapes."""
    cfg = task(table_z=-0.085)
    T = np.array(CAMERAS["cam1"], float)
    bgr, depth, K, _ = make_scene(T, table_z=cfg["table_z"], seed=3)
    scene = process_frame(bgr, depth, K, T, COLORS, cfg, "cam1")
    # Simulate missing triangle socket
    partial_sockets = [s for s in scene["sockets"] if s["shape"] != "triangle"]
    assert len(partial_sockets) == 3
    pairs = match_pegs_to_sockets(scene["pegs"], partial_sockets)
    assert len(pairs) == 3
    order, errors = plan_order(pairs, cfg, partial_sockets)
    assert len(order) == 3
    placed = []
    for peg, sock in order:
        others = [p for p, _ in pairs if p is not peg and p not in [pl[0] for pl in placed]]
        steps, info = plan_pick_place(peg, sock, cfg, None, obstacles=others, placed=placed, sockets=partial_sockets)
        assert steps
        placed.append((peg, sock))
    assert len(placed) == 3


def test_yaw_error_symmetry():
    peg = dict(x=0.2, y=0.0, yaw=0.0)
    sock = dict(x=0.2, y=0.0, yaw=math.pi / 2)
    assert abs(yaw_error(peg, sock, 4)) < 1e-9          # square rotated 90 deg looks identical
    assert abs(yaw_error(peg, dict(sock, yaw=0.3), 0)) == 0  # circle never misaligned
