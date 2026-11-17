"""Collision-aware planning: every planned pose is free, routes go over or around pegs."""
import math

import pytest

from peg_trial.planning.collision import GripperModel, build_world, check_state
from peg_trial.planning.grasp_geometry import held_width, with_known_size
from peg_trial.planning.kinematics import fk
from peg_trial.planning.pick_place_planner import PlanError, plan_order, plan_pick_place
from synthetic_cfg import task

NAMES = [("blue", "square"), ("yellow", "triangle"), ("orange", "hexagon"), ("green", "circle")]
PEGS = [(0.09, 0.06), (0.17, 0.06), (0.09, 0.14), (0.165, 0.135)]
SOCKS = [(0.09, -0.06), (0.17, -0.06), (0.09, -0.14), (0.17, -0.14)]


def layout(tz, peg_xy=PEGS, sock_xy=SOCKS):
    pegs = [dict(color=n[0], shape=n[1], x=x, y=y, z_top=tz + 0.098, yaw=0, size=0.045)
            for n, (x, y) in zip(NAMES, peg_xy)]
    socks = [dict(shape=n[1], x=x, y=y, z_top=tz + 0.012, yaw=0, size=0.05) for n, (x, y) in zip(NAMES, sock_xy)]
    return list(zip(pegs, socks)), socks


@pytest.mark.parametrize("tz", [-0.085, -0.05])
def test_all_pegs_collision_free(tz):
    cfg = task(table_z=tz, peg_height=0.098)
    pairs, socks = layout(tz)
    order, errors = plan_order(pairs, cfg, socks)
    assert len(order) == 4, errors
    model = GripperModel(cfg["collision"])
    done = []
    for peg, sock in order:
        others = [p for p, _ in pairs if p is not peg and p["color"] not in {d["color"] for d, _ in done}]
        steps, info = plan_pick_place(peg, sock, cfg, None, others, done, socks)
        # grip high on the peg: never lower than allowed, fingertips stay above the board when inserted
        assert cfg["grasp"]["grasp_depth"] <= info["grasp_depth"] <= cfg["grasp"]["max_grasp_depth"] + 1e-9
        kw = lambda p: with_known_size(p, cfg)     # noqa: E731  planner uses the measured peg sizes
        world = build_world(cfg, [kw(o) for o in others], [(kw(p), s) for p, s in done], socks, 0.098)
        holding, gap = False, info["open_gap"]
        for s in steps:
            if s["type"] == "gripper":
                holding = s["cmd"] == "close"
                gap = held_width(peg["shape"], kw(peg)["size"]) if holding else s.get("gap", gap)
                continue
            if s["type"] != "move" or s["name"] in ("to_ready", "home", "insert", "approach") \
                    or s["name"].startswith("retreat"):
                continue
            for q, _ in s["points"]:
                x, y, z, _p = fk(q)
                hold = (0.098, info["grasp_depth"], max(0.015, kw(peg)["size"] / 2)) if holding else None
                assert check_state(x, y, z, gap, world, model, hold) is None, (peg["color"], s["name"])
        done.append((peg, sock))


def test_robot_on_table_reports_reach_problem():
    cfg = task(table_z=0.0, peg_height=0.098)
    pairs, socks = layout(0.0)
    peg, sock = pairs[1]                      # r ~ 0.17: fingertips cannot get above a 98 mm peg
    with pytest.raises(PlanError) as e:
        plan_pick_place(peg, sock, cfg, None, [p for p, _ in pairs if p is not peg], [], socks)
    assert "raise the robot" in str(e.value) or "too far" in str(e.value)


def test_transit_goes_around_a_blocking_peg():
    cfg = task(table_z=-0.05, peg_height=0.098)        # 5 cm riser: cannot fly OVER 98 mm pegs
    peg = dict(color="blue", shape="square", x=0.15, y=0.10, z_top=-0.05 + 0.098, yaw=0, size=0.045)
    blocker = dict(color="red", shape="circle", x=0.15, y=0.0, z_top=-0.05 + 0.098, yaw=0, size=0.045)
    sock = dict(shape="square", x=0.15, y=-0.10, z_top=-0.05 + 0.012, yaw=0, size=0.05)
    steps, info = plan_pick_place(peg, sock, cfg, None, [blocker], [], [sock])
    assert info["route"] == "around"
    transit = [fk(q)[:2] for s in steps if s["name"].startswith("transit") for q, _ in s["points"]]
    assert min(math.hypot(x - 0.15, y) for x, y in transit) > 0.045    # keeps clear of the blocker
