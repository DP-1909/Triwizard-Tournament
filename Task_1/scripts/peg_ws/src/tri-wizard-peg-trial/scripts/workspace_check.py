#!/usr/bin/env python3
"""Where can the OMX actually pick and place? Run BEFORE you lay out the table.

    python3 scripts/workspace_check.py                 # uses config/task.yaml
    python3 scripts/workspace_check.py --board 0.02    # board thickness (m)

Prints the safe ring (distance from the robot base centre) for pegs and for
the socket board, given your table_z, peg height and grasp settings.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peg_trial.planning.pick_place_planner import PlanError, plan_pick_place  # noqa: E402
from peg_trial.utils.config import load  # noqa: E402


def ok_peg(r, cfg, sock):
    peg = dict(x=r, y=0.0, z_top=cfg["table_z"] + cfg["peg_height"], yaw=0, shape="circle")
    try:
        plan_pick_place(peg, sock, cfg)
        return True
    except (PlanError, Exception):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=float, default=0.02, help="board thickness (m)")
    a = ap.parse_args()
    cfg = load("task.yaml")
    rs = [i / 100 for i in range(6, 31)]
    board_top = cfg["table_z"] + a.board

    # pegs: test against a socket placed at a comfortable spot
    good_sock = None
    for r in rs:
        s = dict(x=r, y=0.0, z_top=board_top, yaw=0, shape="circle")
        if ok_peg(0.15, cfg, s):
            good_sock = s
            break
    print(f"table_z = {cfg['table_z']} m, peg height = {cfg['peg_height']} m, board thickness = {a.board} m\n")
    if good_sock is None:
        print("Nothing reachable! Raise the robot (more negative table_z) or lower pregrasp/hover offsets.")
        return
    peg_ok = [r for r in rs if ok_peg(r, cfg, good_sock)]
    peg_ref = dict(x=peg_ok[len(peg_ok) // 2], y=0.0, z_top=cfg["table_z"] + cfg["peg_height"], yaw=0,
                   shape="circle") if peg_ok else None
    sock_ok = []
    for r in rs:
        s = dict(x=r, y=0.0, z_top=board_top, yaw=0, shape="circle")
        try:
            plan_pick_place(peg_ref, s, cfg)
            sock_ok.append(r)
        except Exception:
            pass
    fmt = lambda v: f"{min(v) * 100:.0f} - {max(v) * 100:.0f} cm" if v else "NONE"
    print(f"Peg centres   : keep {fmt(peg_ok)} from the robot base centre")
    print(f"Socket centres: keep {fmt(sock_ok)} from the robot base centre")
    print("\nThe ring is the same in every direction around the base.")
    print("Practical minimum ~12 cm: closer and the peg/gripper hits the robot base.")
    print("Leave ~1-2 cm margin from the max — calibration is never perfect.")


if __name__ == "__main__":
    main()
