"""ROS 2 node: FULL AUTONOMY.

  1. Wait for the vision scene, cache the socket poses (board is empty at start).
  2. For each shape (order from task.yaml): find the peg, plan, execute.
  3. Verify: is the peg now sitting in the socket? If not -> retry (max_retries).
  4. Log every attempt to logs/trials.csv and the run to logs/runs.csv
     (scripts/evaluate_timing.py turns those into success-rate / timing tables).

E-STOP: publish `true` on /peg_trial/estop (teleop SPACE key, or
   ros2 topic pub --once /peg_trial/estop std_msgs/Bool "{data: true}")
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import uuid

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from peg_trial.control.arm_interface import ArmInterface
from peg_trial.planning.pick_place_planner import PlanError, match_pegs_to_sockets, plan_order, plan_pick_place
from peg_trial.utils.config import load
from peg_trial.utils.scene_buffer import SceneBuffer

# ---------------------------------------------------------------------------
# HARDCODED PRE-MEASURED TARGETS (from robot forward kinematics)
# Set USE_HARDCODED_TARGETS = False to revert back to camera-detected targets.
# ---------------------------------------------------------------------------
USE_HARDCODED_TARGETS = True
USE_HARDCODED_PEG_PICK = True
HARDCODED_PEG_PICK_XY = (0.090, 0.000)

SHAPE_TARGETS = {
    "circle": {
        "point": "POINT_2",
        "q": [-0.4876, 0.7063, -0.5663, 1.4308],  # rad (joint1..joint4)
        "x": 0.210,                               # m (gripper Cartesian X in base frame)
        "y": -0.105,                              # m (gripper Cartesian Y in base frame)
        "z": 0.015,                               # m (gripper Cartesian Z in base frame)
        "pitch_deg": 90.0,                        # deg (gripper pitch)
    },
    "hexagon": {
        "point": "POINT_3",
        "q": [-0.1405, 0.4919, -0.1891, 1.2679],  # rad (joint1..joint4)
        "x": 0.210,                               # m (gripper Cartesian X in base frame)
        "y": -0.028,                              # m (gripper Cartesian Y in base frame)
        "z": 0.015,                               # m (gripper Cartesian Z in base frame)
        "pitch_deg": 90.0,                        # deg (gripper pitch)
    },
    "square": {
        "point": "POINT_4",
        "q": [0.2235, 0.5165, -0.2307, 1.2850],  # rad (joint1..joint4)
        "x": 0.210,                               # m (gripper Cartesian X in base frame)
        "y": 0.045,                               # m (gripper Cartesian Y in base frame)
        "z": 0.015,                               # m (gripper Cartesian Z in base frame)
        "pitch_deg": 90.0,                        # deg (gripper pitch)
    },
    "triangle": {
        "point": "POINT_1",
        "q": [0.5449, 0.7900, -0.7211, 1.5019],  # rad (joint1..joint4)
        "x": 0.210,                               # m (gripper Cartesian X in base frame)
        "y": 0.120,                               # m (gripper Cartesian Y in base frame)
        "z": 0.015,                               # m (gripper Cartesian Z in base frame)
        "pitch_deg": 90.0,                        # deg (gripper pitch)
    },
}


class AutonomyNode(Node):
    def __init__(self):
        super().__init__("peg_autonomy")
        self.declare_parameter("task_file", "")
        self.declare_parameter("backend", "")          # override robot.backend
        self.declare_parameter("log_dir", os.environ.get("PEG_TRIAL_LOGS", "logs"))
        self.cfg = load("task.yaml", self.get_parameter("task_file").value or None)
        if self.get_parameter("backend").value:
            self.cfg["robot"]["backend"] = self.get_parameter("backend").value
        self.log_dir = self.get_parameter("log_dir").value
        os.makedirs(self.log_dir, exist_ok=True)

        self.arm = ArmInterface(self, self.cfg["robot"])
        self.scenes = SceneBuffer(20)
        self.create_subscription(String, "/peg_trial/scene", lambda m: self.scenes.add(json.loads(m.data)), 10)
        self.status_pub = self.create_publisher(String, "/peg_trial/status", 10)

        if USE_HARDCODED_TARGETS:
            from peg_trial.planning.kinematics import ik, Pose
            for sh_name, tgt in SHAPE_TARGETS.items():
                p = Pose(tgt["x"], tgt["y"], tgt["z"], math.radians(tgt["pitch_deg"]))
                try:
                    q_ik = ik(p)
                    diff = max(abs(a - b) for a, b in zip(tgt["q"], q_ik))
                    if diff > 0.02:
                        self.get_logger().warn(
                            f"Hardcoded target {tgt['point']} ({sh_name}): IK joint difference is {diff:.4f} rad (> 0.02 rad)"
                        )
                    else:
                        self.get_logger().info(
                            f"Verified hardcoded {tgt['point']} ({sh_name}): max IK joint diff = {diff:.4f} rad (<= 0.02 rad)"
                        )
                except Exception as e:
                    self.get_logger().warn(f"Hardcoded target {tgt['point']} ({sh_name}) IK check error: {e}")

    # ---------- helpers ----------
    def say(self, text):
        self.get_logger().info(text)
        self.status_pub.publish(String(data=text))

    def run_steps(self, steps) -> bool:
        for s in steps:
            if s["type"] == "move":
                ok = self.arm.move(s["points"])
            elif s["type"] == "gripper":
                ok = self.arm.gripper(s["cmd"], wait=0.3, gap=s.get("gap"))
            else:
                ok = self.arm._sleep(s["sec"])
            if not ok:
                self.get_logger().error(f"step '{s['name']}' aborted")
                return False
        return True

    def fresh_snapshot(self, expected_sockets=0):
        self.scenes.clear()
        return self.scenes.snapshot(min_frames=6, timeout=6.0, expected_sockets=expected_sockets)

    def verify(self, color, sock, peg_h) -> tuple[bool, str]:
        ex = self.cfg["execution"]
        snap = self.fresh_snapshot()
        if snap is None:
            return False, "no vision"
        peg = next((p for p in snap["pegs"] if p["color"] == color), None)
        if peg is None:
            return False, "peg not visible"
        dxy = math.hypot(peg["x"] - sock["x"], peg["y"] - sock["y"])
        z_expect = sock["z_top"] + peg_h - self.cfg["place"]["insert_depth"]
        dz = peg["z_top"] - z_expect
        ok = dxy < ex["verify_xy_tol"] and abs(dz) < ex["verify_z_tol"]
        return ok, f"dxy={dxy * 1000:.0f}mm dz={dz * 1000:.0f}mm"

    def log_row(self, fname, row: dict):
        path = os.path.join(self.log_dir, fname)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---------- main task ----------
    def run(self):
        if not self.arm.wait_for_joints(10.0):
            self.get_logger().error("no /joint_states — is the robot bringup running?")
            return
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        ex = self.cfg["execution"]
        self.say(f"RUN {run_id}: going home")
        self.arm.gripper("open")
        self.arm.move([(self.cfg["motion"]["home_joints"], 3.0)])

        snap = self.fresh_snapshot(expected_sockets=1)
        if snap is None or not snap["sockets"]:
            self.get_logger().error("no sockets detected — check board HSV / camera calibration")
            return
        sockets = snap["sockets"]
        # Filter out sockets beyond physical reach (>24.5cm)
        max_sock_r = self.cfg.get("vision", {}).get("max_socket_radius_m", 0.205)
        before_r = len(sockets)
        sockets = [s for s in sockets if math.hypot(s["x"] - 0.012, s["y"]) <= max_sock_r]
        if len(sockets) < before_r:
            self.get_logger().warn(
                f"Filtered {before_r - len(sockets)} socket(s) beyond r={max_sock_r*100:.1f}cm — "
                f"board is placed too far from robot! Please move board closer (ideal: 17 to 19 cm from robot base)")

        # De-duplicate by spatial proximity: sockets within 18mm are duplicate detections of the same physical hole.
        # Sockets separated by >= 18mm are distinct holes on the board!
        spatial_sockets = []
        for s in sorted(sockets, key=lambda s: -s.get("n", 1)):
            if not any(math.hypot(s["x"] - u["x"], s["y"] - u["y"]) < 0.018 for u in spatial_sockets):
                spatial_sockets.append(s)
        sockets = spatial_sockets
        if len(sockets) < before_r:
            dropped = before_r - len(sockets)
            self.get_logger().info(f"Deduplicated {dropped} redundant socket cluster(s)")

        # Disambiguate duplicate shape labels:
        # If two distinct physical sockets share the same shape label (e.g. circle vs hexagon),
        # assign the missing shape from ['circle', 'hexagon', 'square', 'triangle']
        assigned_shapes = [s["shape"] for s in sockets]
        missing_shapes = [sh for sh in ["circle", "hexagon", "square", "triangle"] if sh not in assigned_shapes]
        if missing_shapes:
            from collections import Counter
            counts = Counter(s["shape"] for s in sockets)
            for dupe_shape, cnt in counts.items():
                if cnt > 1:
                    dupes = [s for s in sockets if s["shape"] == dupe_shape]
                    # Sort along distance from base
                    dupes.sort(key=lambda s: math.hypot(s["x"], s["y"]))
                    for d_sock in dupes[1:]:
                        if missing_shapes:
                            new_sh = missing_shapes.pop(0)
                            self.get_logger().info(
                                f"Disambiguated socket at ({d_sock['x']:.3f},{d_sock['y']:.3f}) from '{dupe_shape}' to missing shape '{new_sh}'"
                            )
                            d_sock["shape"] = new_sh

        if USE_HARDCODED_TARGETS:
            for sh_key, tgt in SHAPE_TARGETS.items():
                existing = next((s for s in sockets if s["shape"].lower().strip() == sh_key), None)
                if existing is not None:
                    existing["x"] = tgt["x"]
                    existing["y"] = tgt["y"]
                    existing["z_top"] = tgt["z"]
                else:
                    sockets.append({
                        "shape": sh_key,
                        "x": tgt["x"],
                        "y": tgt["y"],
                        "z_top": tgt["z"],
                        "size": 0.04,
                        "conf": 1.0,
                    })

        detected_shapes = [s["shape"] for s in sockets]
        all_shapes = ["square", "hexagon", "circle", "triangle"]
        missing_shapes = [sh for sh in all_shapes if sh not in detected_shapes]
        if missing_shapes:
            self.say(f"WARNING: {len(missing_shapes)} socket(s) not detected: {missing_shapes}. "
                     f"Will execute pick-and-place for {len(sockets)} detected socket(s): {detected_shapes}")
        else:
            self.say(f"All 4 sockets detected: {detected_shapes}")
        self.say(f"pegs in view: {[(p['color'], p['shape']) for p in snap['pegs']]}")

        if ex.get("mode", "one_by_one") == "one_by_one":
            return self.run_one_by_one(run_id, sockets)

        t_run = time.time()
        done = 0
        order = ex["order"]
        pairs = match_pegs_to_sockets(snap["pegs"], sockets)
        pairs.sort(key=lambda ps: order.index(ps[0]["shape"]) if ps[0]["shape"] in order else 99)
        if len(pairs) < len(snap["pegs"]):
            unmatched = [p for p in snap["pegs"] if not any(p is ps[0] for ps in pairs)]
            self.get_logger().warn(
                f"Some pegs have no matching detected socket: {[(p['color'], p['shape']) for p in unmatched]}. "
                f"Proceeding with {len(pairs)} matched peg(s): {[(p['color'], p['shape']) for p, _ in pairs]}."
            )
        if not pairs:
            self.get_logger().error("No pegs match any of the detected sockets.")
            return

        # find an order in which every peg can be placed without collisions (look-ahead search)
        best, errs = plan_order(pairs, self.cfg, sockets, self.arm.joints(), preferred=order)
        if len(best) < len(pairs):
            self.get_logger().warn(f"only {len(best)}/{len(pairs)} pegs have a collision-free plan:")
            for col, e in errs.items():
                self.get_logger().warn(f"  {col}: {e}")
        self.say(f"placement order: {[p['color'] for p, _ in best]}")
        remaining = list(best) + [ps for ps in pairs if ps not in best]   # (peg, socket) still to do
        placed = []                      # (peg, socket) already inserted -> obstacles for later pegs
        latest = snap
        idx = 0
        while remaining and not self.arm.estopped:
            idx += 1
            # ---- pick the first peg (in preferred order) that has a COLLISION-FREE plan
            chosen, plan_errors = None, []
            for peg, sock in remaining:
                cur = next((p for p in latest["pegs"] if p["color"] == peg["color"]), peg)
                if USE_HARDCODED_PEG_PICK:
                    cur["x"] = HARDCODED_PEG_PICK_XY[0]
                    cur["y"] = HARDCODED_PEG_PICK_XY[1]
                sh_key = cur["shape"].lower().strip() if "shape" in cur else ""
                if USE_HARDCODED_TARGETS and sh_key in SHAPE_TARGETS:
                    tgt = SHAPE_TARGETS[sh_key]
                    self.say(f"Shape={sh_key} -> using hardcoded {tgt['point']}")
                    sock = dict(sock)
                    sock["x"] = tgt["x"]
                    sock["y"] = tgt["y"]
                    sock["z_top"] = tgt["z"]
                    sock["shape"] = sh_key
                done_colors = {p["color"] for p, _ in placed}
                others = [p for p in latest["pegs"] if p["color"] != cur["color"] and p["color"] not in done_colors]
                try:
                    steps, info = plan_pick_place(cur, sock, self.cfg, self.arm.joints(), obstacles=others,
                                                  placed=placed, sockets=sockets)
                    chosen = (peg, cur, sock, steps, info)
                    break
                except PlanError as e:
                    plan_errors.append(f"{peg['color']}: {e}")
            if chosen is None:
                for e in plan_errors:
                    self.get_logger().error(f"no safe plan — {e}")
                for peg, _ in remaining:
                    self.log_row("trials.csv", dict(run_id=run_id, index=idx, color=peg["color"], shape=peg["shape"],
                                                    attempts=0, success=0, duration_s=0, yaw_error_deg=0,
                                                    note="no collision-free plan"))
                break
            peg0, peg, sock, steps, info = chosen
            remaining.remove((peg0, sock))
            color = peg["color"]
            t0, attempts, ok, note = time.time(), 0, False, ""
            yaw_err = info["yaw_error_deg"]
            while attempts <= ex["max_retries"] and not ok and not self.arm.estopped:
                attempts += 1
                if attempts > 1:   # peg may have moved / fallen -> look again and re-plan
                    s2 = self.fresh_snapshot()
                    peg = next((p for p in (s2 or {"pegs": []})["pegs"] if p["color"] == color), None)
                    if peg is None:
                        note = "peg lost"
                        break
                    done_colors = {p["color"] for p, _ in placed}
                    others = [p for p in s2["pegs"] if p["color"] != color and p["color"] not in done_colors]
                    try:
                        steps, info = plan_pick_place(peg, sock, self.cfg, self.arm.joints(), obstacles=others,
                                                      placed=placed, sockets=sockets)
                    except PlanError as e:
                        note = str(e)
                        self.get_logger().error(note)
                        break
                self.say(f"[{idx}/{len(pairs)}] {color} {peg['shape']} -> socket ({sock['x']:.3f},{sock['y']:.3f}) "
                         f"attempt {attempts} | grip {info['grasp_depth'] * 1000:.0f} mm below top, "
                         f"carry z={info['carry_z']:.3f}, route={info['route']}")
                for w in info.get("warnings", []):
                    self.get_logger().warn(w)
                if abs(yaw_err) > self.cfg["execution"].get("yaw_warn_deg", 8):
                    self.get_logger().warn(
                        f"predicted socket misalignment {yaw_err:+.0f} deg after the squeeze grasp "
                        f"(OMX has no wrist roll, so this cannot be corrected by the arm; it depends only on "
                        f"how the socket is turned relative to the robot) — relying on socket clearance")
                if not self.run_steps(steps):
                    note = "aborted (e-stop)"
                    break
                if ex.get("verify", True) and self.cfg["robot"]["backend"] != "dry_run":
                    ok, note = self.verify(color, sock, info["peg_height"])
                    self.say(f"verify {color}: {'OK' if ok else 'FAIL'} {note}")
                else:
                    ok = True
            done += ok
            if ok:
                placed.append((peg, sock))
            self.log_row("trials.csv", dict(
                run_id=run_id, index=idx, color=color, shape=peg["shape"] if peg else "", attempts=attempts,
                success=int(ok), duration_s=round(time.time() - t0, 2), yaw_error_deg=round(yaw_err, 1), note=note))
            if remaining and self.cfg["robot"]["backend"] != "dry_run":
                latest = self.fresh_snapshot() or latest      # pegs may have been nudged: refresh positions

        total = time.time() - t_run
        self.log_row("runs.csv", dict(run_id=run_id, backend=self.cfg["robot"]["backend"], pegs=len(pairs),
                                      success=done, total_time_s=round(total, 2), estop=int(self.arm.estopped)))
        self.say(f"RUN {run_id} finished: {done}/{len(pairs)} pegs in {total:.1f}s")

    # ------------------------------------------------------------------ one peg at a time
    def wait_for_new_peg(self, placed_colors, sockets, bad_spots=(), timeout=600.0):
        """Wait (arm at home) until ONE new peg stands still in the workspace.

        'New' = colour not placed yet and not sitting in a socket. 'Still' = the same
        position (±5 mm) in two snapshots taken 1.5 s apart, so a hand putting the peg
        down is never chased.
        """
        t0, last = time.time(), None
        while time.time() - t0 < timeout and not self.arm.estopped:
            snap = self.fresh_snapshot()
            cands = []
            for p in (snap or {"pegs": []})["pegs"]:
                in_socket = any(math.hypot(p["x"] - s["x"], p["y"] - s["y"]) < 0.025 for s in sockets)
                tried = any(math.hypot(p["x"] - bx, p["y"] - by) < 0.01 for bx, by in bad_spots)
                if p["color"] not in placed_colors and not in_socket and not tried:
                    cands.append(p)
            if cands:
                # Prioritize candidates that have an available socket detected
                avail_shapes = {s["shape"] for s in sockets}
                matched = [p for p in cands if p["shape"] in avail_shapes]
                p = matched[0] if matched else cands[0]

                if last and last["color"] == p["color"] and math.hypot(p["x"] - last["x"], p["y"] - last["y"]) < 0.005:
                    return p, snap
                last = p
            else:
                last = None
            time.sleep(1.5)
        return None, None

    def run_one_by_one(self, run_id, sockets):
        """Task flow: a person puts ONE peg anywhere -> robot places it -> next peg ... until all sockets are full.
        Pegs already in their sockets stay there and are obstacles for every later move."""
        ex = self.cfg["execution"]
        placed = []                         # (peg, socket) already inserted
        t_run = time.time()
        n_total = len(sockets)
        bad_spots = []                      # positions where a peg could not be done safely
        while len(placed) < n_total and not self.arm.estopped:
            avail_shapes = [s["shape"] for s in sockets if id(s) not in {id(s_p) for _, s_p in placed}]
            self.say(f"[{len(placed)}/{n_total} placed] waiting for next peg (active sockets: {avail_shapes})...")
            peg, snap = self.wait_for_new_peg({p["color"] for p, _ in placed}, sockets, bad_spots)
            if peg is None:
                self.get_logger().error("no new peg appeared — stopping")
                break
            if USE_HARDCODED_PEG_PICK:
                peg["x"] = HARDCODED_PEG_PICK_XY[0]
                peg["y"] = HARDCODED_PEG_PICK_XY[1]
                peg["z_top"] = self.cfg["table_z"] + self.cfg["peg_height"]   # known 100 mm, ignore camera depth
                peg.setdefault("yaw", 0.0)
                self.say(f"Peg position locked to hardcoded ({peg['x']:.3f}, {peg['y']:.3f})")
            used = {id(s) for _, s in placed}
            sh_key = peg["shape"].lower().strip() if "shape" in peg else ""
            if USE_HARDCODED_TARGETS and sh_key in SHAPE_TARGETS:
                tgt = SHAPE_TARGETS[sh_key]
                self.say(f"Shape={sh_key} -> using hardcoded {tgt['point']}")
                matched = next((s for s in sockets if s["shape"].lower().strip() == sh_key and id(s) not in used), None)
                if matched is None:
                    matched = next((s for s in sockets if s["shape"].lower().strip() == sh_key), None)
                sock = dict(matched) if matched is not None else {"shape": sh_key, "size": 0.04}
                sock["x"] = tgt["x"]
                sock["y"] = tgt["y"]
                sock["z_top"] = tgt["z"]
                sock["shape"] = sh_key
                sock.setdefault("yaw", 0.0)
            else:
                sock = next((s for s in sockets if s["shape"] == peg["shape"] and id(s) not in used), None)
                if sock is None:
                    # 1) Check if an existing unused socket is recognized as peg['shape'] in latest snap
                    if snap and "sockets" in snap:
                        for s_new in snap["sockets"]:
                            for s_old in sockets:
                                if id(s_old) not in used and math.hypot(s_new["x"] - s_old["x"], s_new["y"] - s_old["y"]) < 0.020:
                                    if s_new["shape"] == peg["shape"]:
                                        self.say(f"Socket at ({s_old['x']:.3f},{s_old['y']:.3f}) updated to {peg['shape']}!")
                                        s_old["shape"] = peg["shape"]
                                        sock = s_old
                                        break
                            if sock is not None:
                                break
                    # 2) Check if a brand new socket appeared dynamically in the latest snapshot
                    if sock is None and snap and "sockets" in snap:
                        for s_new in snap["sockets"]:
                            # Skip candidate if it is near any already-placed peg
                            if any(math.hypot(s_new["x"] - p_p["x"], s_new["y"] - p_p["y"]) < 0.020 for p_p, _ in placed):
                                continue
                            if any(math.hypot(s_new["x"] - s_old["x"], s_new["y"] - s_old["y"]) < 0.018 for s_old in sockets):
                                continue
                            if s_new["shape"] == peg["shape"]:
                                self.say(f"Socket for {peg['shape']} detected dynamically! Adding to active sockets.")
                                sockets.append(s_new)
                                n_total = len(sockets)
                                sock = s_new
                                break
                    if sock is None:
                        avail = [s["shape"] for s in sockets if id(s) not in used]
                        self.get_logger().warn(
                            f"Skipping {peg['color']} {peg['shape']}: no detected socket for this shape! "
                            f"Detected sockets available: {avail}. Please ensure socket is in camera view."
                        )
                        time.sleep(2.0)
                        continue
            t0, attempts, ok, note, yaw_err = time.time(), 0, False, "", 0.0
            while attempts <= ex["max_retries"] and not ok and not self.arm.estopped:
                attempts += 1
                if attempts > 1:
                    s2 = self.fresh_snapshot()
                    peg = next((p for p in (s2 or {"pegs": []})["pegs"] if p["color"] == peg["color"]), None)
                    if peg is None:
                        note = "peg lost"
                        break
                sh_key = peg["shape"].lower().strip() if "shape" in peg else ""
                if USE_HARDCODED_TARGETS and USE_HARDCODED_PEG_PICK and sh_key in SHAPE_TARGETS:
                    tgt = SHAPE_TARGETS[sh_key]
                    home = self.cfg["motion"]["home_joints"]
                    q_pick_align = [0.0, home[1], home[2], home[3]]
                    q_pick_pre = [0.0, -0.6007, -0.0595, 1.9692]       # z=0.150m (5cm above 10cm peg)
                    q_pick_grasp = [0.0, -0.5434, 0.4665, 1.6477]      # z=0.082m (18mm below peg top, vertical 90 deg)
                    q_sock_target = tgt["q"]
                    theta_sock = q_sock_target[0]
                    q_sock_align = [theta_sock, home[1], home[2], home[3]]
                    q_sock_hover = tgt.get("q_hover", [theta_sock, q_sock_target[1] - 0.05, q_sock_target[2] - 0.15, q_sock_target[3] + 0.20])

                    steps = [
                        {"type": "move", "name": "align_pick", "points": [(q_pick_align, 1.2)]},
                        {"type": "move", "name": "to_pregrasp", "points": [(q_pick_pre, 1.5)]},
                        {"type": "gripper", "name": "open_pregrasp", "cmd": "open", "wait": 1.2},
                        {"type": "sleep", "name": "settle_pre", "sec": 0.3},
                        {"type": "move", "name": "approach", "points": [(q_pick_grasp, 1.2)]},
                        {"type": "sleep", "name": "settle", "sec": 0.4},
                        {"type": "gripper", "name": "close", "cmd": "close", "wait": 1.2},
                        {"type": "sleep", "name": "close_delay", "sec": 0.5},
                        {"type": "move", "name": "lift", "points": [(q_pick_pre, 1.2)]},
                        {"type": "move", "name": "retract_high", "points": [(q_pick_align, 1.5)]},
                        {"type": "move", "name": "transit_to_sock", "points": [(q_sock_align, 1.5)]},
                        {"type": "move", "name": "descend_hover", "points": [(q_sock_hover, 1.5)]},
                        {"type": "move", "name": "insert", "points": [(q_sock_target, 1.2)]},
                        {"type": "sleep", "name": "insert_settle", "sec": 0.4},
                        {"type": "gripper", "name": "release", "cmd": "open", "wait": 1.0},
                        {"type": "sleep", "name": "release_delay", "sec": 0.5},
                        {"type": "move", "name": "lift_clear", "points": [(q_sock_hover, 1.2)]},
                        {"type": "move", "name": "go_home", "points": [(home, 2.0)]},
                    ]
                    info = {"grasp_depth": 0.018, "peg_height": 0.100, "carry_z": 0.25,
                            "route": "high_transit", "yaw_error_deg": 0.0, "warnings": []}
                else:
                    try:
                        # obstacles = only the pegs already in sockets (plus board/table); the new peg is the target
                        steps, info = plan_pick_place(peg, sock, self.cfg, self.arm.joints(), obstacles=[],
                                                      placed=placed, sockets=sockets)
                    except PlanError as e:
                        note = str(e)
                        self.get_logger().error(f"{peg['color']}: no safe plan — {e}  (put the peg somewhere else)")
                        break
                yaw_err = info["yaw_error_deg"]
                self.say(f"{peg['color']} {peg['shape']} at ({peg['x']:.3f},{peg['y']:.3f}) -> socket "
                         f"({sock['x']:.3f},{sock['y']:.3f}) | grip {info['grasp_depth'] * 1000:.0f} mm below top, "
                         f"route={info['route']}, predicted yaw error {yaw_err:+.0f} deg")
                for w in info.get("warnings", []):
                    self.get_logger().warn(w)
                if not self.run_steps(steps):
                    note = "aborted (e-stop)"
                    break
                if ex.get("verify", True) and self.cfg["robot"]["backend"] != "dry_run":
                    ok, note = self.verify(peg["color"], sock, info["peg_height"])
                    self.say(f"verify {peg['color']}: {'OK' if ok else 'FAIL'} {note}")
                else:
                    ok = True
            if ok:
                placed.append((peg, sock))
            self.log_row("trials.csv", dict(
                run_id=run_id, index=len(placed) + (0 if ok else 1), color=peg["color"] if peg else "",
                shape=peg["shape"] if peg else "", attempts=attempts, success=int(ok),
                duration_s=round(time.time() - t0, 2), yaw_error_deg=round(yaw_err, 1), note=note))
            if self.arm.estopped:
                break
            if not ok and peg is not None:
                bad_spots.append((peg["x"], peg["y"]))      # wait until the person moves it
        total = time.time() - t_run
        self.log_row("runs.csv", dict(run_id=run_id, backend=self.cfg["robot"]["backend"], pegs=n_total,
                                      success=len(placed), total_time_s=round(total, 2),
                                      estop=int(self.arm.estopped)))
        self.say(f"RUN {run_id} finished: {len(placed)}/{n_total} pegs placed in {total:.1f}s")


def main():
    rclpy.init()
    node = AutonomyNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)   # ROS callbacks in background
    spin.start()
    try:
        node.run()                                          # task logic in main thread
    except KeyboardInterrupt:
        node.arm.hold()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()