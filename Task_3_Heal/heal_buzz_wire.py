#!/usr/bin/env python3
"""
heal_buzz_wire.py  --  Group 3, Project 3: HEAL plays the buzz-wire game (hard-coded path)

HEAL goes to the hoop resting at the LEFT end of the wire, closes the gripper on its handle,
swings the handle around the wire so it points toward the robot (out of the wire's plane, so it
never sweeps through the other bends), then moves the loop along the hard-coded wire centre-line,
turning it so the loop always stays perpendicular to the wire.

The loop is small (~3.5 mm free space around the wire), so measurements alone are not enough:
`touchup` drives the hoop along the wire, stops at every bend and lets you nudge it to the centre
with 1 mm steps; the corrections are saved and used by the next `plan`.

Needs heal_joint_replay.py and heal_ee_replay.py in the same folder (control loop + IK).

Sub-commands
    where     print joints, gripper-centre position and which way the tool axes point
    jog       interactive Cartesian jog + gripper open/close + `mark NAME`
              marks needed:  left, right (gripper centre ON the wire ends), grasp (fingers around
              the resting hoop handle); optional pregrasp
    plan      build the motion from WIRE_CM (+ saved corrections) and the marks
    touchup   grasp, flip, then go bend by bend and nudge the loop to the centre (saves corrections)
    run       execute: pre-grasp -> grasp -> close -> flip -> trace -> stop   (--sim to rehearse)

Frames: HEAL's base_link, z up. "table" = the plane HEAL is mounted on.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heal_joint_replay as J   # noqa: E402
import heal_ee_replay as E      # noqa: E402

info, die, DATA = J.info, J.die, J.DATA
MARKS = DATA / "buzz_marks.json"
CORR = DATA / "buzz_corrections.json"
GEOM = DATA / "buzz_geom.npz"

# =============================================================================
# ====================  EDIT HERE: hard-coded geometry  =======================
# =============================================================================
# Wire CENTRE-LINE corners (u, h) in CENTIMETRES, left end -> right end
#   u = horizontal distance along the board from the LEFT END of the wire (where it leaves the left post)
#   h = height of the wire centre above the TABLE that HEAL stands on
# Read off your annotated photo (scale from the 30 cm span and the 30 cm height): about +-3 mm.
# Corners of a vertical stretch have different u because the stretches lean slightly.
WIRE_CM = [
    (0.00, 28.88),   # 0  left end (leaves the left post)
    (2.20, 28.88),   # 1  turns down
    (3.24, 19.38),   # 2  bottom-left of 1st U   (hoop rests here in the photo)
    (5.89, 19.38),   # 3  bottom-right of 1st U
    (5.74, 28.37),   # 4  top-left of 1st hump
    (8.80, 28.37),   # 5  top-right of 1st hump
    (9.40, 19.61),   # 6  2nd U
    (12.38, 19.61),  # 7
    (11.51, 28.40),  # 8  2nd hump
    (14.92, 28.40),  # 9
    (16.52, 19.52),  # 10 3rd U
    (18.94, 19.52),  # 11
    (18.08, 29.16),  # 12 3rd hump
    (21.09, 29.16),  # 13
    (22.65, 20.65),  # 14 4th U
    (25.76, 20.65),  # 15
    (24.93, 29.83),  # 16 turns right at the top
    (26.97, 29.83),  # 17 right end (enters the right post)
]
START_AT = 2.5            # where the hoop's loop sits at the start, as a corner number:
                          #   2   = in the bottom-left bend (where it rests in your photo)
                          #   2.5 = middle of the 1st U's bottom (RECOMMENDED: gripper can come straight down)
BEND_RADIUS_CM = 0.6      # radius of the rounded bends (centre-line)

# Surfaces the gripper must stay above (heights above the table, MEASURE):
BOARD_TOP_CM = 17.7       # top of the wooden buzz-wire board
BOARD_FRONT_CM = 8.0      # how far the board extends from the wire plane toward HEAL
MAT_TOP_CM = 15.7         # top of the black mat/foam beyond the board's edge

# Gripper (see `where`): tip-frame axis that points from the wrist OUT through the fingers
APPROACH_AXIS = "+z"
TIP_TO_GRIP_CM = 0.0      # URDF tip link -> point between the finger pads, along APPROACH_AXIS.
                          # 0 if the tip link is already at the pads; ~gripper length if it is the flange.
FINGER_AXIS = "+y"        # tip-frame axis along which the fingers close
# Gripper size for the clearance check (CLOSED on the handle, measure with a ruler):
GRIPPER_BACK_CM = 15.0        # length from the pad centre back to the wrist
GRIPPER_HALF_WIDTH_CM = 4.0   # half of the width across the fingers/linkages (along FINGER_AXIS)
GRIPPER_HALF_THICK_CM = 2.0   # half of the thickness along the handle
PREGRASP_CM = 8.0         # back-off before closing in on the handle (along -APPROACH_AXIS)

# Gripper service (utils/gripper_commands/heal_dynamixel_gripper.py)
GRIPPER_SRV = "/gripper_controller/command"
GRIPPER_TYPE = "addverb_cobot_msgs/srv/Gripper"
GRIP_OPEN, GRIP_CLOSE, GRIP_FORCE = 1.0, 0.0, 100.0
# =============================================================================


def approach_vec():
    s = 1.0 if APPROACH_AXIS[0] == "+" else -1.0
    return s * np.eye(3)["xyz".index(APPROACH_AXIS[1])]


def grip_offset():
    return approach_vec() * TIP_TO_GRIP_CM / 100.0


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def rot(axis, ang):
    a = np.asarray(axis, float)
    return J.axis_angle_R(a / np.linalg.norm(a), ang)


class TcpChain:
    """URDF chain with a tool-centre-point offset r (in the tip frame)."""

    def __init__(self, chain, r_tip):
        self.c, self.r = chain, np.asarray(r_tip, float)
        self.lower, self.upper = chain.lower, chain.upper

    def fk(self, q):
        T = self.c.fk(q).copy()
        T[:3, 3] += T[:3, :3] @ self.r
        return T

    def jacobian(self, q):
        Jm = self.c.jacobian(q)
        r = self.c.fk(q)[:3, :3] @ self.r
        return np.vstack([Jm[:3] - skew(r) @ Jm[3:], Jm[3:]])


def wire_centerline(pts_cm, r_bend_cm, ds=0.001):
    """Rounded polyline (u,h) [m] resampled every ds; also the sample index nearest to each corner."""
    P = np.array(pts_cm, float) / 100.0
    r = r_bend_cm / 100.0
    out = [P[0]]
    for i in range(1, len(P) - 1):
        V = P[i]
        a, b = V - P[i - 1], P[i + 1] - V
        La, Lb = np.linalg.norm(a), np.linalg.norm(b)
        a, b = a / La, b / Lb
        turn = math.acos(max(-1.0, min(1.0, float(a @ b))))
        if turn < 1e-3:
            out.append(V)
            continue
        t = min(r * math.tan(turn / 2), 0.45 * La, 0.45 * Lb)
        P1, P2 = V - a * t, V + b * t
        for s in np.linspace(0, 1, 30):
            out.append((1 - s) ** 2 * P1 + 2 * s * (1 - s) * V + s ** 2 * P2)
    out.append(P[-1])
    Q = np.array(out)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(L) > 1e-9])
    Q, L = Q[keep], L[keep]
    S = np.append(np.arange(0, L[-1], ds), L[-1])
    UH = np.column_stack([np.interp(S, L, Q[:, 0]), np.interp(S, L, Q[:, 1])])
    vidx, lo = [], 0
    for V in P:                       # nearest sample, searched forward so the order is kept
        d = np.linalg.norm(UH[lo:] - V, axis=1)
        k = lo + int(np.argmin(d))
        vidx.append(k)
        lo = k
    return S, UH, np.array(vidx)


def time_parametrize(S, theta, v_max, w_max, a_max, dt):
    ds = np.diff(S)
    dth = np.abs(np.gradient(theta, S))
    v = np.minimum(v_max, w_max / np.maximum(dth, 1e-9))
    v[0] = v[-1] = 0.0
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_max * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_max * ds[i]))
    T = np.concatenate([[0], np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-9))])
    tt = np.arange(0, T[-1] + 1e-9, dt)
    return tt, np.interp(tt, T, S)


def smoothstep5(x):
    x = np.clip(x, 0, 1)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def retime(tt, Q, qd_max, dt):
    """Slow down LOCALLY wherever a joint would exceed qd_max; smooth the slow-down; resample."""
    if len(tt) < 3:
        return tt, Q
    dtn = np.diff(tt)
    need = np.max(np.abs(np.diff(Q, axis=0)), axis=1) / qd_max / dtn
    slow = np.maximum(1.0, need)
    w = max(1, int(0.4 / dt))
    pad = np.pad(slow, w, mode="edge")
    slow = np.array([pad[i:i + 2 * w + 1].max() for i in range(len(slow))])
    slow = J.gaussian_smooth(slow[:, None], w / 2)[:, 0]
    slow = np.maximum(slow, np.maximum(1.0, need))
    T = np.concatenate([[0], np.cumsum(dtn * slow)])
    t2 = np.arange(0, T[-1] + 1e-9, dt)
    return t2, np.column_stack([np.interp(t2, T, Q[:, j]) for j in range(Q.shape[1])])


# ---------------------------------------------------------------------------
# robot runner (ROS or simulation)
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, args, chain, sim_q0=None):
        self.args, self.chain, self.sim = args, chain, args.sim
        self.n = len(args.joints)
        if self.sim:
            self.q = np.array(sim_q0 if sim_q0 is not None else np.zeros(self.n), float)
            self.v = np.zeros(self.n)
            self.t_sim = 0.0
            return
        self.rclpy = J.ros_init()
        self.node = J.make_joint_io(args.joints, args.topic, args.cmd_topic, "group3_buzz")
        self.ex = J.start_spinning(self.rclpy, self.node)
        if self.node.wait_for_q(5.0) is None:
            self.close()
            die("No /joint_states. Are heal_server AND velocity_control.launch.py running?")
        if self.node.pub.get_subscription_count() == 0:
            info("WARNING: nobody subscribes to the velocity command topic -> the arm will not move.")

    def get_q(self):
        return self.q.copy() if self.sim else self.node.get_q()[0]

    def execute(self, traj, speed=1.0, min_approach=0.8, label="move", quiet=False):
        a = self.args
        peak = float(np.max(np.abs(traj.dq))) if len(traj.t) > 2 else 0.0
        if peak * speed > 0.8 * a.vmax and peak > 0:
            speed = 0.8 * a.vmax / peak
            info(f"[{label}] speed reduced to {speed:.2f} to respect vmax")
        f = J.Follower(traj, self.get_q(), speed=speed, kp=a.kp, vmax=a.vmax, amax=a.amax,
                       approach_speed=a.approach_speed, min_approach=min_approach,
                       abort_err=a.abort_err, rate=a.rate, auto_go=True)
        if self.sim:
            dt = 1.0 / a.rate
            rng = np.random.default_rng(0)
            while not f.finished():
                cmd = f.step(self.t_sim, self.q + rng.normal(0, 2e-4, self.n))
                self.v += (cmd - self.v) * min(1.0, dt / 0.05)
                self.q += self.v * dt
                self.t_sim += dt
        else:
            if self.node.timer is None:
                self.node.start_control(f, a.rate)
            else:
                self.node.follower = f
            last = time.monotonic()
            while not f.finished():
                time.sleep(0.02)
                if not quiet and time.monotonic() - last > 1.0 and f.log:
                    last = time.monotonic()
                    row = f.log[-1]
                    err = np.max(np.abs(row[2:2 + self.n] - row[2 + self.n:2 + 2 * self.n]))
                    print(f"   [{label}] {J.PHASE_NAMES[f.phase]:8s} max joint err {err:.4f} rad", flush=True)
        if f.phase == J.ABORT:
            raise RuntimeError(f"[{label}] ABORTED: {f.reason}")
        return f

    def move_to_q(self, q_goal, label="move", quiet=False):
        q_goal = np.asarray(q_goal, float)
        tr = J.SampledTrajectory(np.array([0.0, 0.01]), np.vstack([q_goal, q_goal]), np.zeros((2, self.n)))
        return self.execute(tr, min_approach=0.8, label=label, quiet=quiet)

    def gripper(self, pos, what):
        a = self.args
        if self.sim:
            info(f"[sim] gripper {what}")
            return
        if a.no_gripper or shutil.which("ros2") is None:
            input(f"   {what.upper()} the gripper yourself (other terminal), then press ENTER ...")
            return
        req = f"{{position: {pos}, grasp_force: {a.grip_force}}}"
        info(f"gripper {what}: ros2 service call {GRIPPER_SRV} {GRIPPER_TYPE} \"{req}\"")
        try:
            r = subprocess.run(["ros2", "service", "call", GRIPPER_SRV, GRIPPER_TYPE, req],
                               capture_output=True, text=True, timeout=25)
            print("   " + " | ".join((r.stdout + r.stderr).strip().splitlines()[-2:]))
        except subprocess.TimeoutExpired:
            info("gripper service call timed out")
        time.sleep(a.grip_wait)

    def close(self):
        if self.sim:
            return
        try:
            if self.node.follower is not None:
                self.node.follower.abort("shutdown")
            self.node.stop_and_zero()
        finally:
            J.ros_shutdown(self.rclpy, self.ex, self.node)


def solve_pose(tcp, q_seed, T_goal, what="target", tol_mm=1.0):
    q, ep, eo = E.ik_solve(tcp, np.array(q_seed, float), T_goal[:3, 3], T_goal[:3, :3], "pose",
                           0.02, 500, 1e-4, 1e-3, 0.1)
    if ep * 1000 > tol_mm or eo > 0.01:
        raise ValueError(f"IK for {what} failed (residual {ep * 1000:.1f} mm, {math.degrees(eo):.1f} deg) "
                         f"-> out of reach or joint limit")
    return q


def ik_path(tcp, q0, P, R, label, quiet=False):
    Q = np.zeros((len(P), len(q0)))
    ep = np.zeros(len(P))
    q = np.array(q0, float)
    for k in range(len(P)):
        q, ep[k], _ = E.ik_solve(tcp, q, P[k], R[k], "pose", 0.02, 300 if k == 0 else 60, 1e-4, 1e-3,
                                 0.2 if k == 0 else 0.05)
        Q[k] = q
    jump = np.max(np.abs(np.diff(Q, axis=0))) if len(Q) > 1 else 0
    if not quiet:
        info(f"[{label}] IK {len(P)} pts: residual max {ep.max() * 1000:.2f} mm, largest joint step {jump:.4f} rad")
    if ep.max() > 2e-3:
        i = int(np.argmax(ep))
        info(f"[{label}] WARNING: unreachable around sample {i} ({ep[i] * 1000:.1f} mm).")
    if jump > 0.05:
        info(f"[{label}] WARNING: joint jump {jump:.3f} rad between samples -> IK branch flip / singularity.")
    return Q, ep


def save_traj(name, tt, Q):
    dt = tt[1] - tt[0] if len(tt) > 1 else 0.01
    dQ = np.gradient(Q, dt, axis=0) if len(Q) > 2 else np.zeros_like(Q)
    dQ[0] = dQ[-1] = 0
    n = Q.shape[1]
    J.save_csv(DATA / f"buzz_{name}.csv", ["t"] + [f"q{i + 1}" for i in range(n)] + [f"dq{i + 1}" for i in range(n)],
               np.column_stack([tt, Q, dQ]))
    return np.max(np.abs(dQ), axis=0)


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
# where / jog
# ---------------------------------------------------------------------------
def cmd_where(args):
    chain = J.load_chain(args, required=True)
    r = Runner(args, chain, sim_q0=J.parse_vec(args.sim_start, len(args.joints)) if args.sim_start else None)
    try:
        q = r.get_q()
    finally:
        r.close()
    print_pose(chain, q)


def print_pose(chain, q):
    T = TcpChain(chain, grip_offset()).fk(q)
    print("   joints [rad]: " + ", ".join(f"{x:+.4f}" for x in q))
    print(f"   gripper centre [mm, base frame]: {np.round(T[:3, 3] * 1000, 1)}")
    for k, nm in enumerate("xyz"):
        print(f"   tool {nm}-axis points along base {np.round(T[:3, k], 3)}")
    last = max(i for i, c in enumerate(chain.chain) if c["type"] != "fixed")
    off = sum(np.linalg.norm(c["T0"][:3, 3]) for c in chain.chain[last + 1:])
    print(f"   (URDF: the tip link is {off * 1000:.0f} mm past the last joint '{chain.chain[last]['name']}';"
          f" APPROACH_AXIS={APPROACH_AXIS}, TIP_TO_GRIP_CM={TIP_TO_GRIP_CM})")


JOG_HELP = """
  x+ / x- / y+ / y- / z+ / z- [mm]   move gripper centre along BASE axes (default = step)
  tx+ ... tz- [mm]                   move along TOOL axes
  rx+ ... rz- [deg]                  rotate about BASE axes around the gripper centre
  rtx+ ... rtz- [deg]                rotate about TOOL axes around the gripper centre
  j1+ ... j6- [deg]                  single joint
  step <mm>   astep <deg>            set default steps (now {step} mm, {astep} deg)
  open / close                       gripper
  where                              print pose
  mark <name>                        save current pose: left | right | grasp | pregrasp
  q                                  quit
"""


def cmd_jog(args):
    chain = J.load_chain(args, required=True)
    tcp = TcpChain(chain, grip_offset())
    r = Runner(args, chain, sim_q0=J.parse_vec(args.sim_start, len(args.joints)) if args.sim_start else None)
    step, astep = 10.0, 10.0
    marks = load_json(MARKS)
    try:
        print(JOG_HELP.format(step=step, astep=astep))
        while True:
            try:
                s = input(f"jog [{step:g} mm / {astep:g} deg] > ").strip().lower()
            except EOFError:
                break
            if not s:
                continue
            if s in ("q", "quit", "exit"):
                break
            if s == "where":
                print_pose(chain, r.get_q())
                continue
            if s in ("open", "close"):
                r.gripper(GRIP_OPEN if s == "open" else GRIP_CLOSE, s)
                continue
            m = re.match(r"^(a?step)\s+([\d.]+)$", s)
            if m:
                if m.group(1) == "step":
                    step = float(m.group(2))
                else:
                    astep = float(m.group(2))
                continue
            m = re.match(r"^mark\s+(\w+)$", s)
            if m:
                q = r.get_q()
                marks[m.group(1)] = {"q": [float(x) for x in q], "T": tcp.fk(q).tolist()}
                DATA.mkdir(parents=True, exist_ok=True)
                MARKS.write_text(json.dumps(marks, indent=1))
                info(f"mark '{m.group(1)}' saved  (gripper centre {np.round(tcp.fk(q)[:3, 3] * 1000, 1)} mm)")
                continue
            q = r.get_q()
            T = tcp.fk(q)
            Tg = T.copy()
            m1 = re.match(r"^(t?)([xyz])([+-])\s*([\d.]+)?$", s)
            m2 = re.match(r"^r(t?)([xyz])([+-])\s*([\d.]+)?$", s)
            m3 = re.match(r"^j([1-6])([+-])\s*([\d.]+)?$", s)
            if m1:
                d = (float(m1.group(4)) if m1.group(4) else step) / 1000 * (1 if m1.group(3) == "+" else -1)
                e = np.eye(3)["xyz".index(m1.group(2))]
                Tg[:3, 3] += d * (T[:3, :3] @ e if m1.group(1) else e)
            elif m2:
                ang = math.radians(float(m2.group(4)) if m2.group(4) else astep) * (1 if m2.group(3) == "+" else -1)
                e = np.eye(3)["xyz".index(m2.group(2))]
                Tg[:3, :3] = rot(T[:3, :3] @ e if m2.group(1) else e, ang) @ T[:3, :3]
            elif m3:
                qg = q.copy()
                qg[int(m3.group(1)) - 1] += math.radians(float(m3.group(3)) if m3.group(3) else astep) * (1 if m3.group(2) == "+" else -1)
                r.move_to_q(qg, "jog")
                continue
            else:
                print("   ?" + JOG_HELP.format(step=step, astep=astep))
                continue
            try:
                qg = solve_pose(tcp, q, Tg, "jog target")
            except ValueError as ex:
                info(str(ex))
                continue
            if np.max(np.abs(qg - q)) > 0.6 and input(f"   big joint change ({np.max(np.abs(qg - q)):.2f} rad). y to go: ").strip() != "y":
                continue
            r.move_to_q(qg, "jog")
    except KeyboardInterrupt:
        print()
    except RuntimeError as ex:
        info(str(ex))
    finally:
        r.close()
        if r.sim:
            info("[sim] final joints " + ",".join(f"{x:.4f}" for x in r.q))


# ---------------------------------------------------------------------------
# geometry from marks + corrections
# ---------------------------------------------------------------------------
def start_point_uh(UH, vidx, P0):
    """Sample index of START_AT on the rounded centre-line."""
    k0 = int(math.floor(START_AT))
    f = START_AT - k0
    if f < 1e-6:
        return int(vidx[k0])
    target = (1 - f) * P0[k0] + f * P0[k0 + 1]
    lo, hi = int(vidx[k0]), int(vidx[k0 + 1])
    return lo + int(np.argmin(np.linalg.norm(UH[lo:hi + 1] - target, axis=1)))


def anchors(vidx, i0):
    """Touch-up points: the start, then every corner after it. Returns [(key, sample_index)]."""
    out = [("start", i0)]
    for k, j in enumerate(vidx):
        if j > i0 + 5:
            out.append((str(k), int(j)))
    return out


def build_geometry(args, chain, verbose=True):
    tcp_grip = TcpChain(chain, grip_offset())
    marks = load_json(MARKS)
    for k in ("left", "right", "grasp"):
        if k not in marks:
            die(f"mark '{k}' missing in {MARKS}. Teach it with `jog` (BUZZ_WIRE.md).")
    corr = load_json(CORR)
    z = np.array([0, 0, 1.0])
    W_left = tcp_grip.fk(np.array(marks["left"]["q"]))[:3, 3]
    W_right = tcp_grip.fk(np.array(marks["right"]["q"]))[:3, 3]
    v = W_right - W_left
    vh = v - (v @ z) * z
    u_hat = vh / np.linalg.norm(vh)
    n_hat = np.cross(z, u_hat)
    if n_hat @ (-W_left) < 0:            # point from the wire toward HEAL's base
        n_hat = -n_hat
    P0 = np.array(WIRE_CM, float) / 100
    span_geo = P0[-1, 0] - P0[0, 0]
    span_m = float(np.linalg.norm(vh))
    scale = span_m / span_geo if args.fit_span else 1.0
    S, UH, vidx = wire_centerline(WIRE_CM, BEND_RADIUS_CM)
    i0 = start_point_uh(UH, vidx, P0)
    base = W_left + np.outer((UH[:, 0] - P0[0, 0]) * scale, u_hat) + np.outer(UH[:, 1] - P0[0, 1], z)
    # touch-up corrections: offsets (du along board, dh up, dn toward HEAL) at anchors, linear in between
    anc = anchors(vidx, i0)
    idx = np.array([j for _, j in anc], float)
    C = np.array([corr.get(key, [0.0, 0.0, 0.0]) for key, _ in anc], float)
    smp = np.arange(len(S))
    off = np.column_stack([np.interp(smp, idx, C[:, c]) for c in range(3)])
    P_world = base + np.outer(off[:, 0], u_hat) + np.outer(off[:, 1], z) + np.outer(off[:, 2], n_hat)
    theta = np.unwrap(np.arctan2(np.gradient(UH[:, 1], S), np.gradient(UH[:, 0], S)))
    theta = J.gaussian_smooth(theta[:, None], args.theta_smooth / 0.001)[:, 0]
    table_z = W_left[2] - P0[0, 1]
    if verbose:
        info(f"left end {np.round(W_left * 1000, 1)} mm, right end {np.round(W_right * 1000, 1)} mm (base frame)")
        info(f"CHECK span: taught {span_m * 100:.1f} cm vs WIRE_CM {span_geo * 100:.1f} cm "
             f"(diff {(span_m - span_geo) * 1000:+.0f} mm{', stretched with --fit-span' if args.fit_span else ''})")
        dz_geo = (P0[-1, 1] - P0[0, 1])
        info(f"CHECK height right-left: taught {v @ z * 100:+.1f} cm vs WIRE_CM {dz_geo * 100:+.1f} cm "
             f"(diff {(v @ z - dz_geo) * 1000:+.0f} mm)")
        info(f"table is at z = {table_z * 1000:.0f} mm in base_link (from the left mark and WIRE_CM[0] height)")
        if corr:
            info(f"using {len(corr)} touch-up corrections from {CORR.name}")
    return dict(tcp_grip=tcp_grip, marks=marks, z=z, u_hat=u_hat, n_hat=n_hat, scale=scale, S=S, UH=UH,
                vidx=vidx, i0=i0, P_world=P_world, base=base, theta=theta, W_start0=base[i0],
                table_z=table_z, P0=P0, anc=anc, C=C)


def trace_rotations(theta_s, th0, u_hat, z, R_fe):
    """Loop turns about the handle axis (= wire-plane normal) by the change of the wire direction."""
    c = np.cross(u_hat, z)
    return [rot(c, th - th0) @ R_fe for th in theta_s]


def finger_vec():
    s = 1.0 if FINGER_AXIS[0] == "+" else -1.0
    return s * np.eye(3)["xyz".index(FINGER_AXIS[1])]


def gripper_clearance(chain, Q, g, theta_q, step=10):
    """Gripper = box swept from the pad centre back GRIPPER_BACK_CM (pads 1.5 x 1.2 cm half-size,
    full GRIPPER_HALF_WIDTH x GRIPPER_HALF_THICK from 5 cm back). Returns lowest clearance above
    board/mat [cm], closest to the wire plane [cm], 'gripper above the handle' score, worst index."""
    a, f = approach_vec(), finger_vec()
    tg = g["tcp_grip"]
    W = g["P_world"][g["i0"]]
    zz, nn = g["z"], g["n_hat"]
    board_z = g["table_z"] + BOARD_TOP_CM / 100
    mat_z = g["table_z"] + MAT_TOP_CM / 100
    low, front, ups, i_low = 1e9, 1e9, [], 0
    for i in range(0, len(Q), step):
        T = tg.fk(Q[i])
        aw, fw = T[:3, :3] @ a, T[:3, :3] @ f
        tw = np.cross(aw, fw)
        if abs(theta_q[i]) < 0.15:
            ups.append(-aw @ zz)
        for k in np.linspace(0, GRIPPER_BACK_CM / 100, 10):
            x = min(1.0, k / 0.05)
            hw = (1.5 + (GRIPPER_HALF_WIDTH_CM - 1.5) * x) / 100
            ht = (1.2 + (GRIPPER_HALF_THICK_CM - 1.2) * x) / 100
            p = T[:3, 3] - k * aw
            down = hw * abs(fw @ zz) + ht * abs(tw @ zz)      # box extent below its centre
            toward = hw * abs(fw @ nn) + ht * abs(tw @ nn)    # box extent toward the wire plane
            dist_n = (p - W) @ nn
            floor = board_z if dist_n - toward < BOARD_FRONT_CM / 100 else mat_z
            if p[2] - down - floor < low:
                low, i_low = p[2] - down - floor, i
            front = min(front, dist_n - toward)
    return low * 100, front * 100, (float(np.mean(ups)) if ups else float("nan")), i_low


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args):
    chain = J.load_chain(args, required=True)
    g = build_geometry(args, chain)
    tcp_grip, z, u_hat, n_hat = g["tcp_grip"], g["z"], g["u_hat"], g["n_hat"]
    S, P_world, theta, vidx = g["S"], g["P_world"], g["theta"], g["vidx"]
    i0 = g["i0"]
    W_start = P_world[i0]
    W_start0 = g["W_start0"]
    th0 = theta[i0]
    t0 = math.cos(th0) * u_hat + math.sin(th0) * z

    # ---- grasp & the loop-centre TCP (from the UNcorrected start point, fixed forever) ----------
    q_g = np.array(g["marks"]["grasp"]["q"])
    T_g = tcp_grip.fk(q_g)
    T_tip_g = chain.fk(q_g)
    R_g = T_tip_g[:3, :3]
    r_loop = R_g.T @ (W_start0 - T_tip_g[:3, 3])
    tcp_loop = TcpChain(chain, r_loop)
    handle = T_g[:3, 3] - W_start0
    info(f"loop centre -> gripper centre distance {np.linalg.norm(handle) * 100:.1f} cm "
         f"(should be roughly where you grip on the 9.7 cm handle + ~2 cm)")

    # ---- phase 1: pre-grasp -> grasp ------------------------------------------------------------
    if "pregrasp" in g["marks"]:
        T_pre = tcp_grip.fk(np.array(g["marks"]["pregrasp"]["q"]))
    else:
        T_pre = T_g.copy()
        T_pre[:3, 3] -= (R_g @ approach_vec()) * PREGRASP_CM / 100
    q_pre = solve_pose(tcp_grip, q_g, T_pre, "pre-grasp", tol_mm=2.0)
    L = np.linalg.norm(T_g[:3, 3] - T_pre[:3, 3])
    Tm = max(2.5, 1.875 * L / args.approach_v)
    tt = np.arange(0, Tm + 1e-9, args.dt)
    sg = smoothstep5(tt / Tm)
    qa, qb = E.R_to_q(T_pre[:3, :3]), E.R_to_q(R_g)
    P1 = T_pre[:3, 3] + np.outer(sg, T_g[:3, 3] - T_pre[:3, 3])
    R1 = [E.q_to_R(E.slerp(qa, qb, s)) for s in sg]
    Q1, _ = ik_path(tcp_grip, q_pre, P1, R1, "approach")
    tt, Q1 = retime(tt, Q1, args.qd_max, args.dt)
    pk1 = save_traj("approach", tt, Q1)

    # ---- phase 2: flip (swing the handle around the wire until it points at HEAL) ---------------
    # The handle lies in the loop's plane and, at rest, the wire passes through the loop, so the
    # loop axis is ~ the wire direction t0. Rotate (about the loop centre) so the handle points at
    # HEAL AND the loop is exactly square to the wire (this also fixes a hoop resting a bit crooked).
    h_tgt = n_hat
    h0 = handle / np.linalg.norm(handle)
    a0 = t0 - (t0 @ h0) * h0
    a0 /= np.linalg.norm(a0)
    F0 = np.column_stack([h0, a0, np.cross(h0, a0)])
    F1 = np.column_stack([h_tgt, t0, np.cross(h_tgt, t0)])
    Rc = F1 @ F0.T
    phi = math.acos(max(-1.0, min(1.0, (np.trace(Rc) - 1) / 2)))
    crooked = math.degrees(math.asin(min(1.0, abs(h0 @ t0))))
    info(f"hoop at rest: handle is {crooked:.0f} deg away from square to the wire (the flip corrects it)")
    qc = E.R_to_q(Rc)
    Tf = max(3.0, 1.875 * phi / args.rot_w)
    tt2 = np.arange(0, Tf + 1e-9, args.dt)
    s2 = smoothstep5(tt2 / Tf)
    R2 = [E.q_to_R(E.slerp(np.array([0, 0, 0, 1.0]), qc, s)) @ R_g for s in s2]
    P2 = W_start0 + np.outer(s2, W_start - W_start0)
    Q2, _ = ik_path(tcp_loop, Q1[-1], P2, R2, "flip")
    tt2, Q2 = retime(tt2, Q2, args.qd_max, args.dt)
    pk2 = save_traj("flip", tt2, Q2)
    R_fe = R2[-1]

    # ---- phase 3: trace ----------------------------------------------------------------------------
    S_tr = S[i0:] - S[i0]
    th_tr = theta[i0:]
    tt3, s3 = time_parametrize(S_tr, th_tr, args.v_trace, args.rot_w, args.a_trace, args.dt)
    P3 = np.column_stack([np.interp(s3, S_tr, P_world[i0:, k]) for k in range(3)])
    th3 = np.interp(s3, S_tr, th_tr)
    R3 = trace_rotations(th3, th0, u_hat, z, R_fe)
    Q3, ep3 = ik_path(tcp_loop, Q2[-1], P3, R3, "trace")
    low, front, up, i_low = gripper_clearance(chain, Q3, g, th3)          # during the trace only
    u_low = (P3[i_low] - g["base"][0]) @ u_hat * 100
    h_low = (P3[i_low][2] - g["table_z"]) * 100
    T3n = tt3[-1]
    tt3, Q3 = retime(tt3, Q3, args.qd_max, args.dt)
    if tt3[-1] > 1.05 * T3n:
        info(f"[trace] slowed down locally for joint speed: {T3n:.0f}s -> {tt3[-1]:.0f}s")
    pk3 = save_traj("trace", tt3, Q3)

    np.savez(GEOM, r_loop=r_loop, R_fe=R_fe, th0=th0, u_hat=u_hat, n_hat=n_hat, z=z,
             P_world=P_world, theta=theta, S=S, vidx=vidx, i0=i0, q_trace0=Q2[-1])

    # ---- checks + report ---------------------------------------------------------------------------
    allQ = np.vstack([Q1, Q2, Q3])
    lim_bad = np.any((allQ < chain.lower - 1e-6) | (allQ > chain.upper + 1e-6))
    lines = [
        "BUZZ-WIRE PLAN",
        f"start (loop centre) {np.round(W_start * 1000, 1)} mm; wire dir u {np.round(u_hat, 3)}; toward HEAL n {np.round(n_hat, 3)}",
        f"loop centre in tip frame [mm] {np.round(r_loop * 1000, 1)}",
        f"flip {math.degrees(phi):.0f} deg around the loop centre (handle ends pointing at HEAL)",
        f"durations: approach {tt[-1]:.1f}s  flip {tt2[-1]:.1f}s  trace {tt3[-1]:.1f}s",
        f"peak joint speed [rad/s]: approach {pk1.max():.3f}  flip {pk2.max():.3f}  trace {pk3.max():.3f}",
        f"trace IK residual max {ep3.max() * 1000:.2f} mm",
        f"TRACE: gripper on the UPPER side of the handle at the bottoms/tops: "
        f"{'yes' if up > 0.7 else 'NO -> the gripper body will swing DOWN at the bottoms (see BUZZ_WIRE.md: grasp)'}  ({up:+.2f})",
        f"TRACE: gripper (approx. shape) lowest clearance above board/mat: {low:.1f} cm   "
        f"(worst with the loop at u={u_low:.1f} cm, h={h_low:.1f} cm)   "
        f"{'OK' if low > 0.5 else 'TOO LOW -> see BUZZ_WIRE.md: clearance'}",
        f"TRACE: gripper closest to the wire plane: {front:.1f} cm   "
        f"{'OK' if front > 0.5 else 'TOO CLOSE -> grip the handle further from the loop'}",
        f"joint limits {'VIOLATED' if lim_bad else 'ok'}",
    ]
    rep = "\n".join(lines)
    print(rep)
    (DATA / "buzz_plan_report.txt").write_text(rep + "\n")
    flags = []
    if up <= 0.7:
        flags.append("gripper is below the handle on horizontal stretches")
    if low <= 0.5:
        flags.append(f"gripper may hit the board/mat ({low:.1f} cm) near u={u_low:.1f} cm, h={h_low:.1f} cm")
    if front <= 0.5:
        flags.append(f"gripper reaches the wire plane ({front:.1f} cm)")
    if lim_bad:
        flags.append("joint limits violated")
    if ep3.max() > 2e-3:
        flags.append("part of the path is unreachable")
    (DATA / "buzz_plan_flags.json").write_text(json.dumps(flags))
    if lim_bad:
        info("WARNING: the plan leaves the URDF joint limits. Do not run it.")
    plt = J.plt_or_none()
    if plt is not None:
        fig = plt.figure(figsize=(14, 6))
        a3 = fig.add_subplot(121, projection="3d")
        PW = P_world * 1000
        a3.plot(*PW.T, "k", lw=2, label="wire")
        a3.plot(*(P3 * 1000).T, "b--", lw=1, label="loop centre")
        for k in range(0, len(Q3), max(1, len(Q3) // 25)):
            a = tcp_loop.fk(Q3[k])[:3, 3] * 1000
            b = tcp_grip.fk(Q3[k])[:3, 3] * 1000
            a3.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "r-", lw=0.8)
        a3.scatter(0, 0, 0, c="g", s=60, label="HEAL base")
        a3.scatter(*(T_g[:3, 3] * 1000), c="m", label="grasp")
        J.equal_3d(a3, PW, np.zeros((1, 3)))
        a3.set_xlabel("x mm"); a3.set_ylabel("y mm"); a3.set_zlabel("z mm"); a3.legend(fontsize=8)
        a3.set_title("red = handle (loop -> gripper) during the trace")
        a2 = fig.add_subplot(122)
        UH = g["UH"]
        a2.plot(UH[:, 0] * 100, UH[:, 1] * 100, "k", lw=2, label="wire centre-line")
        a2.plot(g["P0"][:, 0] * 100, g["P0"][:, 1] * 100, "o", ms=3, color="gray", label="WIRE_CM corners")
        if np.any(g["C"]):
            off_u = (P_world - g["base"]) @ u_hat
            off_h = (P_world - g["base"]) @ z
            a2.plot((UH[:, 0] + off_u) * 100, (UH[:, 1] + off_h) * 100, "r-", lw=1, label="after touch-up")
        a2.axhline(BOARD_TOP_CM, color="peru", lw=3, label="board top")
        u_s = np.interp(s3, S_tr, UH[i0:, 0]) * 100
        h_s = np.interp(s3, S_tr, UH[i0:, 1]) * 100
        for k in range(0, len(s3), max(1, len(s3) // 70)):
            nn = np.array([-math.sin(th3[k]), math.cos(th3[k])]) * 0.73
            a2.plot([u_s[k] - nn[0], u_s[k] + nn[0]], [h_s[k] - nn[1], h_s[k] + nn[1]], "b-", lw=0.7)
        a2.plot(UH[i0, 0] * 100, UH[i0, 1] * 100, "g*", ms=12, label="start (hoop)")
        a2.set_aspect("equal"); a2.grid(alpha=0.3)
        a2.set_xlabel("u from left end [cm]"); a2.set_ylabel("height above table [cm]")
        a2.set_title("blue = loop (incircle diameter) - must cross the wire at 90 deg")
        a2.legend(fontsize=7, loc="lower right")
        J.finish_plot(plt, fig, DATA / "buzz_plan.png", args.show)
    info("next: run --sim, then touchup (recommended) or run")


# ---------------------------------------------------------------------------
# run / touchup
# ---------------------------------------------------------------------------
def load_phases(args, chain):
    n = len(args.joints)
    phases = []
    for nm in ("approach", "flip", "trace"):
        p = DATA / f"buzz_{nm}.csv"
        if not p.exists():
            die(f"{p} missing - run `plan` first.")
        tr = J.SampledTrajectory.from_csv(p, n)
        if np.any((tr.q < chain.lower - 1e-6) | (tr.q > chain.upper + 1e-6)):
            die("plan leaves the joint limits - re-plan.")
        phases.append(tr)
    return phases


def plan_flags_ok(args):
    p = DATA / "buzz_plan_flags.json"
    flags = json.loads(p.read_text()) if p.exists() else []
    if not flags:
        return True
    print("\nThe plan check reported:")
    for f in flags:
        print(f"   - {f}")
    if args.sim:
        info("[sim] continuing anyway")
        return True
    if input("Type yes to run anyway (e-stop in hand, watch those spots): ").strip().lower() != "yes":
        info("not running. Fix the plan (BUZZ_WIRE.md) and re-run `plan`.")
        return False
    return True


def grasp_and_flip(r, phases, args, step):
    q_start = phases[0].q[0]
    q_now = r.get_q()
    d = np.abs(q_start - q_now)
    info("current  : [" + ", ".join(f"{x:+.3f}" for x in q_now) + "]")
    info("pre-grasp: [" + ", ".join(f"{x:+.3f}" for x in q_start) + f"]   max distance {d.max():.2f} rad")
    if d.max() > 1.0:
        info("WARNING: > 1 rad to travel to the pre-grasp. Watch the whole move (straight line in JOINT space).")
    if not r.sim and input("\nType y + ENTER to start (gripper opens, arm goes to pre-grasp): ").strip() != "y":
        return False
    r.gripper(GRIP_OPEN, "open")
    r.move_to_q(q_start, "to pre-grasp")
    step("Move in around the hoop handle")
    r.execute(phases[0], speed=args.speed, min_approach=0.5, label="approach")
    step("Close the gripper on the handle")
    r.gripper(GRIP_CLOSE, "close")
    step("Swing the handle toward HEAL (loop turns around the wire)")
    r.execute(phases[1], speed=args.speed, min_approach=0.5, label="flip")
    return True


def make_step(r, args):
    def step(msg):
        if not args.auto and not r.sim:
            input(f"\n>>> {msg}  -- press ENTER (Ctrl+C = stop) ")
        else:
            info(f">>> {msg}")
    return step


def cmd_run(args):
    chain = J.load_chain(args, required=True)
    n = len(args.joints)
    phases = load_phases(args, chain)
    if not plan_flags_ok(args):
        return
    sim_q0 = J.parse_vec(args.sim_start, n) if args.sim_start else phases[0].q[0] + 0.15
    r = Runner(args, chain, sim_q0=sim_q0)
    step = make_step(r, args)
    f_trace = None
    try:
        if not grasp_and_flip(r, phases, args, step):
            info("cancelled")
            return
        step("TRACE the wire")
        f_trace = r.execute(phases[2], speed=args.speed, min_approach=0.5, label="trace")
        info("FINISHED at the right end. The hoop is still held.")
        if args.release:
            step("Open the gripper (release the hoop)")
            r.gripper(GRIP_OPEN, "open")
    except KeyboardInterrupt:
        print()
        info("Ctrl+C -> stopping")
    except RuntimeError as ex:
        info(str(ex))
    finally:
        r.close()
        info("zero velocity sent." if not r.sim else "[sim] done")
    if f_trace is not None:
        L = f_trace.log_array()
        Lt = L[L[:, 1] == J.TRACK]
        geo = np.load(GEOM)
        tcp_loop = TcpChain(chain, geo["r_loop"])
        Pd = np.array([tcp_loop.fk(q)[:3, 3] for q in Lt[::5, 2:2 + n]])
        Pa = np.array([tcp_loop.fk(q)[:3, 3] for q in Lt[::5, 2 + n:2 + 2 * n]])
        e = np.linalg.norm(Pd - Pa, axis=1) * 1000
        tag = "_sim" if r.sim else ""
        J.save_csv(DATA / f"buzz_run{tag}_log.csv", J.log_header(n), L)
        rep = (f"loop-centre deviation from the planned path while tracing (joint encoders + FK):\n"
               f"   RMS {np.sqrt(np.mean(e ** 2)):.2f} mm   max {e.max():.2f} mm   (free space around the wire ~3.5 mm)")
        print(rep)
        (DATA / f"buzz_run{tag}_report.txt").write_text(rep + "\n")


TOUCH_HELP = """
  u+ / u- [mm]   move the loop along the board (u+ = toward the right end)
  h+ / h- [mm]   up / down
  n+ / n- [mm]   toward HEAL / toward the wire plane's far side
  (default 1 mm)    ENTER or ok = centred: save this bend, go to the next
  q = stop here (bends you already confirmed stay saved; this one is not)
"""


def cmd_touchup(args):
    chain = J.load_chain(args, required=True)
    n = len(args.joints)
    phases = load_phases(args, chain)
    if not GEOM.exists():
        die("run `plan` first")
    geo = np.load(GEOM)
    tcp_loop = TcpChain(chain, geo["r_loop"])
    u_hat, n_hat, z = geo["u_hat"], geo["n_hat"], geo["z"]
    P_world, theta, S, vidx, i0 = geo["P_world"], geo["theta"], geo["S"], geo["vidx"], int(geo["i0"])
    corr = load_json(CORR)
    if not plan_flags_ok(args):
        return
    sim_q0 = J.parse_vec(args.sim_start, n) if args.sim_start else phases[0].q[0] + 0.15
    r = Runner(args, chain, sim_q0=sim_q0)
    step = make_step(r, args)
    off = np.zeros(3)          # running nudge [du, dh, dn] in metres, carried to the next bend
    cur = i0
    axes = {"u": u_hat, "h": z, "n": n_hat}
    sim_cmds = list(args.sim_cmds.split(";")) if args.sim_cmds else []

    def save():
        DATA.mkdir(parents=True, exist_ok=True)
        CORR.write_text(json.dumps(corr, indent=1))

    try:
        if not grasp_and_flip(r, phases, args, step):
            info("cancelled")
            return
        print(TOUCH_HELP)
        anc = anchors(vidx, i0)
        for num, (key, j) in enumerate(anc):
            name = "start" if key == "start" else f"corner {key}"
            if j > cur:
                seg = slice(cur, j + 1)
                Ss = S[seg] - S[cur]
                tt, ss = time_parametrize(Ss, theta[seg], args.touch_v, args.rot_w, args.a_trace, args.dt)
                Pw = np.column_stack([np.interp(ss, Ss, P_world[seg, c]) for c in range(3)])
                Pw = Pw + off[0] * u_hat + off[1] * z + off[2] * n_hat
                th = np.interp(ss, Ss, theta[seg])
                Rs = trace_rotations(th, float(geo["th0"]), u_hat, z, geo["R_fe"])
                Qs, _ = ik_path(tcp_loop, r.get_q(), Pw, Rs, f"to {name}", quiet=True)
                tt, Qs = retime(tt, Qs, args.qd_max, args.dt)
                dQ = np.gradient(Qs, args.dt, axis=0) if len(Qs) > 2 else np.zeros_like(Qs)
                dQ[0] = dQ[-1] = 0
                r.execute(J.SampledTrajectory(tt, Qs, dQ), min_approach=0.5, label=f"to {name}", quiet=True)
                cur = j
            while True:
                if r.sim:
                    s = sim_cmds.pop(0) if sim_cmds else ""
                else:
                    s = input(f"[{num + 1}/{len(anc)}] {name}: offset u {off[0] * 1000:+.0f} h {off[1] * 1000:+.0f} "
                              f"n {off[2] * 1000:+.0f} mm  > ").strip().lower()
                if s in ("", "ok"):
                    old = np.array(corr.get(key, [0.0, 0.0, 0.0]))
                    corr[key] = [float(x) for x in old + off]
                    save()
                    info(f"{name}: correction saved {np.round(np.array(corr[key]) * 1000, 1)} mm (u,h,n)")
                    break
                if s == "q":
                    save()
                    raise KeyboardInterrupt
                m = re.match(r"^([uhn])([+-])\s*([\d.]+)?$", s)
                if not m:
                    print(TOUCH_HELP)
                    continue
                d = (float(m.group(3)) if m.group(3) else 1.0) / 1000 * (1 if m.group(2) == "+" else -1)
                if abs(d) > 0.005:
                    print("   max 5 mm per nudge")
                    continue
                q = r.get_q()
                T = tcp_loop.fk(q)
                T[:3, 3] += d * axes[m.group(1)]
                try:
                    qg = solve_pose(tcp_loop, q, T, "nudge")
                except ValueError as ex:
                    info(str(ex))
                    continue
                r.move_to_q(qg, "nudge", quiet=True)
                off["uhn".index(m.group(1))] += d
        info("touch-up complete. Now:  plan  then  run")
    except KeyboardInterrupt:
        print()
        info("stopped; corrections so far are saved. Re-run `plan` to use them.")
    except RuntimeError as ex:
        info(str(ex))
    finally:
        r.close()


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--joints", type=lambda s: s.split(","), default=J.DEFAULT_JOINTS)
    common.add_argument("--topic", default=J.JS_TOPIC)
    common.add_argument("--cmd-topic", default=J.CMD_TOPIC)
    common.add_argument("--urdf", default=None)
    common.add_argument("--base-link", default="base_link")
    common.add_argument("--tip-link", default="end-effector")
    common.add_argument("--show", action="store_true")
    common.add_argument("--sim", action="store_true", help="simulated arm, no ROS")
    common.add_argument("--sim-start", default=None)
    common.add_argument("--no-gripper", action="store_true", help="operate the gripper yourself")
    common.add_argument("--grip-force", type=float, default=GRIP_FORCE)
    common.add_argument("--grip-wait", type=float, default=1.5)
    common.add_argument("--kp", type=float, default=2.0)
    common.add_argument("--vmax", type=float, default=0.4)
    common.add_argument("--amax", type=float, default=1.0)
    common.add_argument("--approach-speed", type=float, default=0.15, help="rad/s for joint-space moves")
    common.add_argument("--abort-err", type=float, default=0.15)
    common.add_argument("--rate", type=float, default=100.0)
    common.add_argument("--fit-span", action="store_true", help="stretch WIRE_CM to the taught left-right span")
    common.add_argument("--theta-smooth", type=float, default=0.004, help="smooth the loop rotation over this many m")
    common.add_argument("--rot-w", type=float, default=0.3, help="max loop rotation rate [rad/s]")
    common.add_argument("--a-trace", type=float, default=0.05)
    common.add_argument("--qd-max", type=float, default=0.25, help="joint speed limit used for re-timing [rad/s]")
    common.add_argument("--dt", type=float, default=0.01)
    common.add_argument("--speed", type=float, default=1.0)
    common.add_argument("--auto", action="store_true", help="no ENTER between phases")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("where", parents=[common]).set_defaults(fn=cmd_where)
    sp.add_parser("jog", parents=[common]).set_defaults(fn=cmd_jog)
    s = sp.add_parser("plan", parents=[common])
    s.add_argument("--v-trace", type=float, default=0.015, help="loop speed along the wire [m/s]")
    s.add_argument("--approach-v", type=float, default=0.02)
    s.set_defaults(fn=cmd_plan)
    s = sp.add_parser("touchup", parents=[common])
    s.add_argument("--touch-v", type=float, default=0.01, help="speed between corners [m/s]")
    s.add_argument("--sim-cmds", default="", help="(testing) ';'-separated answers used with --sim")
    s.set_defaults(fn=cmd_touchup)
    s = sp.add_parser("run", parents=[common])
    s.add_argument("--release", action="store_true", help="open the gripper at the end")
    s.set_defaults(fn=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
