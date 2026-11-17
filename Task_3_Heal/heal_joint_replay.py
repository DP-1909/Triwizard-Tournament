#!/usr/bin/env python3
"""
heal_joint_replay.py  --  Group 3, Project 3: teach-by-demonstration on HEAL (JOINT SPACE)

Use this script when /joint_states is available while the arm is in ./free_drive.

Pipeline (each step is a sub-command):

    check            see what /joint_states publishes (rate, names, values change?)
    save-urdf        save /robot_description to data/heal.urdf (needs heal_server + launch)
    record           record /joint_states while you move the arm by hand (free drive)
    teach-waypoints  fallback: capture single poses with Enter instead of a continuous recording
    process          trim + smooth + resample a recording  -> data/<name>_traj.csv
    from-waypoints   build a smooth trajectory through captured waypoints (quintic segments)
    goto             move smoothly to one joint configuration (e.g. --home)
    replay           go to the path start, wait for Enter, trace the path, stop
    analyze          tracking error (joint + end-effector) and plots from a replay log
    demo-data        make a fake recording so you can rehearse the pipeline without the robot

Everything that does not talk to the robot (process, from-waypoints, analyze, demo-data,
replay --sim) runs WITHOUT ROS, so you can rehearse on any laptop with numpy + matplotlib.

Control law used during replay (100 Hz by default):
    cmd = speed * dq_ref(t*speed) + kp * (q_ref(t*speed) - q_measured)
then clamped to +-vmax and rate-limited by amax (rad/s^2). Published as
std_msgs/Float64MultiArray (6 values, rad/s, order joint1..joint6) on /velocity_controller/commands.

Safety built in: aborts (publishes zeros) if tracking error > --abort-err, if /joint_states goes
stale, or on Ctrl+C. Always keep the e-stop in reach anyway.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_JOINTS = [f"joint{i}" for i in range(1, 7)]
JS_TOPIC = "/joint_states"
CMD_TOPIC = "/velocity_controller/commands"

APPROACH, HOLD, TRACK, SETTLE, DONE, ABORT = range(6)
PHASE_NAMES = ["APPROACH", "HOLD", "TRACK", "SETTLE", "DONE", "ABORT"]


# =============================================================================
# Small helpers
# =============================================================================
def die(msg: str, code: int = 1):
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)
    sys.exit(code)


def info(msg: str):
    print(f"[group3] {msg}", flush=True)


def parse_vec(text: str, n: int | None = None) -> np.ndarray:
    v = np.array([float(x) for x in text.replace(" ", "").split(",") if x != ""], dtype=float)
    if n is not None and len(v) != n:
        die(f"Expected {n} comma-separated numbers, got {len(v)}: '{text}'")
    return v


def save_csv(path: Path, header: list[str], rows: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_2d(rows), delimiter=",", header=",".join(header),
               comments="", fmt="%.7f")


def load_csv(path: Path):
    path = Path(path)
    if not path.exists():
        die(f"File not found: {path}")
    with open(path) as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return header, arr


def data_path(name: str, suffix: str) -> Path:
    return DATA / f"{name}{suffix}"


def gaussian_smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 with edge padding (keeps endpoints)."""
    x = np.asarray(x, dtype=float)
    if sigma_samples < 0.5 or len(x) < 3:
        return x.copy()
    r = int(math.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(xp[:, j], k, mode="valid")
    return out


def resample(t: np.ndarray, X: np.ndarray, dt: float):
    tt = np.arange(0.0, t[-1] + 1e-9, dt)
    Y = np.column_stack([np.interp(tt, t, X[:, j]) for j in range(X.shape[1])])
    return tt, Y


def clean_time(t: np.ndarray, X: np.ndarray):
    """Drop samples whose timestamp does not increase; start time at 0."""
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, X = t[keep], X[keep]
    return t - t[0], X


def trim_still(t, X, tol, pad_s):
    """Remove the stationary parts at the start and the end of a recording."""
    Xs = gaussian_smooth(X, 5.0)            # so sensor noise does not count as motion
    dev0 = np.max(np.abs(Xs - Xs[0]), axis=1)
    mv = np.nonzero(dev0 > tol)[0]
    if len(mv) == 0:
        die(f"No motion found (nothing moved more than {tol} from the first sample). "
            f"Did the values change while you moved the arm? Try a smaller --still-tol.")
    i0 = mv[0]
    dev1 = np.max(np.abs(Xs - Xs[-1]), axis=1)
    i1 = np.nonzero(dev1 > tol)[0][-1]
    i0 = int(np.searchsorted(t, t[i0] - pad_s))
    i1 = int(min(len(t) - 1, np.searchsorted(t, t[i1] + pad_s)))
    t2 = t[i0:i1 + 1] - t[i0]
    return t2, X[i0:i1 + 1], (t[i0], t[i1])


def plt_or_none():
    try:
        import matplotlib
        if "--show" not in sys.argv:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        info("matplotlib not available -> skipping plots")
        return None


def equal_3d(ax, *Ps):
    P = np.vstack(Ps)
    c = (P.max(axis=0) + P.min(axis=0)) / 2
    r = max(float(np.max(P.max(axis=0) - P.min(axis=0))) / 2, 1.0)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)


def finish_plot(plt, fig, path: Path, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    info(f"plot saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Trajectory math (no ROS)
# =============================================================================
class Quintic:
    """Same math as utils/quintic_planner.QuinticJointSpacePlanner (5th order per joint)."""

    def __init__(self, q0, q1, T, v0=None, v1=None, a0=None, a1=None):
        q0 = np.asarray(q0, float)
        q1 = np.asarray(q1, float)
        z = np.zeros_like(q0)
        v0 = z if v0 is None else np.asarray(v0, float)
        v1 = z if v1 is None else np.asarray(v1, float)
        a0 = z if a0 is None else np.asarray(a0, float)
        a1 = z if a1 is None else np.asarray(a1, float)
        T = float(max(T, 1e-3))
        self.T = T
        self.c0, self.c1, self.c2 = q0, v0, a0 / 2.0
        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.vstack([q1 - (q0 + v0 * T + a0 / 2 * T ** 2),
                       v1 - (v0 + a0 * T),
                       a1 - a0])
        self.c3, self.c4, self.c5 = np.linalg.solve(A, b)

    def evaluate(self, t):
        t = min(max(float(t), 0.0), self.T)
        c0, c1, c2, c3, c4, c5 = self.c0, self.c1, self.c2, self.c3, self.c4, self.c5
        q = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
        dq = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
        ddq = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3
        return q, dq, ddq


class SampledTrajectory:
    """Dense reference (t, q, dq) with linear interpolation."""

    def __init__(self, t, q, dq):
        self.t = np.asarray(t, float)
        self.q = np.asarray(q, float)
        self.dq = np.asarray(dq, float)
        self.T = float(self.t[-1])

    @classmethod
    def from_csv(cls, path, n_joints):
        header, arr = load_csv(path)
        if arr.shape[1] < 1 + 2 * n_joints:
            die(f"{path} does not look like a trajectory file (need t, {n_joints} q, {n_joints} dq).")
        return cls(arr[:, 0], arr[:, 1:1 + n_joints], arr[:, 1 + n_joints:1 + 2 * n_joints])

    def evaluate(self, tt):
        if len(self.t) == 1 or tt <= self.t[0]:
            return self.q[0].copy(), (self.dq[0].copy() if len(self.t) > 1 else np.zeros_like(self.q[0]))
        if tt >= self.T:
            return self.q[-1].copy(), np.zeros_like(self.q[-1])
        i = int(np.searchsorted(self.t, tt, side="right") - 1)
        i = min(max(i, 0), len(self.t) - 2)
        a = (tt - self.t[i]) / (self.t[i + 1] - self.t[i])
        return (1 - a) * self.q[i] + a * self.q[i + 1], (1 - a) * self.dq[i] + a * self.dq[i + 1]


class Follower:
    """Phase machine + control law. Pure python: used by the ROS node and the simulator."""

    def __init__(self, traj: SampledTrajectory, q_now, *, speed=0.5, kp=2.0, vmax=0.4, amax=1.0,
                 approach_speed=0.15, min_approach=3.0, abort_err=0.2, hold_tol=0.01,
                 settle_tol=0.005, rate=100.0, hold_vmax=0.15, auto_go=False):
        self.traj = traj
        self.speed, self.kp, self.vmax, self.amax = speed, kp, vmax, amax
        self.abort_err, self.hold_tol, self.settle_tol = abort_err, hold_tol, settle_tol
        self.hold_vmax = hold_vmax
        self.dt = 1.0 / rate
        self.n = len(q_now)
        self.q_start = traj.q[0].copy()
        self.q_end = traj.q[-1].copy()
        dist = float(np.max(np.abs(self.q_start - np.asarray(q_now))))
        self.T_approach = max(min_approach, 1.875 * dist / max(approach_speed, 1e-3))
        self.approach = Quintic(q_now, self.q_start, self.T_approach)
        self.phase = APPROACH
        self.t_phase = None
        self.t_ok = None
        self.prev_cmd = np.zeros(self.n)
        self.go_track = auto_go
        self.reason = ""
        self.log = []
        self.max_err_track = 0.0

    # -- phase helpers ---------------------------------------------------------
    def _enter(self, phase, now):
        self.phase = phase
        self.t_phase = now
        self.t_ok = None
        if phase in (DONE, ABORT):
            self.prev_cmd = np.zeros(self.n)

    def abort(self, reason):
        if self.phase not in (DONE, ABORT):
            self.reason = reason
            self.phase = ABORT
            self.prev_cmd = np.zeros(self.n)

    def finished(self):
        return self.phase in (DONE, ABORT)

    def _held(self, ok, now, need):
        if not ok:
            self.t_ok = None
            return False
        if self.t_ok is None:
            self.t_ok = now
        return now - self.t_ok >= need

    # -- main step -------------------------------------------------------------
    def step(self, now, q):
        q = np.asarray(q, float)
        if self.t_phase is None:
            self.t_phase = now
        if self.phase in (DONE, ABORT):
            return np.zeros(self.n)
        tau = now - self.t_phase
        vlim = self.vmax

        if self.phase == APPROACH:
            qd, dqd, _ = self.approach.evaluate(tau)
            if tau >= self.T_approach:
                self._enter(HOLD, now)
        elif self.phase == HOLD:
            qd, dqd = self.q_start, np.zeros(self.n)
            vlim = min(self.vmax, self.hold_vmax)
            err = np.max(np.abs(qd - q))
            tol = self.hold_tol if tau < 8.0 else 3 * self.hold_tol   # relax if it never settles
            if self.go_track and self._held(err < tol, now, 0.2):
                self._enter(TRACK, now)
        elif self.phase == TRACK:
            ts = tau * self.speed
            qd, dqd = self.traj.evaluate(ts)
            dqd = dqd * self.speed
            if ts >= self.traj.T:
                self._enter(SETTLE, now)
        else:  # SETTLE
            qd, dqd = self.q_end, np.zeros(self.n)
            vlim = min(self.vmax, self.hold_vmax)
            err = np.max(np.abs(qd - q))
            if self._held(err < self.settle_tol, now, 0.25) or tau > 4.0:
                self._enter(DONE, now)

        e = qd - q
        emax = float(np.max(np.abs(e)))
        if self.phase in (APPROACH, TRACK) and emax > self.abort_err:
            j = int(np.argmax(np.abs(e)))
            self.abort(f"tracking error {emax:.3f} rad on joint {j + 1} > --abort-err {self.abort_err}")
        if self.phase == TRACK:
            self.max_err_track = max(self.max_err_track, emax)

        if self.phase in (DONE, ABORT):
            cmd = np.zeros(self.n)
        else:
            cmd = dqd + self.kp * e
            cmd = np.clip(cmd, -vlim, vlim)
            dmax = self.amax * self.dt
            cmd = self.prev_cmd + np.clip(cmd - self.prev_cmd, -dmax, dmax)
        self.prev_cmd = cmd
        self.log.append(np.concatenate([[now, self.phase], qd, q, cmd]))
        return cmd

    def log_array(self):
        if not self.log:
            return np.zeros((0, 2 + 3 * self.n))
        a = np.array(self.log)
        a[:, 0] -= a[0, 0]
        return a


def log_header(n):
    return (["t", "phase"] + [f"qd{i + 1}" for i in range(n)] + [f"q{i + 1}" for i in range(n)]
            + [f"cmd{i + 1}" for i in range(n)])


# =============================================================================
# URDF kinematics in plain numpy (no PyKDL needed)
# =============================================================================
def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _vec3(text):
    return np.array([float(x) for x in text.split()], dtype=float)


def axis_angle_R(axis, ang):
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


class UrdfChain:
    """Serial chain base_link -> tip_link parsed from URDF text. q is given in `joint_names` order."""

    def __init__(self, urdf_text: str, base="base_link", tip="end-effector", joint_names=None):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(urdf_text)
        self.all_links = sorted(l.get("name") for l in root.findall("link"))
        by_child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = _vec3(o.get("xyz", "0 0 0")) if o is not None else np.zeros(3)
            rpy = _vec3(o.get("rpy", "0 0 0")) if o is not None else np.zeros(3)
            T0 = np.eye(4)
            T0[:3, :3] = _rpy_R(*rpy)
            T0[:3, 3] = xyz
            a = j.find("axis")
            axis = _vec3(a.get("xyz", "1 0 0")) if a is not None else np.array([1.0, 0, 0])
            axis = axis / (np.linalg.norm(axis) or 1.0)
            jtype = j.get("type")
            lo, hi = -np.inf, np.inf
            lim = j.find("limit")
            if lim is not None and jtype in ("revolute", "prismatic"):
                lo = float(lim.get("lower", -np.inf))
                hi = float(lim.get("upper", np.inf))
            child = j.find("child").get("link")
            by_child[child] = dict(name=j.get("name"), type=jtype, parent=j.find("parent").get("link"),
                                   T0=T0, axis=axis, lower=lo, upper=hi)
        chain, link = [], tip
        while link != base:
            if link not in by_child:
                die(f"URDF: cannot walk from tip '{tip}' to base '{base}' (stuck at '{link}').\n"
                    f"Links in this URDF: {', '.join(self.all_links)}\n"
                    f"Pass the right names with --base-link / --tip-link.")
            chain.append(by_child[link])
            link = by_child[link]["parent"]
        chain.reverse()
        self.chain = chain
        self.movable = [c for c in chain if c["type"] in ("revolute", "continuous", "prismatic")]
        chain_names = [c["name"] for c in self.movable]
        if joint_names is None:
            joint_names = chain_names
        missing = [n for n in chain_names if n not in joint_names]
        if missing or len(chain_names) != len(joint_names):
            die(f"URDF chain joints {chain_names} do not match --joints {joint_names}.")
        self.joint_names = list(joint_names)
        self.u2c = np.array([self.joint_names.index(n) for n in chain_names])   # chain i <- user u2c[i]
        self.lower = np.full(len(joint_names), -np.inf)
        self.upper = np.full(len(joint_names), np.inf)
        for i, c in enumerate(self.movable):
            self.lower[self.u2c[i]] = c["lower"]
            self.upper[self.u2c[i]] = c["upper"]

    def _fk_frames(self, q_user):
        qc = np.asarray(q_user, float)[self.u2c]
        T = np.eye(4)
        frames, k = [], 0
        for c in self.chain:
            T = T @ c["T0"]
            if c["type"] in ("revolute", "continuous", "prismatic"):
                z = T[:3, :3] @ c["axis"]
                frames.append((z, T[:3, 3].copy(), c["type"]))
                M = np.eye(4)
                if c["type"] == "prismatic":
                    M[:3, 3] = c["axis"] * qc[k]
                else:
                    M[:3, :3] = axis_angle_R(c["axis"], qc[k])
                T = T @ M
                k += 1
        return T, frames

    def fk(self, q_user) -> np.ndarray:
        return self._fk_frames(q_user)[0]

    def jacobian(self, q_user) -> np.ndarray:
        """Geometric Jacobian in the base frame, rows [v; w], columns in user joint order."""
        T, frames = self._fk_frames(q_user)
        pe = T[:3, 3]
        Jc = np.zeros((6, len(frames)))
        for i, (z, p, typ) in enumerate(frames):
            if typ == "prismatic":
                Jc[:3, i] = z
            else:
                Jc[:3, i] = np.cross(z, pe - p)
                Jc[3:, i] = z
        J = np.zeros_like(Jc)
        J[:, self.u2c] = Jc
        return J

    def fk_positions(self, Q) -> np.ndarray:
        return np.array([self.fk(q)[:3, 3] for q in Q])


def load_chain(args, required=False):
    urdf = Path(args.urdf) if getattr(args, "urdf", None) else DATA / "heal.urdf"
    if not urdf.exists():
        if required:
            die(f"URDF not found at {urdf}. Run `python3 heal_joint_replay.py save-urdf` while "
                f"heal_server + velocity_control.launch.py are running (or pass --urdf FILE).")
        return None
    return UrdfChain(urdf.read_text(), args.base_link, args.tip_link, args.joints)


# =============================================================================
# ROS 2 layer (imported lazily so offline commands work without ROS)
# =============================================================================
def ros_init():
    try:
        import rclpy
    except ImportError:
        die("rclpy not found. In this terminal run:\n"
            "   source /opt/ros/humble/setup.bash\n"
            "   source ~/Debojit_WS/addverb_ws/install/setup.bash   (or your workspace path)")
    try:   # keep Ctrl+C as a plain KeyboardInterrupt so we can still publish zeros afterwards
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except Exception:
        rclpy.init()
    return rclpy


def start_spinning(rclpy, node):
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    return ex


def ros_shutdown(rclpy, ex, node):
    try:
        ex.shutdown(timeout_sec=1.0)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


def make_joint_io(joints, js_topic, cmd_topic=None, node_name="group3_joint_io"):
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray

    class JointIO(Node):
        def __init__(self):
            super().__init__(node_name)
            # BEST_EFFORT receives from both best-effort and reliable publishers.
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=50)
            self.create_subscription(JointState, js_topic, self._cb, qos)
            self.pub = self.create_publisher(Float64MultiArray, cmd_topic, 10) if cmd_topic else None
            self.lock = threading.Lock()
            self.q = None
            self.q_time = None
            self.count = 0
            self.idx = None
            self.names = None
            self.recording = False
            self.samples = []
            self.raw_names = None
            self.warned = False
            self.follower = None
            self.timer = None
            self.stale_s = 0.25

        def _cb(self, msg):
            names = list(msg.name)
            if self.idx is None or names != self.names:
                self.names = names
                self.raw_names = names
                if names:
                    try:
                        self.idx = [names.index(j) for j in joints]
                    except ValueError:
                        if not self.warned:
                            self.get_logger().error(f"/joint_states names {names} do not contain {joints}. "
                                                    f"Use --joints to set the names.")
                            self.warned = True
                        self.idx = None
                        return
                else:
                    self.idx = list(range(len(joints)))
            if len(msg.position) <= max(self.idx):
                return
            q = np.array([msg.position[i] for i in self.idx], dtype=float)
            now = time.monotonic()
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self.lock:
                self.q, self.q_time = q, now
                self.count += 1
                if self.recording:
                    self.samples.append(np.concatenate([[st, now], q]))

        def get_q(self):
            with self.lock:
                return (None if self.q is None else self.q.copy()), self.q_time

        def wait_for_q(self, timeout=5.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                q, _ = self.get_q()
                if q is not None:
                    return q
                time.sleep(0.02)
            return None

        def publish(self, v):
            if self.pub is None:
                return
            m = Float64MultiArray()
            m.data = [float(x) for x in v]
            self.pub.publish(m)

        def start_control(self, follower, rate):
            self.follower = follower
            self.timer = self.create_timer(1.0 / rate, self._tick)

        def _tick(self):
            f = self.follower
            if f is None:
                return
            q, qt = self.get_q()
            now = time.monotonic()
            if q is None or now - qt > self.stale_s:
                f.abort(f"/joint_states stale for more than {self.stale_s}s")
                cmd = np.zeros(len(joints))
            else:
                cmd = f.step(now, q)
            self.publish(cmd)

        def stop_and_zero(self):
            if self.timer is not None:
                self.timer.cancel()
            for _ in range(15):
                self.publish(np.zeros(len(joints)))
                time.sleep(0.01)

    return JointIO()


def get_robot_description(timeout=8.0, topic="/robot_description") -> str:
    rclpy = ros_init()
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    node = Node("group3_urdf_grabber")
    box = {}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(String, topic, lambda m: box.setdefault("xml", m.data), qos)
    t0 = time.monotonic()
    while "xml" not in box and time.monotonic() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if "xml" not in box:
        die(f"No message on {topic} within {timeout}s. Is velocity_control.launch.py running "
            f"(it needs heal_server)?")
    return box["xml"]


# =============================================================================
# Sub-commands
# =============================================================================
def cmd_check(args):
    rclpy = ros_init()
    node = make_joint_io(args.joints, args.topic, None, "group3_check")
    ex = start_spinning(rclpy, node)
    try:
        time.sleep(1.0)
        topics = node.get_topic_names_and_types()
        info("Topics visible right now:")
        for name, types in sorted(topics):
            mark = "   <-- joint states" if "JointState" in ",".join(types) else ""
            print(f"    {name:45s} {', '.join(types)}{mark}")
        info(f"Listening to {args.topic} for {args.seconds:.0f}s. MOVE THE ARM NOW if you are in free drive ...")
        node.recording = True
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.5)
            q, _ = node.get_q()
            if q is not None:
                print("    q = [" + ", ".join(f"{x:+.4f}" for x in q) + "]", flush=True)
        node.recording = False
        S = np.array(node.samples) if node.samples else None
        print()
        if S is None or len(S) == 0:
            info(f"RESULT: NO joint states received on {args.topic}.")
            info("  -> check `ros2 topic list`, sourcing, ROS_DOMAIN_ID, and whether this robot mode publishes "
                 "joint states. If not, use heal_ee_replay.py or the teach-waypoints fallback (see RUNBOOK).")
            return
        rate = (len(S) - 1) / max(S[-1, 1] - S[0, 1], 1e-6)
        rng = S[:, 2:].max(axis=0) - S[:, 2:].min(axis=0)
        info(f"RESULT: {len(S)} messages, ~{rate:.0f} Hz, names in message: {node.raw_names}")
        info("  range of motion seen per joint (rad): " + ", ".join(f"{r:.4f}" for r in rng))
        stamps_ok = np.all(S[:, 0] > 0)
        info(f"  header stamps {'present' if stamps_ok else 'MISSING (receive time will be used)'}")
        if np.max(rng) < 1e-3:
            info("  Values did NOT change. If you moved the arm, this topic is frozen in this mode -> not usable.")
        else:
            info("  Values change -> joint recording will work. Use `record`.")
    finally:
        ros_shutdown(rclpy, ex, node)


def cmd_save_urdf(args):
    xml = get_robot_description()
    out = Path(args.urdf) if args.urdf else DATA / "heal.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    info(f"URDF saved to {out} ({len(xml)} chars)")
    ch = UrdfChain(xml, args.base_link, args.tip_link, args.joints)
    info(f"chain {args.base_link} -> {args.tip_link}: joints {[c['name'] for c in ch.movable]}")
    info(f"joint limits lower: {np.round(ch.lower, 3)}")
    info(f"joint limits upper: {np.round(ch.upper, 3)}")


def cmd_record(args):
    rclpy = ros_init()
    node = make_joint_io(args.joints, args.topic, None, "group3_recorder")
    ex = start_spinning(rclpy, node)
    out = data_path(args.name, "_raw.csv")
    try:
        if node.wait_for_q(5.0) is None:
            die(f"No messages on {args.topic}. Run `check` first.")
        input(f"\nReady. Put the arm at the START of the path, keep it still.\n"
              f"Press ENTER to start recording ...")
        node.samples = []
        node.recording = True
        t_start = time.monotonic()
        info("RECORDING. Hold still ~1 s, trace the path, hold still ~1 s at the end.")
        if args.duration > 0:
            info(f"Will stop automatically after {args.duration:.0f}s (Ctrl+C also stops and saves).")
            while time.monotonic() - t_start < args.duration:
                time.sleep(1.0)
                print(f"   {time.monotonic() - t_start:5.1f}s  samples={len(node.samples)}", flush=True)
        else:
            stop = threading.Event()
            threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
            info("Press ENTER (or Ctrl+C) to STOP recording.")
            while not stop.is_set():
                time.sleep(1.0)
                print(f"   {time.monotonic() - t_start:5.1f}s  samples={len(node.samples)}", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        node.recording = False
        S = np.array(node.samples) if node.samples else np.zeros((0, 2 + len(args.joints)))
        ros_shutdown(rclpy, ex, node)
    if len(S) < 10:
        die("Fewer than 10 samples recorded - nothing saved.")
    stamps = S[:, 0]
    use_stamps = np.all(stamps > 0) and np.all(np.diff(stamps) >= 0)
    t = stamps if use_stamps else S[:, 1]
    save_csv(out, ["t"] + list(args.joints), np.column_stack([t - t[0], S[:, 2:]]))
    info(f"saved {len(S)} samples ({t[-1] - t[0]:.1f}s, time source: {'header stamps' if use_stamps else 'receive time'}) -> {out}")
    info(f"next:  python3 {Path(__file__).name} process --name {args.name}")


def cmd_teach_waypoints(args):
    rclpy = ros_init()
    node = make_joint_io(args.joints, args.topic, None, "group3_waypoints")
    ex = start_spinning(rclpy, node)
    out = data_path(args.name, "_waypoints.csv")
    W = []
    if args.append and out.exists():
        W = list(load_csv(out)[1])
        info(f"appending to {len(W)} existing waypoints in {out}")
    try:
        if node.wait_for_q(5.0) is None:
            die(f"No messages on {args.topic}.")
        while True:
            s = input(f"[{len(W)} waypoints] ENTER = capture current pose, 'u' = undo last, 'q' = save & quit: ").strip().lower()
            if s == "q":
                break
            if s == "u":
                if W:
                    W.pop()
                continue
            node.samples, node.recording = [], True
            time.sleep(0.5)
            node.recording = False
            if not node.samples:
                info("no samples in the last 0.5 s - not captured")
                continue
            q = np.array(node.samples)[:, 2:].mean(axis=0)
            W.append(q)
            print("   captured q = [" + ", ".join(f"{x:+.4f}" for x in q) + "]")
            if args.single:
                break
    except KeyboardInterrupt:
        print()
    finally:
        ros_shutdown(rclpy, ex, node)
    if W:
        save_csv(out, list(args.joints), np.array(W))
        info(f"{len(W)} waypoints saved -> {out}")
        info(f"next:  python3 {Path(__file__).name} from-waypoints --name {args.name}")


def cmd_process(args):
    src = Path(args.input) if args.input else data_path(args.name, "_raw.csv")
    header, arr = load_csv(src)
    t, Q = clean_time(arr[:, 0], arr[:, 1:])
    n = Q.shape[1]
    gaps = np.diff(t)
    info(f"loaded {len(t)} samples, {t[-1]:.2f}s, median dt {np.median(gaps) * 1000:.1f} ms, "
         f"largest gap {gaps.max() * 1000:.0f} ms")
    if gaps.max() > 0.2:
        info("WARNING: gaps > 200 ms in the recording (dropped messages). Result is interpolated across them.")
    t_tr, Q_tr, (a, b) = trim_still(t, Q, args.still_tol, args.pad)
    info(f"trimmed still parts: kept {a:.2f}s .. {b:.2f}s  ({t_tr[-1]:.2f}s of motion)")
    tt, Qr = resample(t_tr, Q_tr, args.dt)
    Qs = gaussian_smooth(Qr, args.smooth / args.dt)
    dQ = np.gradient(Qs, args.dt, axis=0)
    dQ[0] = dQ[-1] = 0.0
    out = data_path(args.name, "_traj.csv")
    save_csv(out, ["t"] + [f"q{i + 1}" for i in range(n)] + [f"dq{i + 1}" for i in range(n)],
             np.column_stack([tt, Qs, dQ]))
    peak = np.max(np.abs(dQ), axis=0)
    info(f"trajectory saved -> {out}  ({len(tt)} points, {tt[-1]:.2f}s at recorded speed)")
    info("peak joint speed (rad/s) at speed 1.0: " + ", ".join(f"{p:.3f}" for p in peak))
    info(f"smoothing changed the path by at most {np.max(np.abs(Qs - Qr)):.4f} rad")
    for v in (0.3, 0.4, 0.5):
        info(f"  with --vmax {v}: fastest sensible --speed = {min(1.0, 0.8 * v / max(peak.max(), 1e-6)):.2f}")
    chain = load_chain(args)
    plt = plt_or_none()
    if plt is None:
        return
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for j in range(n):
        ax[0].plot(t_tr, Q_tr[:, j], ".", ms=2, alpha=0.4, color=f"C{j}")
        ax[0].plot(tt, Qs[:, j], "-", label=f"joint{j + 1}", color=f"C{j}")
        ax[1].plot(tt, dQ[:, j], "-", label=f"joint{j + 1}", color=f"C{j}")
    ax[0].set_ylabel("q [rad] (dots raw, lines smoothed)")
    ax[1].set_ylabel("dq [rad/s] at speed 1.0")
    ax[1].set_xlabel("t [s]")
    ax[0].legend(ncol=6, fontsize=8)
    ax[0].set_title(f"{args.name}: processed joint trajectory")
    finish_plot(plt, fig, data_path(args.name, "_traj.png"), args.show)
    if chain is not None:
        P = chain.fk_positions(Qs) * 1000
        fig = plt.figure(figsize=(7, 6))
        a3 = fig.add_subplot(111, projection="3d")
        a3.plot(P[:, 0], P[:, 1], P[:, 2])
        a3.scatter(*P[0], c="g", label="start")
        a3.scatter(*P[-1], c="r", label="end")
        a3.set_xlabel("x mm"); a3.set_ylabel("y mm"); a3.set_zlabel("z mm")
        equal_3d(a3, P)
        a3.legend()
        a3.set_title("end-effector path (FK of processed joints)")
        finish_plot(plt, fig, data_path(args.name, "_traj_ee.png"), args.show)


def cmd_from_waypoints(args):
    src = Path(args.input) if args.input else data_path(args.name, "_waypoints.csv")
    _, W = load_csv(src)
    if len(W) < 2:
        die("Need at least 2 waypoints.")
    n = W.shape[1]
    seg_T = []
    for i in range(len(W) - 1):
        d = float(np.max(np.abs(W[i + 1] - W[i])))
        seg_T.append(max(args.min_seg, 1.875 * d / args.joint_speed))
    tk = np.concatenate([[0.0], np.cumsum(seg_T)])
    V = np.zeros_like(W)
    if not args.stop_at_waypoints:      # Catmull-Rom style through-velocities
        for i in range(1, len(W) - 1):
            V[i] = (W[i + 1] - W[i - 1]) / (tk[i + 1] - tk[i - 1])
    segs = [Quintic(W[i], W[i + 1], seg_T[i], V[i], V[i + 1]) for i in range(len(W) - 1)]
    tt = np.arange(0.0, tk[-1] + 1e-9, args.dt)
    Q = np.zeros((len(tt), n))
    dQ = np.zeros((len(tt), n))
    for k, t in enumerate(tt):
        i = min(int(np.searchsorted(tk, t, side="right") - 1), len(segs) - 1)
        Q[k], dQ[k], _ = segs[i].evaluate(t - tk[i])
    out = data_path(args.name, "_traj.csv")
    save_csv(out, ["t"] + [f"q{i + 1}" for i in range(n)] + [f"dq{i + 1}" for i in range(n)],
             np.column_stack([tt, Q, dQ]))
    info(f"{len(W)} waypoints -> {len(segs)} quintic segments, {tk[-1]:.1f}s -> {out}")
    info("peak joint speed (rad/s): " + ", ".join(f"{p:.3f}" for p in np.max(np.abs(dQ), axis=0)))


# -- running the follower -------------------------------------------------------
def follower_from_args(traj, q_now, args, auto_go):
    return Follower(traj, q_now, speed=args.speed, kp=args.kp, vmax=args.vmax, amax=args.amax,
                    approach_speed=args.approach_speed, abort_err=args.abort_err,
                    rate=args.rate, auto_go=auto_go)


def preflight(traj, q_now, args):
    peak = float(np.max(np.abs(traj.dq))) if len(traj.t) > 1 else 0.0
    if peak * args.speed > 0.8 * args.vmax and peak > 0:
        new = 0.8 * args.vmax / peak
        info(f"speed {args.speed} would need {peak * args.speed:.2f} rad/s > 80% of vmax {args.vmax}; "
             f"reducing speed to {new:.2f}")
        args.speed = new
    d = np.abs(traj.q[0] - q_now)
    print()
    info("PRE-FLIGHT")
    print("    current q : [" + ", ".join(f"{x:+.3f}" for x in q_now) + "]")
    print("    start   q : [" + ", ".join(f"{x:+.3f}" for x in traj.q[0]) + "]")
    print("    distance  : [" + ", ".join(f"{x:.3f}" for x in d) + "]  rad")
    T_app = max(3.0, 1.875 * d.max() / args.approach_speed)
    print(f"    approach  : {T_app:.1f}s   path: {traj.T / max(args.speed, 1e-6):.1f}s at speed {args.speed:.2f}"
          f"   peak cmd ~{peak * args.speed:.2f} rad/s (vmax {args.vmax})")
    if d.max() > 1.0:
        info("WARNING: the start pose is more than 1 rad away on some joint. Watch the whole approach.")
    return d.max()


def run_sim(traj, args, q0, auto_go=True):
    f = follower_from_args(traj, q0, args, auto_go)
    rng = np.random.default_rng(0)
    q = np.array(q0, float)
    v = np.zeros_like(q)
    dt = 1.0 / args.rate
    now = 0.0
    while not f.finished() and now < 600:
        cmd = f.step(now, q + rng.normal(0, 2e-4, q.shape))
        v += (cmd - v) * min(1.0, dt / 0.05)          # 50 ms velocity lag
        q += v * dt
        now += dt
    return f


def run_follower(traj, args, out_log: Path, ask=True):
    n = len(args.joints)
    if args.sim:
        q0 = parse_vec(args.sim_start, n) if args.sim_start else np.zeros(n)
        preflight(traj, q0, args)
        f = run_sim(traj, args, q0)
    else:
        rclpy = ros_init()
        node = make_joint_io(args.joints, args.topic, args.cmd_topic, "group3_replay")
        ex = start_spinning(rclpy, node)
        f = None
        try:
            q_now = node.wait_for_q(5.0)
            if q_now is None:
                die(f"No /joint_states. Are heal_server AND velocity_control.launch.py running?")
            if node.pub.get_subscription_count() == 0:
                info(f"WARNING: nobody subscribes to {args.cmd_topic}. The velocity controller is not "
                     f"running/active -> the arm will not move. (ros2 control list_controllers)")
            dist = preflight(traj, q_now, args)
            if not args.yes:
                s = input("\nType y + ENTER to move to the start pose (anything else = cancel): ").strip().lower()
                if s != "y":
                    info("cancelled, nothing was sent.")
                    return None
            q_now = node.wait_for_q(1.0)
            f = follower_from_args(traj, q_now, args, auto_go=not ask)
            node.start_control(f, args.rate)
            info(f"moving to start ({f.T_approach:.1f}s) ...")
            prompted = False
            last_print = 0.0
            while not f.finished():
                if f.phase == HOLD and not f.go_track and not prompted:
                    prompted = True
                    input("\nAt the start pose. Press ENTER to trace the path (Ctrl+C = stop) ... ")
                    f.go_track = True
                    info("tracing ...")
                if time.monotonic() - last_print > 1.0 and f.log:
                    last_print = time.monotonic()
                    row = f.log[-1]
                    err = np.max(np.abs(row[2:2 + n] - row[2 + n:2 + 2 * n]))
                    print(f"   phase={PHASE_NAMES[f.phase]:8s} max joint err={err:.4f} rad", flush=True)
                time.sleep(0.02)
        except KeyboardInterrupt:
            print()
            if f is not None:
                f.abort("Ctrl+C")
            info("Ctrl+C -> stopping")
        finally:
            node.stop_and_zero()
            info("zero velocity sent.")
            ros_shutdown(rclpy, ex, node)
        if f is None:
            return None
    L = f.log_array()
    save_csv(out_log, log_header(n), L)
    if f.phase == ABORT:
        info(f"ABORTED: {f.reason}")
    else:
        info("DONE.")
    info(f"max joint tracking error while tracing: {f.max_err_track:.4f} rad")
    info(f"log -> {out_log}")
    return f


def cmd_replay(args):
    src = Path(args.input) if args.input else data_path(args.name, "_traj.csv")
    traj = SampledTrajectory.from_csv(src, len(args.joints))
    chain = load_chain(args)
    if chain is not None:
        bad = (traj.q < chain.lower - 1e-6) | (traj.q > chain.upper + 1e-6)
        if bad.any():
            j = np.nonzero(bad.any(axis=0))[0]
            die(f"Trajectory leaves the URDF joint limits on joints {[int(x) + 1 for x in j]}. Not replaying.")
    tag = "_sim" if args.sim else ""
    out = data_path(args.name, f"_replay{tag}_log.csv")
    f = run_follower(traj, args, out, ask=True)
    if f is not None:
        info(f"next:  python3 {Path(__file__).name} analyze --name {args.name}" + (" --sim" if args.sim else ""))


def cmd_goto(args):
    n = len(args.joints)
    if args.home:
        q = np.zeros(n)
    elif args.q:
        q = parse_vec(args.q, n)
    elif args.start_of:
        q = SampledTrajectory.from_csv(data_path(args.start_of, "_traj.csv"), n).q[0]
    elif args.delta:
        d = parse_vec(args.delta, n)
        if args.sim:
            base = parse_vec(args.sim_start, n) if args.sim_start else np.zeros(n)
        else:
            rclpy = ros_init()
            node = make_joint_io(args.joints, args.topic, None, "group3_goto_base")
            ex = start_spinning(rclpy, node)
            base = node.wait_for_q(5.0)
            ros_shutdown(rclpy, ex, node)
            if base is None:
                die("No /joint_states.")
        q = base + d
        info(f"relative move by {d} rad")
    else:
        die("give --q 'a,b,c,d,e,f', --delta 'a,b,c,d,e,f', --home or --start-of NAME")
    traj = SampledTrajectory(np.array([0.0, 0.01]), np.vstack([q, q]), np.zeros((2, n)))
    run_follower(traj, args, DATA / "goto_log.csv", ask=False)


def cmd_analyze(args):
    tag = "_sim" if args.sim else ""
    src = Path(args.input) if args.input else data_path(args.name, f"_replay{tag}_log.csv")
    _, L = load_csv(src)
    n = len(args.joints)
    t, ph = L[:, 0], L[:, 1].astype(int)
    QD, Q, C = L[:, 2:2 + n], L[:, 2 + n:2 + 2 * n], L[:, 2 + 2 * n:2 + 3 * n]
    m = ph == TRACK
    if not m.any():
        die("No TRACK phase in this log (aborted before tracing?).")
    E = QD[m] - Q[m]
    lines = [f"Replay analysis: {src.name}",
             f"tracing time {t[m][-1] - t[m][0]:.2f}s, {m.sum()} samples",
             "joint   RMS err [rad]   max err [rad]   max |cmd| [rad/s]"]
    for j in range(n):
        lines.append(f"  {j + 1}     {np.sqrt(np.mean(E[:, j] ** 2)):.5f}         {np.max(np.abs(E[:, j])):.5f}"
                     f"          {np.max(np.abs(C[m, j])):.3f}")
    chain = load_chain(args)
    if chain is not None:
        Pd = chain.fk_positions(QD[m]) * 1000
        Pa = chain.fk_positions(Q[m]) * 1000
        de = np.linalg.norm(Pd - Pa, axis=1)
        lines.append(f"end-effector error [mm]: RMS {np.sqrt(np.mean(de ** 2)):.2f}   max {de.max():.2f}   "
                     f"mean {de.mean():.2f}")
        settle = ph == SETTLE
        if settle.any():
            fin = np.linalg.norm(chain.fk(QD[settle][-1])[:3, 3] - chain.fk(Q[settle][-1])[:3, 3]) * 1000
            lines.append(f"final position error [mm]: {fin:.2f}")
    else:
        lines.append("(no URDF -> end-effector error skipped; run save-urdf)")
    report = "\n".join(lines)
    print(report)
    rp = src.with_name(src.stem + "_report.txt")
    rp.write_text(report + "\n")
    info(f"report -> {rp}")
    plt = plt_or_none()
    if plt is None:
        return
    fig, ax = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    for j in range(n):
        ax[j].plot(t, QD[:, j], "k--", lw=1, label="desired")
        ax[j].plot(t, Q[:, j], lw=1, label="actual")
        ax[j].set_ylabel(f"j{j + 1} [rad]")
        for k in range(len(PHASE_NAMES)):
            mk = ph == k
            if mk.any() and k == TRACK:
                ax[j].axvspan(t[mk][0], t[mk][-1], color="y", alpha=0.15)
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("t [s] (yellow = tracing)")
    finish_plot(plt, fig, src.with_name(src.stem + "_joints.png"), args.show)
    if chain is not None:
        fig = plt.figure(figsize=(12, 5))
        a3 = fig.add_subplot(121, projection="3d")
        a3.plot(Pd[:, 0], Pd[:, 1], Pd[:, 2], "k--", label="desired")
        a3.plot(Pa[:, 0], Pa[:, 1], Pa[:, 2], label="actual")
        a3.set_xlabel("x mm"); a3.set_ylabel("y mm"); a3.set_zlabel("z mm"); a3.legend()
        equal_3d(a3, Pd, Pa)
        a2 = fig.add_subplot(122)
        a2.plot(t[m], de, lw=0.7)
        a2.set_xlabel("t [s]"); a2.set_ylabel("EE position error [mm]"); a2.grid(alpha=0.3)
        finish_plot(plt, fig, src.with_name(src.stem + "_ee.png"), args.show)


def cmd_demo_data(args):
    rng = np.random.default_rng(1)
    n = len(args.joints)
    t = np.cumsum(rng.uniform(0.008, 0.012, int(14 / 0.01)))
    t -= t[0]
    q0 = np.array([0.0, -0.5, 1.0, 0.0, 0.6, 0.0])[:n]
    s = np.clip((t - 2.0) / 10.0, 0, 1)
    s = s * s * (3 - 2 * s)
    amp = np.array([0.5, 0.25, -0.3, 0.4, 0.3, 0.6])[:n]
    Q = q0 + np.outer(np.sin(np.pi * s), amp) + np.outer(s, amp * 0.3)
    Q += rng.normal(0, 4e-4, Q.shape)
    out = data_path(args.name, "_raw.csv")
    save_csv(out, ["t"] + list(args.joints), np.column_stack([t, Q]))
    info(f"fake recording ({t[-1]:.1f}s) -> {out}")


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--joints", type=lambda s: s.split(","), default=DEFAULT_JOINTS,
                        help="joint names in controller order (default joint1..joint6)")
    common.add_argument("--topic", default=JS_TOPIC, help="joint state topic")
    common.add_argument("--cmd-topic", default=CMD_TOPIC)
    common.add_argument("--urdf", default=None, help="URDF file (default data/heal.urdf)")
    common.add_argument("--base-link", default="base_link")
    common.add_argument("--tip-link", default="end-effector")
    common.add_argument("--show", action="store_true", help="also open plot windows")

    motion = argparse.ArgumentParser(add_help=False)
    motion.add_argument("--speed", type=float, default=0.5, help="time scale of the path (0.5 = half speed)")
    motion.add_argument("--kp", type=float, default=2.0, help="position feedback gain [1/s]")
    motion.add_argument("--vmax", type=float, default=0.4, help="joint speed clamp [rad/s]")
    motion.add_argument("--amax", type=float, default=1.0, help="joint accel limit [rad/s^2]")
    motion.add_argument("--approach-speed", type=float, default=0.15, help="peak speed when going to start [rad/s]")
    motion.add_argument("--abort-err", type=float, default=0.2, help="abort if any joint error exceeds this [rad]")
    motion.add_argument("--rate", type=float, default=100.0, help="control rate [Hz]")
    motion.add_argument("--yes", action="store_true", help="skip the first confirmation")
    motion.add_argument("--sim", action="store_true", help="no robot: simulate the arm (for rehearsal)")
    motion.add_argument("--sim-start", default=None, help="sim initial joints 'a,b,c,d,e,f' (default zeros)")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("check", parents=[common], help="inspect /joint_states")
    s.add_argument("--seconds", type=float, default=8.0)
    s.set_defaults(fn=cmd_check)

    s = sp.add_parser("save-urdf", parents=[common], help="save /robot_description to data/heal.urdf")
    s.set_defaults(fn=cmd_save_urdf)

    s = sp.add_parser("record", parents=[common], help="record joint states (free drive)")
    s.add_argument("--name", required=True)
    s.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until ENTER/Ctrl+C")
    s.set_defaults(fn=cmd_record)

    s = sp.add_parser("teach-waypoints", parents=[common], help="capture poses one by one with ENTER")
    s.add_argument("--name", required=True)
    s.add_argument("--append", action="store_true", help="add to an existing waypoint file")
    s.add_argument("--single", action="store_true", help="capture one pose and exit")
    s.set_defaults(fn=cmd_teach_waypoints)

    s = sp.add_parser("process", parents=[common], help="trim/smooth/resample a recording")
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None)
    s.add_argument("--still-tol", type=float, default=0.003, help="rad; motion below this counts as still")
    s.add_argument("--pad", type=float, default=0.3, help="s of stillness kept at both ends")
    s.add_argument("--smooth", type=float, default=0.10, help="Gaussian smoothing sigma [s]")
    s.add_argument("--dt", type=float, default=0.01)
    s.set_defaults(fn=cmd_process)

    s = sp.add_parser("from-waypoints", parents=[common], help="quintic trajectory through waypoints")
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None)
    s.add_argument("--joint-speed", type=float, default=0.25, help="approx peak joint speed per segment")
    s.add_argument("--min-seg", type=float, default=1.5, help="minimum segment duration [s]")
    s.add_argument("--stop-at-waypoints", action="store_true", help="come to rest at every waypoint")
    s.add_argument("--dt", type=float, default=0.01)
    s.set_defaults(fn=cmd_from_waypoints)

    s = sp.add_parser("replay", parents=[common, motion], help="trace a processed trajectory")
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None, help="trajectory csv (default data/<name>_traj.csv)")
    s.set_defaults(fn=cmd_replay)

    s = sp.add_parser("goto", parents=[common, motion], help="move to one joint configuration")
    s.add_argument("--q", default=None)
    s.add_argument("--delta", default=None, help="relative move from the current joints, e.g. '0,0,0,0,0,0.1'")
    s.add_argument("--home", action="store_true")
    s.add_argument("--start-of", default=None, help="go to the first point of data/<NAME>_traj.csv")
    s.set_defaults(fn=cmd_goto)

    s = sp.add_parser("analyze", parents=[common], help="tracking error report + plots")
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None)
    s.add_argument("--sim", action="store_true", help="analyze the --sim log")
    s.set_defaults(fn=cmd_analyze)

    s = sp.add_parser("demo-data", parents=[common], help="make a fake recording for rehearsal")
    s.add_argument("--name", default="demo")
    s.set_defaults(fn=cmd_demo_data)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
