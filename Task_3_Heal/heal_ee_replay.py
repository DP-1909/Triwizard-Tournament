#!/usr/bin/env python3
"""
heal_ee_replay.py  --  Group 3, Project 3: teach-by-demonstration on HEAL (END-EFFECTOR ONLY)

Use this script when free drive gives you only an end-effector position/pose (no usable
/joint_states). It turns the recorded end-effector path into joint angles with offline
damped-least-squares IK, then replays it in joint space with heal_joint_replay.py
(must be in the same folder).

    check        list topics, find pose-like ones, show what they contain (units, rate, changes?)
    record       record ANY pose-like topic (PoseStamped, Pose, PointStamped, Odometry,
                 TransformStamped, TFMessage, Float64MultiArray [x,y,z,...]) -> data/<name>_ee_raw.csv
    import-csv   use end-effector data from a file instead (vendor log, or waypoints typed by hand)
    make-shape   fallback: generate a line / circle / square / figure-8 starting at the current pose
    process      trim + smooth + resample -> data/<name>_ee_traj.csv
    ik           offline IK of every sample -> data/<name>_ik_traj.csv (joint trajectory)
    replay       = heal_joint_replay.py replay --name <name>_ik   (same options: --speed, --sim, ...)
    analyze      = heal_joint_replay.py analyze --name <name>_ik

IK details: for each sample, iterate  dq = J^T (J J^T + lambda^2 I)^-1 e,  e = [p_target - p, log(R_target R^T)],
starting from the previous sample's solution (keeps one elbow/wrist branch). The first solution
starts from the robot's REAL joints (read live from /joint_states, needs heal_server + launch),
at the end of the path that the arm is currently closest to (normally the END, because you
finished the recording there and did not move the arm).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heal_joint_replay as J  # noqa: E402  (same folder)

info, die, DATA = J.info, J.die, J.DATA


# =============================================================================
# Rotation helpers (quaternions are [x, y, z, w])
# =============================================================================
def q_to_R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_q(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rpy_to_q(r, p, y):
    return R_to_q(J._rpy_R(r, p, y))


def rot_log(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-7:
        return 0.5 * v
    if math.pi - ang < 1e-4:
        B = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / math.sqrt(max(B[k, k], 1e-12))
        return ang * axis / np.linalg.norm(axis)
    return ang / (2 * math.sin(ang)) * v


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def continuous_quats(Qt):
    Qt = np.array(Qt, float)
    for i in range(1, len(Qt)):
        if np.dot(Qt[i], Qt[i - 1]) < 0:
            Qt[i] = -Qt[i]
    return Qt


EE_HEADER = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]


# =============================================================================
# Pulling a pose out of any message
# =============================================================================
def from_array(d, layout="auto", angles="rad"):
    d = [float(x) for x in d]
    if layout == "auto":
        layout = {3: "xyz", 6: "xyzrpy", 7: "xyzquat"}.get(len(d), "xyz")
    p = d[:3]
    if layout == "xyz" or len(d) < 6:
        return p, None
    if layout == "xyzquat" and len(d) >= 7:
        return p, d[3:7]
    rpy = np.array(d[3:6])
    if angles == "deg":
        rpy = np.radians(rpy)
    return p, list(rpy_to_q(*rpy))


def extract_pose(msg, layout="auto", angles="rad", child_frame=None):
    if hasattr(msg, "data") and not isinstance(msg.data, str):
        return from_array(list(msg.data), layout, angles)
    if hasattr(msg, "transforms"):
        tfs = [t for t in msg.transforms if child_frame is None or t.child_frame_id == child_frame]
        if not tfs:
            return None
        msg = tfs[-1]
    o = msg
    for _ in range(6):
        if hasattr(o, "position") and hasattr(o, "orientation"):
            p, q = o.position, o.orientation
            return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        if hasattr(o, "translation") and hasattr(o, "rotation"):
            p, q = o.translation, o.rotation
            return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        if all(hasattr(o, a) for a in ("x", "y", "z")) and not hasattr(o, "w"):
            return [o.x, o.y, o.z], None
        for a in ("pose", "transform", "point", "vector"):
            if hasattr(o, a):
                o = getattr(o, a)
                break
        else:
            return None
    return None


POSE_HINTS = ("Pose", "Point", "Transform", "TFMessage", "Odometry", "MultiArray", "Vector3")


def make_pose_listener(topics_types, args):
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rosidl_runtime_py.utilities import get_message

    class PoseListener(Node):
        def __init__(self):
            super().__init__("group3_pose_listener")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
            self.buf = {t: [] for t, _ in topics_types}
            self.recording = True
            self.bad = {}
            for topic, typ in topics_types:
                cls = get_message(typ)
                self.create_subscription(cls, topic, lambda m, tp=topic: self._cb(tp, m), qos)

        def _cb(self, topic, msg):
            if not self.recording:
                return
            r = extract_pose(msg, args.layout, args.angles, args.child_frame)
            if r is None:
                self.bad[topic] = self.bad.get(topic, 0) + 1
                return
            p, q = r
            st = 0.0
            if hasattr(msg, "header"):
                st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            q = [np.nan] * 4 if q is None else q
            self.buf[topic].append([st, time.monotonic()] + [v * args.scale for v in p] + list(q))

    return PoseListener()


def discover(rclpy, seconds=1.5):
    from rclpy.node import Node
    n = Node("group3_discover")
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        rclpy.spin_once(n, timeout_sec=0.1)
    tt = n.get_topic_names_and_types()
    n.destroy_node()
    return dict((name, types[0]) for name, types in tt if types)


# =============================================================================
# Sub-commands
# =============================================================================
def cmd_check(args):
    rclpy = J.ros_init()
    topics = discover(rclpy)
    info("Topics visible right now:")
    for name in sorted(topics):
        mark = "   <-- pose-like" if any(h in topics[name] for h in POSE_HINTS) else ""
        mark = "   <-- JOINT STATES (use heal_joint_replay.py!)" if "JointState" in topics[name] else mark
        print(f"    {name:45s} {topics[name]}{mark}")
    if args.topic:
        if args.topic not in topics:
            die(f"{args.topic} is not published right now.")
        cand = [(args.topic, topics[args.topic])]
    else:
        cand = [(n, t) for n, t in topics.items() if any(h in t for h in POSE_HINTS)
                and n not in ("/robot_description",) and "parameter" not in n]
    if not cand:
        info("No pose-like topics. Free drive may not publish anything to ROS -> see RUNBOOK fallback.")
        rclpy.shutdown()
        return
    node = make_pose_listener(cand, args)
    ex = J.start_spinning(rclpy, node)
    info(f"Listening to {len(cand)} topic(s) for {args.seconds:.0f}s. MOVE THE ARM NOW ...")
    time.sleep(args.seconds)
    node.recording = False
    J.ros_shutdown(rclpy, ex, node)
    print()
    for topic, typ in cand:
        S = np.array(node.buf[topic]) if node.buf[topic] else None
        if S is None:
            extra = f" ({node.bad[topic]} msgs could not be read as a pose)" if topic in node.bad else ""
            info(f"{topic}: no usable messages{extra}")
            continue
        P = S[:, 2:5]
        rate = (len(S) - 1) / max(S[-1, 1] - S[0, 1], 1e-6)
        rng = P.max(axis=0) - P.min(axis=0)
        has_q = not np.isnan(S[0, 5])
        info(f"{topic} [{typ}]: {len(S)} msgs, ~{rate:.0f} Hz, orientation {'YES' if has_q else 'NO'}")
        print(f"      last position (after --scale {args.scale}): {np.round(P[-1], 4)}")
        if has_q:
            print(f"      last quaternion [x y z w]: {np.round(S[-1, 5:9], 4)}")
        print(f"      range moved: {np.round(rng, 4)}")
        if np.max(np.abs(P[-1])) > 5:
            print("      ** values > 5 -> these are probably MILLIMETRES: record with --units mm **")
        if np.max(rng) < 1e-4:
            print("      ** did not change -> frozen in this mode (or you did not move the arm) **")


def cmd_record(args):
    rclpy = J.ros_init()
    topics = discover(rclpy)
    if args.topic not in topics:
        die(f"{args.topic} not found. Published topics: {sorted(topics)}")
    typ = args.type or topics[args.topic]
    node = make_pose_listener([(args.topic, typ)], args)
    node.recording = False
    ex = J.start_spinning(rclpy, node)
    out = J.data_path(args.name, "_ee_raw.csv")
    import threading
    try:
        input(f"\nRecording {args.topic} [{typ}]. Put the arm at the START, keep still, press ENTER ...")
        node.buf[args.topic] = []
        node.recording = True
        t0 = time.monotonic()
        stop = threading.Event()
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        info("RECORDING. Hold still ~1 s, trace the path, hold still ~1 s. ENTER (or Ctrl+C) to stop.")
        while not stop.is_set() and (args.duration <= 0 or time.monotonic() - t0 < args.duration):
            time.sleep(1.0)
            print(f"   {time.monotonic() - t0:5.1f}s samples={len(node.buf[args.topic])}", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        node.recording = False
        S = np.array(node.buf[args.topic]) if node.buf[args.topic] else np.zeros((0, 9))
        J.ros_shutdown(rclpy, ex, node)
    if len(S) < 5:
        die("Fewer than 5 samples - nothing saved.")
    use_st = np.all(S[:, 0] > 0) and np.all(np.diff(S[:, 0]) >= 0)
    t = S[:, 0] if use_st else S[:, 1]
    J.save_csv(out, EE_HEADER, np.column_stack([t - t[0], S[:, 2:9]]))
    info(f"saved {len(S)} samples, {t[-1] - t[0]:.1f}s -> {out}")
    info("DO NOT MOVE THE ARM NOW. Switch to heal_server + launch, then run `ik` (it uses the current "
         "joints as the IK seed and checks the frames).")


def cmd_import_csv(args):
    path = Path(args.file)
    if not path.exists():
        die(f"{path} not found")
    with open(path) as f:
        first = f.readline().strip()
    has_header = any(c.isalpha() for c in first.replace("e-", "").replace("e+", ""))
    A = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0, ndmin=2)
    cols = [c.strip().lower() for c in first.split(",")] if has_header else None

    def col(*names):
        if cols is None:
            return None
        for n in names:
            if n in cols:
                return cols.index(n)
        return None

    if cols is not None and col("x") is not None:
        P = A[:, [col("x"), col("y"), col("z")]]
        it = col("t", "time")
        T = A[:, it] if it is not None else None
        if col("qx") is not None:
            Qt = A[:, [col("qx"), col("qy"), col("qz"), col("qw")]]
        elif col("roll") is not None:
            rpy = A[:, [col("roll"), col("pitch"), col("yaw")]]
            rpy = np.radians(rpy) if args.angles == "deg" else rpy
            Qt = np.array([rpy_to_q(*r) for r in rpy])
        else:
            Qt = np.full((len(A), 4), np.nan)
    else:   # no header: x,y,z[,qx,qy,qz,qw | r,p,y]
        T = None
        P = A[:, :3]
        if A.shape[1] >= 7:
            Qt = A[:, 3:7]
        elif A.shape[1] == 6:
            rpy = np.radians(A[:, 3:6]) if args.angles == "deg" else A[:, 3:6]
            Qt = np.array([rpy_to_q(*r) for r in rpy])
        else:
            Qt = np.full((len(A), 4), np.nan)
    P = P * args.scale
    if T is not None:
        out = J.data_path(args.name, "_ee_raw.csv")
        J.save_csv(out, EE_HEADER, np.column_stack([T - T[0], P, Qt]))
        info(f"{len(P)} timed samples -> {out}.  next: process --name {args.name}")
        return
    # waypoints without time: timing from distance, smooth quintic blends
    info(f"{len(P)} waypoints without time -> building a smooth path at ~{args.ee_speed} m/s")
    write_waypoint_path(args.name, P, Qt, args.ee_speed, args.dt, args.stop_at_waypoints)


def write_waypoint_path(name, P, Qt, ee_speed, dt, stop_each=False):
    has_q = not np.isnan(Qt).any()
    if has_q:
        Qt = continuous_quats(Qt)
    seg = [max(0.8, 1.875 * np.linalg.norm(P[i + 1] - P[i]) / ee_speed) for i in range(len(P) - 1)]
    tk = np.concatenate([[0.0], np.cumsum(seg)])
    V = np.zeros_like(P)
    if not stop_each:
        for i in range(1, len(P) - 1):
            V[i] = (P[i + 1] - P[i - 1]) / (tk[i + 1] - tk[i - 1])
    segs = [J.Quintic(P[i], P[i + 1], seg[i], V[i], V[i + 1]) for i in range(len(P) - 1)]
    tt = np.arange(0, tk[-1] + 1e-9, dt)
    out = np.zeros((len(tt), 8))
    for k, t in enumerate(tt):
        i = min(int(np.searchsorted(tk, t, side="right") - 1), len(segs) - 1)
        u = (t - tk[i]) / seg[i]
        p = segs[i].evaluate(t - tk[i])[0]
        q = slerp(Qt[i], Qt[i + 1], u * u * (3 - 2 * u)) if has_q else [np.nan] * 4
        out[k] = [t, *p, *q]
    path = J.data_path(name, "_ee_traj.csv")
    J.save_csv(path, EE_HEADER, out)
    info(f"path {tk[-1]:.1f}s -> {path}.  next: ik --name {name}")


def cmd_process(args):
    src = Path(args.input) if args.input else J.data_path(args.name, "_ee_raw.csv")
    _, A = J.load_csv(src)
    t, X = J.clean_time(A[:, 0], A[:, 1:8])
    P, Qt = X[:, :3], X[:, 3:7]
    has_q = not np.isnan(Qt).any()
    if has_q:
        Qt = continuous_quats(Qt)
        Y = np.column_stack([P / args.still_tol, Qt / 0.01])
    else:
        Y = P / args.still_tol
    info(f"loaded {len(t)} samples, {t[-1]:.1f}s, orientation {'yes' if has_q else 'NO (position only)'}")
    _, _, (a, b) = J.trim_still(t, Y, 1.0, args.pad)
    m = (t >= a) & (t <= b)
    t, P, Qt = t[m] - a, P[m], Qt[m]
    info(f"trimmed still parts: kept {a:.2f}..{b:.2f}s")
    tt, Pr = J.resample(t, P, args.dt)
    Ps = J.gaussian_smooth(Pr, args.smooth / args.dt)
    if has_q:
        _, Qr = J.resample(t, Qt, args.dt)
        Qs = J.gaussian_smooth(Qr, args.smooth / args.dt)
        Qs /= np.linalg.norm(Qs, axis=1, keepdims=True)
    else:
        Qs = np.full((len(tt), 4), np.nan)
    out = J.data_path(args.name, "_ee_traj.csv")
    J.save_csv(out, EE_HEADER, np.column_stack([tt, Ps, Qs]))
    v = np.linalg.norm(np.gradient(Ps, args.dt, axis=0), axis=1)
    L = np.sum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))
    info(f"saved -> {out}: {tt[-1]:.1f}s, path length {L * 1000:.0f} mm, peak EE speed {v.max() * 1000:.0f} mm/s")
    info(f"next:  python3 {Path(__file__).name} ik --name {args.name}")
    plt = J.plt_or_none()
    if plt is None:
        return
    fig = plt.figure(figsize=(7, 6))
    a3 = fig.add_subplot(111, projection="3d")
    a3.plot(P[:, 0] * 1000, P[:, 1] * 1000, P[:, 2] * 1000, ".", ms=2, alpha=0.3, label="raw")
    a3.plot(Ps[:, 0] * 1000, Ps[:, 1] * 1000, Ps[:, 2] * 1000, label="smoothed")
    a3.set_xlabel("x mm"); a3.set_ylabel("y mm"); a3.set_zlabel("z mm"); a3.legend()
    J.equal_3d(a3, P * 1000, Ps * 1000)
    J.finish_plot(plt, fig, J.data_path(args.name, "_ee_traj.png"), args.show)


def get_seed(args, n):
    if args.seed == "live":
        rclpy = J.ros_init()
        node = J.make_joint_io(args.joints, args.topic, None, "group3_seed")
        ex = J.start_spinning(rclpy, node)
        q = node.wait_for_q(5.0)
        J.ros_shutdown(rclpy, ex, node)
        if q is None:
            die("No /joint_states for the IK seed. Start heal_server + launch, or pass --seed 'a,b,c,d,e,f'.")
        return q
    if args.seed == "zeros":
        return np.zeros(n)
    return J.parse_vec(args.seed, n)


def ik_solve(chain, q, p_t, R_t, mode, lam, iters, tol_p, tol_o, max_step, w_ori=1.0):
    I6, I3 = np.eye(6), np.eye(3)
    for _ in range(iters):
        T = chain.fk(q)
        ep = p_t - T[:3, 3]
        if mode == "free":
            if np.linalg.norm(ep) < tol_p:
                break
            Jm = chain.jacobian(q)[:3]
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam ** 2 * I3, ep)
        else:
            eo = rot_log(R_t @ T[:3, :3].T)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(eo) < tol_o:
                break
            Jm = chain.jacobian(q)
            Jm[3:] *= w_ori
            e = np.concatenate([ep, w_ori * eo])
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam ** 2 * I6, e)
        s = np.max(np.abs(dq))
        if s > max_step:
            dq *= max_step / s
        q = np.clip(q + dq, chain.lower, chain.upper)
    T = chain.fk(q)
    ep = np.linalg.norm(p_t - T[:3, 3])
    eo = 0.0 if mode == "free" else np.linalg.norm(rot_log(R_t @ T[:3, :3].T))
    return q, ep, eo


def cmd_ik(args):
    n = len(args.joints)
    chain = J.load_chain(args, required=False)
    if chain is None:
        info("data/heal.urdf missing -> fetching /robot_description ...")
        xml = J.get_robot_description()
        (DATA / "heal.urdf").write_text(xml)
        chain = J.load_chain(args, required=True)
    src = Path(args.input) if args.input else J.data_path(args.name, "_ee_traj.csv")
    _, A = J.load_csv(src)
    t, P, Qt = A[:, 0], A[:, 1:4], A[:, 4:8]
    has_q = not np.isnan(Qt).any()
    mode = args.orientation
    if mode == "auto":
        mode = "recorded" if has_q else "fixed"
    if mode == "recorded" and not has_q:
        die("This recording has no orientation. Use --orientation fixed or free.")

    q_seed = get_seed(args, n)
    T_seed = chain.fk(q_seed)
    p_seed = T_seed[:3, 3]
    d_start = np.linalg.norm(P[0] - p_seed)
    d_end = np.linalg.norm(P[-1] - p_seed)
    seed_at = args.seed_at if args.seed_at != "auto" else ("end" if d_end < d_start else "start")
    k_ref = -1 if seed_at == "end" else 0
    info(f"seed joints: {np.round(q_seed, 3)}")
    info(f"FK(seed) = {np.round(p_seed * 1000, 1)} mm;  recorded {seed_at} = {np.round(P[k_ref] * 1000, 1)} mm")
    dp = np.linalg.norm(P[k_ref] - p_seed) * 1000
    msg = f"FRAME CHECK: |FK(seed) - recorded {seed_at}| = {dp:.1f} mm"
    if has_q:
        da = math.degrees(np.linalg.norm(rot_log(q_to_R(Qt[k_ref]) @ T_seed[:3, :3].T)))
        msg += f", {da:.1f} deg"
    info(msg)
    if dp > 10:
        info("  >10 mm: either the arm moved after the recording, or the recorded pose uses another frame/"
             "tool point/units than the URDF tip. Try --tip-link (links: "
             + ", ".join(chain.all_links) + "), --units, or --offset auto (shifts the whole path).")
    if args.offset == "auto":
        off = p_seed - P[k_ref]
        P = P + off
        info(f"  shifted recorded path by {np.round(off * 1000, 1)} mm (--offset auto)")

    if mode == "fixed":
        R_fix = T_seed[:3, :3]
        Rt = [R_fix] * len(P)
        info("orientation: held fixed at the seed orientation")
    elif mode == "recorded":
        Rt = [q_to_R(q) for q in Qt]
        info("orientation: following the recorded orientation")
    else:
        Rt = [None] * len(P)
        info("orientation: free (position-only IK)")

    order = range(len(P) - 1, -1, -1) if seed_at == "end" else range(len(P))
    Qs = np.zeros((len(P), n))
    ep_all = np.zeros(len(P))
    eo_all = np.zeros(len(P))
    q = q_seed.copy()
    first = True
    t0 = time.time()
    for k in order:
        q, ep, eo = ik_solve(chain, q, P[k], Rt[k], mode, args.damping, 400 if first else args.iters,
                             args.tol_mm / 1000, args.tol_rad, 0.2 if first else 0.05)
        first = False
        Qs[k], ep_all[k], eo_all[k] = q, ep, eo
    info(f"IK done for {len(P)} samples in {time.time() - t0:.1f}s")

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.01
    dQ = np.gradient(Qs, dt, axis=0)
    dQ[0] = dQ[-1] = 0
    bad = ep_all > 2e-3
    jump = np.max(np.abs(np.diff(Qs, axis=0)), axis=1) / dt
    lines = [f"IK report for {src.name} (mode {mode}, seed at {seed_at})",
             f"position residual mm: mean {ep_all.mean() * 1000:.2f}  max {ep_all.max() * 1000:.2f}",
             f"orientation residual rad: mean {eo_all.mean():.4f}  max {eo_all.max():.4f}",
             f"samples with residual > 2 mm: {bad.sum()} of {len(P)}",
             f"peak joint speed at recorded timing (rad/s): {np.round(np.max(np.abs(dQ), axis=0), 3)}",
             f"largest joint step between samples: {jump.max():.2f} rad/s equivalent",
             f"FK(seed) vs recorded {seed_at}: {dp:.1f} mm"]
    report = "\n".join(lines)
    print(report)
    J.data_path(args.name, "_ik_report.txt").write_text(report + "\n")
    if bad.sum() > 0:
        i = np.nonzero(bad)[0]
        info(f"WARNING: unreachable/unsolved samples at t = {t[i[0]]:.2f}..{t[i[-1]]:.2f}s. "
             f"Try --orientation free, or check frames/units.")
    if jump.max() > 3.0:
        info("WARNING: very fast joint jump -> IK probably switched branch (wrist flip / singularity). "
             "Inspect the plot before replaying; try --orientation free or another seed.")
    out = J.data_path(args.name, "_ik_traj.csv")
    J.save_csv(out, ["t"] + [f"q{i + 1}" for i in range(n)] + [f"dq{i + 1}" for i in range(n)],
               np.column_stack([t - t[0], Qs, dQ]))
    info(f"joint trajectory -> {out}")
    info(f"next:  python3 {Path(__file__).name} replay --name {args.name} --sim     (rehearse)")
    info(f"       python3 {Path(__file__).name} replay --name {args.name} --speed 0.3   (robot)")
    plt = J.plt_or_none()
    if plt is None:
        return
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for j in range(n):
        ax[0].plot(t, Qs[:, j], label=f"joint{j + 1}")
    ax[0].set_ylabel("q [rad]"); ax[0].legend(ncol=6, fontsize=8)
    ax[1].plot(t, ep_all * 1000)
    ax[1].set_ylabel("IK position residual [mm]"); ax[1].set_xlabel("t [s]")
    J.finish_plot(plt, fig, J.data_path(args.name, "_ik_traj.png"), args.show)


def cmd_make_shape(args):
    n = len(args.joints)
    chain = J.load_chain(args, required=True)
    q0 = get_seed(args, n)
    T0 = chain.fk(q0)
    p0 = T0[:3, 3]
    ax = {"x": 0, "y": 1, "z": 2}
    a, b = ax[args.plane[0]], ax[args.plane[1]]
    u, v = np.zeros(3), np.zeros(3)
    u[a], v[b] = 1, 1
    s = args.size
    if args.shape == "line":
        W = [p0, p0 + s * u]
    elif args.shape == "square":
        W = [p0, p0 + s * u, p0 + s * u + s * v, p0 + s * v, p0]
    elif args.shape == "circle":
        c = p0 - s * u
        W = [c + s * (math.cos(th) * u + math.sin(th) * v) for th in np.linspace(0, 2 * math.pi, 73)]
    else:  # figure8
        W = [p0 + s * (math.sin(th) * u + math.sin(th) * math.cos(th) * v) for th in np.linspace(0, 2 * math.pi, 97)]
    W = np.array(W)
    if args.shape in ("circle", "figure8"):
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))])
        Ttot = max(3.0, 1.875 * L[-1] / args.ee_speed)
        tt = np.arange(0, Ttot + 1e-9, args.dt)
        x = np.clip(tt / Ttot, 0, 1)
        sc = L[-1] * (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5)       # rest-to-rest along the curve
        P = np.column_stack([np.interp(sc, L, W[:, j]) for j in range(3)])
        q = R_to_q(T0[:3, :3])
        out = J.data_path(args.name, "_ee_traj.csv")
        J.save_csv(out, EE_HEADER, np.column_stack([tt, P, np.tile(q, (len(tt), 1))]))
        info(f"{args.shape} r={s * 1000:.0f} mm in {args.plane}-plane, {Ttot:.1f}s -> {out}")
    else:
        Qt = np.tile(R_to_q(T0[:3, :3]), (len(W), 1))
        write_waypoint_path(args.name, W, Qt, args.ee_speed, args.dt, stop_each=True)
    info(f"starts at the current pose {np.round(p0 * 1000, 1)} mm.  next: ik --name {args.name} --seed-at start")


def cmd_replay(args, extra):
    J.main(["replay", "--name", f"{args.name}_ik"] + extra)


def cmd_analyze(args, extra):
    J.main(["analyze", "--name", f"{args.name}_ik"] + extra)


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--joints", type=lambda s: s.split(","), default=J.DEFAULT_JOINTS)
    common.add_argument("--topic", default=None, help="pose topic (check/record) or joint topic for the seed")
    common.add_argument("--urdf", default=None)
    common.add_argument("--base-link", default="base_link")
    common.add_argument("--tip-link", default="end-effector")
    common.add_argument("--units", choices=["m", "mm"], default="m", help="units of the recorded positions")
    common.add_argument("--layout", choices=["auto", "xyz", "xyzquat", "xyzrpy"], default="auto",
                        help="meaning of array messages")
    common.add_argument("--angles", choices=["rad", "deg"], default="rad", help="units of rpy angles")
    common.add_argument("--child-frame", default=None, help="for TFMessage: which child frame to record")
    common.add_argument("--show", action="store_true")
    seed = argparse.ArgumentParser(add_help=False)
    seed.add_argument("--seed", default="live", help="'live' (read /joint_states), 'zeros', or 'a,b,c,d,e,f'")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("check", parents=[common])
    s.add_argument("--seconds", type=float, default=8.0)
    s.set_defaults(fn=cmd_check)

    s = sp.add_parser("record", parents=[common])
    s.add_argument("--name", required=True)
    s.add_argument("--type", default=None, help="force message type, e.g. geometry_msgs/msg/PoseStamped")
    s.add_argument("--duration", type=float, default=0.0)
    s.set_defaults(fn=cmd_record)

    s = sp.add_parser("import-csv", parents=[common])
    s.add_argument("--name", required=True)
    s.add_argument("--file", required=True)
    s.add_argument("--ee-speed", type=float, default=0.03, help="m/s for waypoints without time")
    s.add_argument("--stop-at-waypoints", action="store_true")
    s.add_argument("--dt", type=float, default=0.01)
    s.set_defaults(fn=cmd_import_csv)

    s = sp.add_parser("make-shape", parents=[common, seed])
    s.add_argument("--name", required=True)
    s.add_argument("--shape", choices=["line", "square", "circle", "figure8"], default="circle")
    s.add_argument("--size", type=float, default=0.05, help="radius / side / length [m]")
    s.add_argument("--plane", default="xy", help="xy, xz or yz (base_link axes)")
    s.add_argument("--ee-speed", type=float, default=0.03, help="approx peak EE speed [m/s]")
    s.add_argument("--dt", type=float, default=0.01)
    s.set_defaults(fn=cmd_make_shape)

    s = sp.add_parser("process", parents=[common])
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None)
    s.add_argument("--still-tol", type=float, default=0.002, help="m; motion below this counts as still")
    s.add_argument("--pad", type=float, default=0.3)
    s.add_argument("--smooth", type=float, default=0.10, help="Gaussian sigma [s]")
    s.add_argument("--dt", type=float, default=0.01)
    s.set_defaults(fn=cmd_process)

    s = sp.add_parser("ik", parents=[common, seed])
    s.add_argument("--name", required=True)
    s.add_argument("--input", default=None)
    s.add_argument("--orientation", choices=["auto", "recorded", "fixed", "free"], default="auto")
    s.add_argument("--seed-at", choices=["auto", "start", "end"], default="auto",
                   help="which end of the path the arm is at now (auto = the closer one)")
    s.add_argument("--offset", choices=["none", "auto"], default="none")
    s.add_argument("--damping", type=float, default=0.02)
    s.add_argument("--iters", type=int, default=60)
    s.add_argument("--tol-mm", type=float, default=0.2)
    s.add_argument("--tol-rad", type=float, default=0.002)
    s.set_defaults(fn=cmd_ik)

    sp.add_parser("replay", help="same options as heal_joint_replay.py replay").add_argument("--name", required=True)
    sp.add_parser("analyze", help="same options as heal_joint_replay.py analyze").add_argument("--name", required=True)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    if argv and argv[0] in ("replay", "analyze"):
        a, extra = p.parse_known_args(argv)
        (cmd_replay if a.cmd == "replay" else cmd_analyze)(a, extra)
        return
    args = p.parse_args(argv)
    args.scale = 0.001 if args.units == "mm" else 1.0
    if args.cmd in ("ik", "make-shape") and args.topic is None:
        args.topic = J.JS_TOPIC
    args.fn(args)


if __name__ == "__main__":
    main()
