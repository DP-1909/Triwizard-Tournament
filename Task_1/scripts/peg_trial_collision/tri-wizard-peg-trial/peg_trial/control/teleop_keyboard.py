"""Keyboard teleoperation (the 30-point baseline + fallback during autonomy).

Run in its OWN terminal (it needs the keyboard):
    ros2 run peg_trial teleop_keyboard

The gripper always points straight down; you move its centre in x / y / z
(robot base frame) and IK computes the joints.
"""
from __future__ import annotations

import json
import math
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from peg_trial.control.arm_interface import ESTOP_TOPIC, ArmInterface
from peg_trial.planning.kinematics import IKError, Pose, fk, ik, ik_best, interpolate_line, LIMITS
from peg_trial.utils.config import load

HELP = """
------------------- PEG TRIAL TELEOP -------------------
  Cartesian Jogging:
    w / s : forward / back   (+x / -x)
    a / d : left / right     (+y / -y)
    r / f : up / down        (+z / -z)
    [ / ] : smaller / bigger step   (now {step} mm)

  Direct Joint Nudging ({joint_step_deg}° per press):
    q / e : Joint 1 (Base Yaw)   left / right
    t / y : Joint 2 (Shoulder)   pitch up / down
    u / i : Joint 3 (Elbow)      pitch up / down
    j / k : Joint 4 (Wrist)      pitch up / down

  Macros & Gripper:
    o / c : gripper OPEN / CLOSE
    g     : GRASP peg (descends, clamps)
    l     : LIFT peg (+3.5 cm)
    p     : PLACE peg (lowers, opens, retreats)
    h     : HOME / READY pose (gripper down above peg)
    1-4   : hover over detected peg (1 tri, 2 sq, 3 hex, 4 circ)
    6-9   : hover over socket (6 tri, 7 sq, 8 hex, 9 circ)
    SPACE : *** EMERGENCY STOP *** (x = release)
    ?     : this help         ESC / Ctrl-C : quit
--------------------------------------------------------
"""
ORDER = ["triangle", "square", "hexagon", "circle"]


class Teleop(Node):
    def __init__(self):
        super().__init__("peg_teleop")
        self.cfg = load("task.yaml")
        self.arm = ArmInterface(self, self.cfg["robot"])
        self.scene = {"pegs": [], "sockets": []}
        self.create_subscription(String, "/peg_trial/scene", self._on_scene, 10)
        self.estop_pub = self.create_publisher(Bool, ESTOP_TOPIC, 10)
        self.step = 0.01
        self.joint_step = 0.05   # ~2.9 degrees per tap
        self.target = None

    def _on_scene(self, m):
        self.scene = json.loads(m.data)

    def current_pose(self) -> Pose:
        q = self.arm.joints()
        if q is not None:
            x, y, z, pitch = fk(q)
            return Pose(x, y, z, pitch)
        if self.target is not None:
            return self.target
        return Pose(0.16, 0.0, 0.115, math.pi / 2)

    def go(self, pose: Pose, speed=0.04, line=True):
        q_start = self.arm.joints()
        if q_start is None:
            print("\r  ! no joint states from arm")
            return False
        x_start, y_start, z_start, pitch_start = fk(q_start)
        start = Pose(x_start, y_start, z_start, pitch_start)

        # 1. Protect against base center singularity and outer reach limits (full 360 deg workspace)
        r = math.hypot(pose.x - 0.012, pose.y)
        if r < 0.09:
            print(f"\r  ! too close to base pivot (r={r:.3f} < 0.09m)")
            return False
        if r > 0.245:
            print(f"\r  ! beyond maximum reach (r={r:.3f} > 0.245m)")
            return False
        # Table collision limit: based on physical table measured ~2cm below base
        real_table_z = max(self.cfg.get("table_z", -0.05), -0.020)
        min_safe_z = real_table_z + 0.035  # gripper center so finger tips never touch table
        if pose.z < min_safe_z:
            print(f"\r  ! table collision limit (z={pose.z:.3f} < {min_safe_z:.3f}m)")
            return False
        if pose.z > 0.22:
            print(f"\r  ! maximum height limit (z={pose.z:.3f} > 0.22m)")
            return False

        # 2. Compute IK for path (try strict requested pitch first, fallback to small tolerance)
        try:
            if line:
                poses = interpolate_line(start, pose, 0.01)
                ik_points = [ik(p) for p in poses]
            else:
                poses = [pose]
                ik_points = [ik(pose)]
        except IKError as e:
            try:
                if line:
                    poses = interpolate_line(start, pose, 0.01)
                    ik_points = [ik_best(p, pitch_tolerance=0.15) for p in poses]
                else:
                    poses = [pose]
                    ik_points = [ik_best(pose, pitch_tolerance=0.15)]
            except IKError:
                print(f"\r  ! unreachable: {e}")
                return False

        # 3. Detect sudden angle jumps (e.g. crossing branch cuts or singularities)
        MAX_SAFE_STEP = 0.40  # rad (~23 deg) per Cartesian step
        prev_q = q_start
        for q in ik_points:
            for j in range(4):
                if abs(q[j] - prev_q[j]) > MAX_SAFE_STEP:
                    print(f"\r  ! motion rejected: sudden joint{j+1} jump ({abs(q[j] - prev_q[j]):.2f} rad)")
                    return False
            prev_q = q

        # 4. Enforce maximum joint velocity so motors never spin wildly
        MAX_JOINT_VEL = 0.8  # rad/s max safe joint speed
        pts = []
        t_acc = 0.0
        prev_q = q_start
        prev_p = start

        for p, q in zip(poses, ik_points):
            cart_dist = math.dist((prev_p.x, prev_p.y, prev_p.z), (p.x, p.y, p.z))
            t_cart = cart_dist / speed
            t_joint = max(abs(q[j] - prev_q[j]) for j in range(4)) / MAX_JOINT_VEL
            dt = max(0.12, t_cart, t_joint)
            t_acc += dt
            pts.append((q, t_acc))
            prev_q = q
            prev_p = p

        self.target = pose
        print(f"\r  -> x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f}")
        return self.arm.move(pts, settle=0.0)

    def handle_key(self, k):
        p = self.current_pose()
        s = self.step
        g = self.cfg["grasp"]
        moves = {"w": (s, 0, 0), "s": (-s, 0, 0), "a": (0, s, 0), "d": (0, -s, 0), "r": (0, 0, s), "f": (0, 0, -s)}
        joint_moves = {
            "q": (0, +self.joint_step, "Joint 1 (Base Yaw) +left"),
            "e": (0, -self.joint_step, "Joint 1 (Base Yaw) -right"),
            "t": (1, -self.joint_step, "Joint 2 (Shoulder) up"),
            "y": (1, +self.joint_step, "Joint 2 (Shoulder) down"),
            "u": (2, -self.joint_step, "Joint 3 (Elbow) up"),
            "i": (2, +self.joint_step, "Joint 3 (Elbow) down"),
            "j": (3, -self.joint_step, "Joint 4 (Wrist) tilt up"),
            "k": (3, +self.joint_step, "Joint 4 (Wrist) tilt down"),
        }
        if k == " ":
            self.estop_pub.publish(Bool(data=True))
            print("\r  *** E-STOP *** (press x to release)")
        elif k == "x":
            self.estop_pub.publish(Bool(data=False))
            self.target = None
        elif self.arm.estopped:
            print("\r  e-stop active — press x first")
        elif k in moves:
            dx, dy, dz = moves[k]
            target_pitch = p.pitch if abs(p.pitch - math.pi / 2) < 0.3 else math.pi / 2
            self.go(Pose(p.x + dx, p.y + dy, p.z + dz, target_pitch), line=False if s >= 0.02 else True)
        elif k in joint_moves:
            idx, dq, name = joint_moves[k]
            q_now = self.arm.joints()
            if q_now is None:
                print("\r  ! no joint states from arm")
                return
            new_q = list(q_now)
            target_val = new_q[idx] + dq
            lo, hi = LIMITS[idx]
            if not (lo <= target_val <= hi):
                print(f"\r  ! {name} limit reached ({target_val:.2f} rad not in [{lo:.2f}, {hi:.2f}])")
                return
            new_q[idx] = target_val
            t = max(0.35, abs(dq) / 0.5)
            self.arm.move([(new_q, t)], settle=0.05)
            self.target = None
            x, y, z, pitch = fk(new_q)
            print(f"\r  {name}: {target_val:.2f} rad ({target_val*180/math.pi:.1f}°) -> (x={x:.3f}, y={y:.3f}, z={z:.3f})")
        elif k in "[]":
            self.step = max(0.002, self.step / 2) if k == "[" else min(0.05, self.step * 2)
            print(f"\r  step = {self.step * 1000:.0f} mm")
        elif k == "o":
            self.arm.gripper("open")
        elif k == "c":
            self.arm.gripper("close")
        elif k == "g":
            q_now = self.arm.joints()
            if q_now is None:
                print("\r  ! no joint states from arm")
                return
            x_now, y_now, z_now, pitch_now = fk(q_now)
            pitch_grasp = pitch_now if abs(pitch_now - math.pi / 2) < 0.3 else math.pi / 2

            real_table_z = max(self.cfg.get("table_z", -0.05), -0.020)
            peg = min(self.scene["pegs"], key=lambda q: math.hypot(q["x"] - x_now, q["y"] - y_now), default=None)
            if peg:
                top_z = peg["z_top"]
            else:
                top_z = real_table_z + self.cfg.get("peg_height", 0.10)
            # Firm grasp: grip 2.5 cm below peg top for deep, tight hold
            z_grasp = max(top_z - 0.025, real_table_z + 0.035)

            self.arm.gripper("open")
            # If above grasp depth, descend straight down at exact current (x, y)
            if z_now > z_grasp + 0.005:
                print(f"\r  descending vertically to peg (z={z_grasp:.3f}m) ...")
                self.go(Pose(x_now, y_now, z_grasp, pitch_grasp), speed=0.025)

            time.sleep(g["settle_delay"])
            print("\r  clamping peg ...")
            self.arm.gripper("close", wait=g["close_delay"])
        elif k == "l":
            q_now = self.arm.joints()
            if q_now is not None:
                x_now, y_now, z_now, pitch_now = fk(q_now)
                pitch_lift = pitch_now if abs(pitch_now - math.pi / 2) < 0.3 else math.pi / 2
                self.go(Pose(x_now, y_now, z_now + 0.035, pitch_lift), speed=0.03)
        elif k == "p":
            print("\r  placing peg into socket ...")
            q_now = self.arm.joints()
            if q_now is not None:
                x_now, y_now, z_now, pitch_now = fk(q_now)
                pitch_place = pitch_now if abs(pitch_now - math.pi / 2) < 0.3 else math.pi / 2
                if self.go(Pose(x_now, y_now, z_now - 0.025, pitch_place), speed=0.015):
                    self.arm.gripper("open")
                    time.sleep(self.cfg["place"].get("release_delay", 0.8))
                    self.go(Pose(x_now, y_now, z_now + 0.040, pitch_place), speed=0.03)
        elif k in "1234" or k in "6789":
            shape = ORDER["1234".index(k)] if k in "1234" else ORDER["6789".index(k)]
            items = self.scene["pegs"] if k in "1234" else self.scene["sockets"]
            it = next((i for i in items if i["shape"] == shape), None)
            if it is None:
                print(f"\r  no {shape} detected")
                return
            z = it["z_top"] + (g["pregrasp_offset"] if k in "1234" else 0.10)
            self.go(Pose(it["x"], it["y"], z), line=False)
        elif k == "h":
            self.target = None
            q_now = self.arm.joints() or [0.0, 0.046, -0.355, 1.880]
            ready_joints = [0.0, 0.046, -0.355, 1.880]  # z = 0.115m (above 10cm peg)
            max_d = max(abs(a - b) for a, b in zip(q_now, ready_joints))
            t_home = max(3.0, max_d / 0.6)
            print(f"\r  moving to ready pose (z=0.115m, clears pegs) in {t_home:.1f}s ...")
            self.arm.move([(ready_joints, t_home)], settle=0.2)
            self.target = Pose(0.16, 0.0, 0.115, math.pi / 2)
        elif k == "?":
            print(HELP.format(step=int(self.step * 1000), joint_step_deg=round(self.joint_step * 180 / math.pi, 1)))


def main():
    rclpy.init()
    node = Teleop()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not node.arm.wait_for_joints(10.0):
        print("no /joint_states — start the robot bringup first")
        return
    print(HELP.format(step=int(node.step * 1000), joint_step_deg=round(node.joint_step * 180 / math.pi, 1)))
    old = termios.tcgetattr(sys.stdin)
    worker = None
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k in ("\x1b", "\x03"):
                    break
                if k == " ":                       # E-STOP is handled HERE, never blocked
                    node.handle_key(k)
                    continue
                if worker and worker.is_alive():   # a motion is still running
                    print("\r  (busy — wait, or SPACE to stop)")
                    continue
                worker = threading.Thread(target=node.handle_key, args=(k,), daemon=True)
                worker.start()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.arm.hold()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
