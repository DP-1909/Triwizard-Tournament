"""One small class that hides HOW we talk to the robot.

backend (config/task.yaml -> robot.backend):
  ros2_control  : ROBOTIS open_manipulator Humble bringup (real robot or Gazebo).
                  Arm  -> trajectory_msgs/JointTrajectory on /arm_controller/joint_trajectory
                  Grip -> control_msgs/action/GripperCommand on /gripper_controller/gripper_cmd
  omx_services  : older open_manipulator_x_controller (services /goal_joint_space_path,
                  /goal_tool_control from open_manipulator_msgs)
  dry_run       : no robot at all, only prints + fakes joint states (test the logic)

Every call checks the emergency-stop flag. When E-STOP is pressed the arm is
told to hold its current position and every running motion returns False.
"""
from __future__ import annotations

import threading
import time

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ESTOP_TOPIC = "/peg_trial/estop"


def _dur(t: float) -> Duration:
    return Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))


class ArmInterface:
    def __init__(self, node, robot_cfg: dict):
        self.node = node
        self.cfg = robot_cfg
        self.backend = robot_cfg.get("backend", "ros2_control")
        self.names = robot_cfg["joint_names"]
        self._q = None
        self._lock = threading.Lock()
        self.estopped = False

        node.create_subscription(Bool, ESTOP_TOPIC, self._on_estop, 10)
        self.estop_pub = node.create_publisher(Bool, ESTOP_TOPIC, 10)

        if self.backend == "dry_run":
            self._q = [0.0, -0.84, -0.10, 0.94]
        else:
            node.create_subscription(JointState, robot_cfg["joint_states_topic"], self._on_js, 10)

        if self.backend == "ros2_control":
            from control_msgs.action import GripperCommand, FollowJointTrajectory
            from rclpy.action import ActionClient
            from std_msgs.msg import Float64MultiArray
            self._GripperCommand = GripperCommand
            self._FollowJointTrajectory = FollowJointTrajectory
            self.traj_pub = node.create_publisher(JointTrajectory, robot_cfg["arm_trajectory_topic"], 10)
            self.arm_client = ActionClient(node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
            self.grip_client = ActionClient(node, GripperCommand, robot_cfg["gripper_action"])
            self.grip_topic_pub = node.create_publisher(Float64MultiArray, "/gripper_controller/commands", 10)
        elif self.backend == "omx_services":
            from open_manipulator_msgs.srv import SetJointPosition
            self._SetJointPosition = SetJointPosition
            self.joint_cli = node.create_client(SetJointPosition, robot_cfg["joint_service"])
            self.tool_cli = node.create_client(SetJointPosition, robot_cfg["tool_service"])

    # ---------------- state ----------------
    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in self.names):
            with self._lock:
                self._q = [msg.position[idx[n]] for n in self.names]

    def _on_estop(self, msg: Bool):
        if msg.data and not self.estopped:
            self.node.get_logger().error("EMERGENCY STOP received -> holding position")
            self.estopped = True
            self.hold()
        elif not msg.data and self.estopped:
            self.node.get_logger().warn("E-stop released")
            self.estopped = False

    def joints(self):
        with self._lock:
            return list(self._q) if self._q else None

    def wait_for_joints(self, timeout=10.0) -> bool:
        t0 = time.time()
        while self.joints() is None and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.joints() is not None

    def trigger_estop(self):
        self.estop_pub.publish(Bool(data=True))

    # ---------------- arm ----------------
    def hold(self):
        try:
            q = self.joints()
            if q is None or self.backend == "dry_run":
                return
            if self.backend == "ros2_control":
                self._publish_traj([(q, 0.2)])
            else:
                self._send_joint(q, 0.2)
        except Exception:
            pass

    def _publish_traj(self, points):
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(self.names)

        q_curr = self.joints()
        # Always prepend current state at t=0 with zero velocity so the splines controller has a valid baseline
        if q_curr is not None:
            p0 = JointTrajectoryPoint()
            p0.positions = [float(a) for a in q_curr]
            p0.velocities = [0.0] * len(self.names)
            p0.time_from_start = Duration(sec=0, nanosec=0)
            msg.points.append(p0)

        # velocities: 0 at the start/end, central differences in between -> smooth, no stop-start
        allp = ([(q_curr, 0.0)] if q_curr is not None else []) + list(points)
        k0 = len(allp) - len(points)
        for i in range(k0, len(allp)):
            q, t = allp[i]
            pt = JointTrajectoryPoint()
            pt.positions = [float(a) for a in q]
            if 0 < i < len(allp) - 1:
                (qa, ta), (qb, tb) = allp[i - 1], allp[i + 1]
                pt.velocities = [float((b - a) / max(tb - ta, 1e-3)) for a, b in zip(qa, qb)]
            else:
                pt.velocities = [0.0] * len(self.names)
            pt.time_from_start = _dur(t)
            msg.points.append(pt)

        # send ONCE: action if available (gives a result), otherwise the topic.
        # Sending both makes the controller receive two goals and replace the first mid-motion.
        if hasattr(self, "arm_client") and self.arm_client.server_is_ready():
            goal = self._FollowJointTrajectory.Goal()
            goal.trajectory = msg
            self.arm_client.send_goal_async(goal)
        else:
            self.traj_pub.publish(msg)

    def _send_joint(self, q, t):
        req = self._SetJointPosition.Request()
        req.planning_group = ""
        req.joint_position.joint_name = list(self.names)
        req.joint_position.position = [float(a) for a in q]
        req.path_time = float(t)
        self.joint_cli.call_async(req)

    def _sleep(self, sec) -> bool:
        """Sleep but wake up immediately on E-STOP. Returns False if stopped."""
        end = time.time() + sec
        while time.time() < end:
            if self.estopped:
                return False
            time.sleep(0.02)
        return not self.estopped

    def move(self, points, settle=0.15) -> bool:
        """points = [(q[4], time_from_start), ...]. Blocks until done."""
        if self.estopped or not points:
            return not self.estopped
        total = points[-1][1]
        if self.backend == "ros2_control":
            self._publish_traj(points)
        elif self.backend == "omx_services":
            prev = 0.0
            for q, t in points:          # services take one goal at a time
                self._send_joint(q, max(0.1, t - prev))
                if not self._sleep(max(0.1, t - prev)):
                    return False
                prev = t
            return self._sleep(settle)
        else:
            self.node.get_logger().info(f"[dry_run] move {len(points)} pts in {total:.2f}s -> {[round(a, 2) for a in points[-1][0]]}")
            ok = self._sleep(min(total, 0.3))
            with self._lock:
                self._q = list(points[-1][0])
            return ok
        return self._sleep(total + settle)

    # ---------------- gripper ----------------
    def gap_to_position(self, gap):
        """Finger gap (m) -> gripper command, by straight-line interpolation between the two
        measured points (gripper_open <-> gripper_gap_open, gripper_close <-> gripper_gap_closed)."""
        g_o, g_c = self.cfg.get("gripper_gap_open"), self.cfg.get("gripper_gap_closed")
        if g_o is None or g_c is None or abs(g_o - g_c) < 1e-6:
            return None
        f = min(1.0, max(0.0, (gap - g_c) / (g_o - g_c)))
        return self.cfg["gripper_close"] + f * (self.cfg["gripper_open"] - self.cfg["gripper_close"])

    def gripper(self, cmd: str, wait=0.8, gap=None) -> bool:
        if self.estopped:
            return False
        # "open", "pregrasp", "release" open wide to gripper_open so peg is freely released
        use_adaptive = self.cfg.get("use_adaptive_gap", False)
        pos = {"open": self.cfg["gripper_open"], "close": self.cfg["gripper_close"],
               "pregrasp": self.cfg["gripper_open"],
               "release": self.cfg["gripper_open"]}.get(cmd, self.cfg["gripper_open"])
        if use_adaptive and gap is not None and cmd != "close" and self.gap_to_position(gap) is not None:
            pos = self.gap_to_position(gap)
        if self.backend == "ros2_control":
            self.node.get_logger().info(f"gripper {cmd}: position={pos:.4f}")
            if not self.grip_client.server_is_ready():
                self.grip_client.wait_for_server(timeout_sec=2.0)
            if self.grip_client.server_is_ready():
                goal = self._GripperCommand.Goal()
                goal.command.position = float(pos)
                goal.command.max_effort = float(self.cfg.get("gripper_max_effort", 5.0))
                self.grip_client.send_goal_async(goal)
            else:
                self.node.get_logger().error(f"Gripper action server not available! Checked: {self.cfg.get('gripper_action')}")
                from std_msgs.msg import Float64MultiArray
                msg = Float64MultiArray()
                msg.data = [float(pos)]
                self.grip_topic_pub.publish(msg)
        elif self.backend == "omx_services":
            req = self._SetJointPosition.Request()
            req.joint_position.joint_name = ["gripper"]
            req.joint_position.position = [float(pos)]
            req.path_time = 0.5
            self.tool_cli.call_async(req)
        else:
            self.node.get_logger().info(f"[dry_run] gripper {cmd}")
            wait = 0.1
        return self._sleep(wait)
