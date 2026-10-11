"""Phan ROS + dieu khien chuyen dong cua panel, KHONG phu thuoc Qt.

Dung chung cho panel (ur5e_panel.py) va cac tool chay headless (tools/*.py), ke ca tren
server Linux khong cai PySide6.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState

import ur5e_kinematics as kin

TICK_HZ = 50.0
MAX_JOINT_SPEED = 1.5  # rad/s o 100%
MAX_LINEAR_SPEED = 0.25  # m/s o 100%
MAX_ANGULAR_SPEED = 1.0  # rad/s o 100%
LINEAR_SETTLE_S = 1.5  # thoi gian toi da de MoveL hoi tu o cuoi duong

# do isaac/ur5e_scene.py ghi ra khi nap scene (gripper, TCP)
GRIPPER_FILE = Path(__file__).resolve().parents[1] / "config" / "gripper.json"


def load_gripper_config():
    try:
        cfg = json.loads(GRIPPER_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cfg if cfg.get("gripper") else None



# --------------------------------------------------------------------------- ROS


class RosBridge(Node):
    def __init__(self, args):
        super().__init__("ur5e_control_panel")
        self._lock = threading.Lock()
        self._joint_pos: dict[str, float] = {}
        self._joint_stamp = 0.0
        self._rgb = None
        self._depth = None
        self._rgb_count = 0
        self._depth_count = 0
        self._sim_time = None

        self.cmd_pub = self.create_publisher(JointState, args.joint_command_topic, 10)
        self.create_subscription(JointState, args.joint_states_topic, self._on_joint_states, 10)
        self.create_subscription(Image, args.rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)

    # callbacks chay trong thread spin: chi luu du lieu
    def _on_joint_states(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos
            self._joint_stamp = time.monotonic()

    def _on_rgb(self, msg: Image):
        with self._lock:
            self._rgb = msg
            self._rgb_count += 1

    def _on_depth(self, msg: Image):
        with self._lock:
            self._depth = msg
            self._depth_count += 1

    def _on_clock(self, msg: Clock):
        with self._lock:
            self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    # truy cap tu thread Qt
    def joint_state(self):
        with self._lock:
            if not all(n in self._joint_pos for n in kin.JOINT_NAMES):
                return None, None
            q = np.array([self._joint_pos[n] for n in kin.JOINT_NAMES])
            return q, time.monotonic() - self._joint_stamp

    def joint_value(self, name):
        with self._lock:
            return self._joint_pos.get(name)

    def take_images(self):
        with self._lock:
            rgb, depth = self._rgb, self._depth
            self._rgb = self._depth = None
            counts = (self._rgb_count, self._depth_count)
        return rgb, depth, counts

    def sim_time(self):
        with self._lock:
            return self._sim_time

    def send_command(self, q, names=kin.JOINT_NAMES):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in q]
        self.cmd_pub.publish(msg)


# ---------------------------------------------------------------- dieu khien


class MotionController:
    """Sinh quy dao muc tieu khop (50 Hz) va gui len /joint_command."""

    def __init__(self, ros: RosBridge):
        self.ros = ros
        self.mode = "idle"  # idle | joint | linear | jog
        self.cmd_q: np.ndarray | None = None
        self.goal_q: np.ndarray | None = None
        self.speed = 0.3
        self.tcp_z = 0.0
        self.error = ""
        self._lin_start = self._lin_goal = None
        self._lin_s = self._lin_settle = 0.0
        self._lin_duration = 1.0
        self._jog_twist = np.zeros(6)
        self._jog_tool_frame = False

    # --- lenh ---
    def _sync(self, actual_q):
        if self.mode == "idle" or self.cmd_q is None:
            if actual_q is None:
                self.error = "Chua nhan duoc /joint_states - kiem tra Isaac Sim da Play chua"
                return False
            self.cmd_q = actual_q.copy()
        self.error = ""
        return True

    def move_joint(self, goal_q, actual_q):
        if not self._sync(actual_q):
            return
        goal_q = np.clip(np.asarray(goal_q, dtype=float), kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
        self.goal_q = goal_q
        self.mode = "joint"

    def move_joint_single(self, index, value, actual_q):
        if not self._sync(actual_q):
            return
        base = self.goal_q if self.mode == "joint" and self.goal_q is not None else self.cmd_q
        goal = base.copy()
        goal[index] = value
        self.move_joint(goal, actual_q)

    def move_pose_joint(self, pose, actual_q):
        if not self._sync(actual_q):
            return
        q = kin.ik_closest(pose, self.cmd_q, self.tcp_z)
        if q is None:
            self.error = "Pose ngoai tam voi (IK vo nghiem)"
            return
        self.move_joint(q, actual_q)

    def move_pose_linear(self, pose, actual_q):
        if not self._sync(actual_q):
            return
        if kin.ik_closest(pose, self.cmd_q, self.tcp_z) is None:
            self.error = "Pose ngoai tam voi (IK vo nghiem)"
            return
        start = kin.fk(self.cmd_q, self.tcp_z)
        dist = np.linalg.norm(pose[:3, 3] - start[:3, 3])
        ang = kin.rotation_angle_between(start, pose)
        v = max(1e-3, MAX_LINEAR_SPEED * self.speed)
        w = max(1e-3, MAX_ANGULAR_SPEED * self.speed)
        self._lin_duration = max(dist / v, ang / w, 0.05)
        self._lin_start, self._lin_goal, self._lin_s, self._lin_settle = start, pose, 0.0, 0.0
        self.mode = "linear"

    def jog(self, twist, tool_frame, actual_q):
        """twist = [vx, vy, vz, wx, wy, wz] da chuan hoa (-1..1)."""
        if not np.any(twist):
            if self.mode == "jog":
                self.mode = "idle"
            return
        if self.mode != "jog" and not self._sync(actual_q):
            return
        self._jog_twist = np.asarray(twist, dtype=float)
        self._jog_tool_frame = tool_frame
        self.mode = "jog"

    def stop(self):
        self.mode = "idle"
        self.goal_q = None

    # --- vong lap 50 Hz ---
    def tick(self, dt):
        if self.mode == "idle" or self.cmd_q is None:
            return
        if self.mode == "joint":
            dq = self.goal_q - self.cmd_q
            vmax = MAX_JOINT_SPEED * self.speed
            t_need = np.max(np.abs(dq)) / max(vmax, 1e-6)
            if t_need <= dt:
                self.cmd_q = self.goal_q.copy()
                self.mode = "idle"
            else:
                self.cmd_q = self.cmd_q + dq * (dt / t_need)
        elif self.mode == "linear":
            self._lin_s = min(1.0, self._lin_s + dt / self._lin_duration)
            # lam min dau/cuoi (smoothstep) de tranh giat
            s = self._lin_s * self._lin_s * (3 - 2 * self._lin_s)
            target = kin.interpolate_pose(self._lin_start, self._lin_goal, s)
            pos_err, rot_err = self._step_to_pose(target, dt)
            if self._lin_s >= 1.0:
                # cho them toi da LINEAR_SETTLE_S de hoi tu (khi bi cham lai gan ky di)
                self._lin_settle += dt
                if pos_err < 2e-4 and rot_err < 2e-3:
                    self.mode = "idle"
                elif self._lin_settle > LINEAR_SETTLE_S:
                    self.error = f"MoveL chua toi dich (sai {pos_err * 1000:.1f} mm) - thu MoveJ"
                    self.mode = "idle"
        elif self.mode == "jog":
            cur = kin.fk(self.cmd_q, self.tcp_z)
            v = self._jog_twist[:3] * MAX_LINEAR_SPEED * self.speed * dt
            w = self._jog_twist[3:] * MAX_ANGULAR_SPEED * self.speed * dt
            target = cur.copy()
            if self._jog_tool_frame:
                target[:3, 3] += cur[:3, :3] @ v
                if np.any(w):
                    target[:3, :3] = cur[:3, :3] @ kin.axis_angle(w, np.linalg.norm(w))
            else:
                target[:3, 3] += v
                if np.any(w):
                    target[:3, :3] = kin.axis_angle(w, np.linalg.norm(w)) @ cur[:3, :3]
            self._step_to_pose(target, dt)
        self.ros.send_command(self.cmd_q)

    def _step_to_pose(self, target, dt):
        """Mot buoc IK DLS tu cmd_q, gioi han toc do khop (an toan gan ky di / bien tam voi)."""
        q, pos_err, rot_err = kin.ik_dls(target, self.cmd_q, self.tcp_z)
        dq = q - self.cmd_q
        max_step = MAX_JOINT_SPEED * dt
        peak = np.max(np.abs(dq))
        if peak > max_step:
            dq *= max_step / peak
        self.cmd_q = self.cmd_q + dq
        return pos_err, rot_err
