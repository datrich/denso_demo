"""Panel dieu khien UR5e trong Isaac Sim qua ROS 2 Jazzy.

Giao tiep (mac dinh, khop voi isaac/ur5e_scene.py):
  pub  /joint_command                 sensor_msgs/JointState  (vi tri muc tieu cac khop)
  sub  /joint_states                  sensor_msgs/JointState  (trang thai khop tu sim)
  sub  /camera/color/image_raw        sensor_msgs/Image       (anh mau D435)
  sub  /camera/depth/image_rect_raw   sensor_msgs/Image       (anh do sau D435, 32FC1)
  sub  /clock                         rosgraph_msgs/Clock

Chay: pixi run panel   (Zenoh router va Isaac Sim phai dang chay)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from PySide6 import QtCore, QtGui, QtWidgets
from rclpy.executors import ExternalShutdownException
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

POSES_FILE = Path(__file__).resolve().parents[1] / "config" / "poses.json"
# do isaac/ur5e_scene.py ghi ra khi nap scene (gripper, TCP)
GRIPPER_FILE = Path(__file__).resolve().parents[1] / "config" / "gripper.json"


def load_gripper_config():
    try:
        cfg = json.loads(GRIPPER_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cfg if cfg.get("gripper") else None
DEFAULT_POSES = {
    "Home (scene)": [0.0, -51.4, 66.5, -67.3, 0.0, 0.0],
    "Ready (hướng xuống)": [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    "Zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


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


# --------------------------------------------------------------------- anh


def _jet_lut():
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


_LUT = _jet_lut()


def image_to_qimage(msg: Image, depth_range=(0.1, 3.0)):
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
    elif enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
    elif enc == "mono8":
        arr = np.repeat(buf.reshape(h, msg.step)[:, :w, None], 3, axis=2)
    elif enc in ("32fc1", "16uc1"):
        if enc == "32fc1":
            d = buf.view(np.float32).reshape(h, msg.step // 4)[:, :w]
        else:
            d = buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w].astype(np.float32) / 1000.0
        lo, hi = depth_range
        valid = np.isfinite(d) & (d > 0)
        # Isaac tra +inf o cho khong co vat -> thay truoc khi nhan de tranh tran so
        d_clean = np.nan_to_num(d, nan=0.0, posinf=hi, neginf=0.0)
        idx = np.clip((d_clean - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        arr = _LUT[idx]
        arr[~valid] = 0
    else:
        return None
    arr = np.ascontiguousarray(arr)
    return QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------- UI


def deg(v):
    return math.degrees(v)


class JointRow(QtWidgets.QWidget):
    changed = QtCore.Signal(int, float)  # (index, rad)

    def __init__(self, index, name, lo, hi):
        super().__init__()
        self.index = index
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        label = QtWidgets.QLabel(name.replace("_joint", ""))
        label.setMinimumWidth(110)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(int(deg(lo) * 10), int(deg(hi) * 10))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(deg(lo), deg(hi))
        self.spin.setDecimals(1)
        self.spin.setSuffix(" °")
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(90)
        self.actual = QtWidgets.QLabel("—")
        self.actual.setMinimumWidth(80)
        self.actual.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.actual.setToolTip("Goc thuc te tu /joint_states")
        lay.addWidget(label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.actual)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._quiet = False

    def _from_slider(self, v):
        if self._quiet:
            return
        self._set_quiet(v / 10.0)
        self.changed.emit(self.index, math.radians(v / 10.0))

    def _from_spin(self, v):
        if self._quiet:
            return
        self._set_quiet(v)
        self.changed.emit(self.index, math.radians(v))

    def _set_quiet(self, value_deg):
        self._quiet = True
        self.slider.setValue(int(round(value_deg * 10)))
        self.spin.setValue(value_deg)
        self._quiet = False

    def show_target(self, q_rad):
        if self.slider.isSliderDown() or self.spin.hasFocus():
            return
        self._set_quiet(deg(q_rad))

    def show_actual(self, q_rad):
        self.actual.setText("—" if q_rad is None else f"{deg(q_rad):7.1f} °")


class JogButton(QtWidgets.QPushButton):
    def __init__(self, text, axis, sign, panel):
        super().__init__(text)
        self.setMinimumHeight(36)
        self.setAutoRepeat(False)
        self.pressed.connect(lambda: panel.set_jog(axis, sign))
        self.released.connect(lambda: panel.set_jog(axis, 0))


class Panel(QtWidgets.QMainWindow):
    def __init__(self, ros: RosBridge):
        super().__init__()
        self.ros = ros
        self.ctrl = MotionController(ros)
        self.jog_twist = np.zeros(6)
        self.actual_q = None
        self.poses = self._load_poses()
        self.gripper = load_gripper_config()
        self.setWindowTitle("UR5e Control Panel — Isaac Sim / ROS 2 Jazzy")
        self.resize(1000, 760)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.addLayout(self._build_top_bar())
        if self.gripper:
            root.addWidget(self._build_gripper_bar())
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_joint_tab(), "Khớp")
        tabs.addTab(self._build_cartesian_tab(), "Cartesian (IK)")
        if self.gripper:  # TCP = grip_frame giua 2 dau ngon (do scene tinh)
            self.tcp_spin.setValue(self.gripper["tcp_offset_m"][2] * 1000.0)
        tabs.addTab(self._build_camera_tab(), "Camera D435")
        root.addWidget(tabs, 1)
        self.setCentralWidget(central)

        self.status = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status, 1)
        self.err = QtWidgets.QLabel()
        self.err.setStyleSheet("color:#e5534b;font-weight:600")
        self.statusBar().addPermanentWidget(self.err)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / TICK_HZ))
        self._last_tick = time.monotonic()
        self._ui_counter = 0
        self._img_fps = [0, 0, time.monotonic(), 0.0, 0.0]

    # ---------- dung UI ----------
    def _build_gripper_bar(self):
        g = self.gripper
        box = QtWidgets.QGroupBox(f"Gripper Robotiq 2F-85 ({g.get('physics_variant', '')})  —  0% = mở, 100% = đóng")
        lay = QtWidgets.QHBoxLayout(box)
        for text, pct in (("Mở", 0), ("Đóng", 100)):
            b = QtWidgets.QPushButton(text)
            b.setMinimumWidth(80)
            b.clicked.connect(lambda _=False, p=pct: self.grip_slider.setValue(p))
            lay.addWidget(b)
        self.grip_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grip_slider.setRange(0, 100)
        self.grip_slider.valueChanged.connect(self._on_grip)
        lay.addWidget(self.grip_slider, 1)
        self.grip_cmd_lbl = QtWidgets.QLabel("lệnh 0%")
        self.grip_act_lbl = QtWidgets.QLabel("thực tế —")
        self.grip_act_lbl.setToolTip("Dừng giữa chừng khi đóng = đang kẹp vật")
        for w in (self.grip_cmd_lbl, self.grip_act_lbl):
            w.setMinimumWidth(95)
            lay.addWidget(w)
        return box

    def _on_grip(self, pct):
        self.grip_cmd_lbl.setText(f"lệnh {pct}%")
        g = self.gripper
        target = g["open_rad"] + (g["closed_rad"] - g["open_rad"]) * pct / 100.0
        self.ros.send_command([target], names=[g["finger_joint"]])

    def _build_top_bar(self):
        bar = QtWidgets.QHBoxLayout()
        self.conn = QtWidgets.QLabel("● chưa kết nối")
        self.conn.setStyleSheet("color:#e5534b;font-weight:600")
        bar.addWidget(self.conn)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Tốc độ"))
        self.speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed.setRange(1, 100)
        self.speed.setValue(30)
        self.speed.setMaximumWidth(220)
        self.speed_lbl = QtWidgets.QLabel("30 %")
        self.speed.valueChanged.connect(self._on_speed)
        bar.addWidget(self.speed)
        bar.addWidget(self.speed_lbl)
        bar.addStretch(1)
        stop = QtWidgets.QPushButton("STOP")
        stop.setMinimumSize(110, 40)
        stop.setStyleSheet(
            "QPushButton{background:#c93c37;color:white;font-weight:700;border-radius:6px}"
            "QPushButton:pressed{background:#8f2723}"
        )
        stop.clicked.connect(self._stop)
        stop.setShortcut(QtGui.QKeySequence("Space"))
        stop.setToolTip("Dừng mọi chuyển động (phím Space)")
        bar.addWidget(stop)
        return bar

    def _build_joint_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)

        left = QtWidgets.QGroupBox("Góc khớp (mục tiêu  |  thực tế)")
        ll = QtWidgets.QVBoxLayout(left)
        self.rows = []
        for i, name in enumerate(kin.JOINT_NAMES):
            row = JointRow(i, name, *kin.JOINT_LIMITS[i])
            row.changed.connect(self._on_joint_row)
            ll.addWidget(row)
            self.rows.append(row)
        ll.addStretch(1)
        lay.addWidget(left, 3)

        right = QtWidgets.QGroupBox("Tư thế đã lưu")
        rl = QtWidgets.QVBoxLayout(right)
        self.pose_list = QtWidgets.QListWidget()
        self.pose_list.addItems(self.poses.keys())
        self.pose_list.itemDoubleClicked.connect(lambda _: self._goto_saved())
        rl.addWidget(self.pose_list, 1)
        for text, fn in (("Đi tới (MoveJ)", self._goto_saved), ("Lưu tư thế hiện tại…", self._save_pose), ("Xóa", self._delete_pose)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            rl.addWidget(b)
        lay.addWidget(right, 1)
        return w

    def _build_cartesian_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)

        # pose hien tai + muc tieu
        pose_box = QtWidgets.QGroupBox("Pose TCP trong khung base_link")
        grid = QtWidgets.QGridLayout(pose_box)
        grid.addWidget(QtWidgets.QLabel("<b>Hiện tại</b>"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>Mục tiêu</b>"), 0, 2)
        self.cur_lbls, self.tgt_spins = [], []
        for r, (name, unit, rng) in enumerate(
            [("X", "mm", 1000), ("Y", "mm", 1000), ("Z", "mm", 1000), ("Roll", "°", 180), ("Pitch", "°", 180), ("Yaw", "°", 180)]
        ):
            grid.addWidget(QtWidgets.QLabel(name), r + 1, 0)
            lbl = QtWidgets.QLabel("—")
            lbl.setMinimumWidth(90)
            grid.addWidget(lbl, r + 1, 1)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-rng, rng)
            sp.setDecimals(1)
            sp.setSuffix(f" {unit}")
            grid.addWidget(sp, r + 1, 2)
            self.cur_lbls.append(lbl)
            self.tgt_spins.append(sp)
        btns = QtWidgets.QHBoxLayout()
        for text, fn in (("← Lấy pose hiện tại", self._copy_pose), ("MoveJ", self._movej), ("MoveL", self._movel)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        grid.addLayout(btns, 7, 0, 1, 3)
        grid.addWidget(QtWidgets.QLabel("TCP offset Z"), 8, 0)
        self.tcp_spin = QtWidgets.QDoubleSpinBox()
        self.tcp_spin.setRange(0, 400)
        self.tcp_spin.setSuffix(" mm")
        self.tcp_spin.setToolTip("Khoảng cách từ tool0 tới đầu công cụ (dùng khi gắn gripper)")
        self.tcp_spin.valueChanged.connect(lambda v: setattr(self.ctrl, "tcp_z", v / 1000.0))
        grid.addWidget(self.tcp_spin, 8, 1, 1, 2)
        grid.setRowStretch(9, 1)
        lay.addWidget(pose_box, 1)

        # jog
        jog_box = QtWidgets.QGroupBox("Jog (nhấn giữ)")
        jl = QtWidgets.QVBoxLayout(jog_box)
        frame_row = QtWidgets.QHBoxLayout()
        self.frame_base = QtWidgets.QRadioButton("Khung Base")
        self.frame_tool = QtWidgets.QRadioButton("Khung Tool")
        self.frame_base.setChecked(True)
        frame_row.addWidget(self.frame_base)
        frame_row.addWidget(self.frame_tool)
        frame_row.addStretch(1)
        jl.addLayout(frame_row)
        g = QtWidgets.QGridLayout()
        for axis, name in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            g.addWidget(JogButton(f"{name} −", axis, -1, self), axis, 0)
            g.addWidget(JogButton(f"{name} +", axis, +1, self), axis, 1)
        jl.addLayout(g)
        jl.addStretch(1)
        lay.addWidget(jog_box, 1)
        return w

    def _build_camera_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QGridLayout(w)
        self.rgb_view = self._image_label()
        self.depth_view = self._image_label()
        self.rgb_info = QtWidgets.QLabel(f"Màu: {ARGS.rgb_topic}")
        self.depth_info = QtWidgets.QLabel(f"Độ sâu: {ARGS.depth_topic}  (0.1–3 m)")
        lay.addWidget(self.rgb_info, 0, 0)
        lay.addWidget(self.depth_info, 0, 1)
        lay.addWidget(self.rgb_view, 1, 0)
        lay.addWidget(self.depth_view, 1, 1)
        lay.setRowStretch(1, 1)
        lay.setColumnStretch(0, 1)
        lay.setColumnStretch(1, 1)
        return w

    @staticmethod
    def _image_label():
        lbl = QtWidgets.QLabel("Chưa có ảnh")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(320, 240)
        lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        lbl.setStyleSheet("background:#111;color:#888;border-radius:4px")
        return lbl

    # ---------- su kien ----------
    def _on_speed(self, v):
        self.ctrl.speed = v / 100.0
        self.speed_lbl.setText(f"{v} %")

    def _stop(self):
        self.jog_twist[:] = 0
        self.ctrl.stop()

    def _on_joint_row(self, index, value):
        self.ctrl.move_joint_single(index, value, self.actual_q)

    def set_jog(self, axis, sign):
        self.jog_twist[axis] = sign
        self.ctrl.jog(self.jog_twist.copy(), self.frame_tool.isChecked(), self.actual_q)

    def _target_pose(self):
        v = [s.value() for s in self.tgt_spins]
        return kin.make_pose(np.array(v[:3]) / 1000.0, np.radians(v[3:]))

    def _copy_pose(self):
        q = self.ctrl.cmd_q if self.ctrl.mode != "idle" else self.actual_q
        if q is None:
            return
        t = kin.fk(q, self.ctrl.tcp_z)
        vals = list(t[:3, 3] * 1000.0) + [deg(a) for a in kin.matrix_to_rpy(t[:3, :3])]
        for sp, v in zip(self.tgt_spins, vals):
            sp.setValue(v)

    def _movej(self):
        self.ctrl.move_pose_joint(self._target_pose(), self.actual_q)

    def _movel(self):
        self.ctrl.move_pose_linear(self._target_pose(), self.actual_q)

    def _goto_saved(self):
        item = self.pose_list.currentItem()
        if item:
            self.ctrl.move_joint(np.radians(self.poses[item.text()]), self.actual_q)

    def _save_pose(self):
        if self.actual_q is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Lưu tư thế", "Tên tư thế:")
        if ok and name.strip():
            name = name.strip()
            if name not in self.poses:
                self.pose_list.addItem(name)
            self.poses[name] = [round(deg(v), 2) for v in self.actual_q]
            self._write_poses()

    def _delete_pose(self):
        item = self.pose_list.currentItem()
        if item:
            self.poses.pop(item.text(), None)
            self.pose_list.takeItem(self.pose_list.row(item))
            self._write_poses()

    def _load_poses(self):
        try:
            return json.loads(POSES_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(DEFAULT_POSES)

    def _write_poses(self):
        POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        POSES_FILE.write_text(json.dumps(self.poses, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- vong lap ----------
    def _tick(self):
        now = time.monotonic()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        self.actual_q, age = self.ros.joint_state()
        if self.actual_q is not None and age > 1.0:
            self.actual_q = None  # du lieu cu: sim dang Pause/Stop
        self.ctrl.tick(dt)

        self._ui_counter += 1
        if self._ui_counter % 3 == 0:  # ~17 Hz cap nhat giao dien
            self._refresh_ui(age)

    def _refresh_ui(self, age):
        if self.actual_q is not None:
            self.conn.setText("● đã kết nối")
            self.conn.setStyleSheet("color:#3fb950;font-weight:600")
        else:
            self.conn.setText("● chưa nhận /joint_states" if age is None else "● sim đang dừng")
            self.conn.setStyleSheet("color:#e5534b;font-weight:600")

        target = self.ctrl.goal_q if self.ctrl.mode == "joint" else self.ctrl.cmd_q
        shown = target if self.ctrl.mode != "idle" and target is not None else self.actual_q
        for i, row in enumerate(self.rows):
            row.show_actual(None if self.actual_q is None else self.actual_q[i])
            if shown is not None:
                row.show_target(shown[i])

        if self.actual_q is not None:
            t = kin.fk(self.actual_q, self.ctrl.tcp_z)
            vals = list(t[:3, 3] * 1000.0) + [deg(a) for a in kin.matrix_to_rpy(t[:3, :3])]
            for lbl, v, unit in zip(self.cur_lbls, vals, ["mm"] * 3 + ["°"] * 3):
                lbl.setText(f"{v:8.1f} {unit}")

        if self.gripper:
            g = self.gripper
            fj = self.ros.joint_value(g["finger_joint"])
            span = g["closed_rad"] - g["open_rad"]
            self.grip_act_lbl.setText("thực tế —" if fj is None else f"thực tế {100.0 * (fj - g['open_rad']) / span:.0f}%")

        rgb, depth, counts = self.ros.take_images()
        if rgb is not None:
            self._show_image(self.rgb_view, rgb)
        if depth is not None:
            self._show_image(self.depth_view, depth)
        f = self._img_fps
        now = time.monotonic()
        if now - f[2] >= 1.0:
            f[3] = (counts[0] - f[0]) / (now - f[2])
            f[4] = (counts[1] - f[1]) / (now - f[2])
            f[0], f[1], f[2] = counts[0], counts[1], now
            self.rgb_info.setText(f"Màu: {ARGS.rgb_topic}  ({f[3]:.0f} Hz)")
            self.depth_info.setText(f"Độ sâu: {ARGS.depth_topic}  ({f[4]:.0f} Hz, 0.1–3 m)")

        sim_t = self.ros.sim_time()
        self.status.setText(
            f"Chế độ: {self.ctrl.mode}   |   sim time: {'—' if sim_t is None else f'{sim_t:.2f} s'}   |   "
            f"RMW: {os.environ.get('RMW_IMPLEMENTATION', 'mặc định')}   |   lệnh → {ARGS.joint_command_topic}"
        )
        self.err.setText(self.ctrl.error)

    def _show_image(self, label, msg):
        img = image_to_qimage(msg)
        if img is None:
            label.setText(f"Encoding chưa hỗ trợ: {msg.encoding}")
            return
        pix = QtGui.QPixmap.fromImage(img).scaled(
            label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(pix)


# -------------------------------------------------------------------- main

ARGS = None


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint-command-topic", default="/joint_command")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_rect_raw")
    ARGS, ros_args = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0], *ros_args])
    ros = RosBridge(ARGS)

    def spin():
        try:
            rclpy.spin(ros)
        except ExternalShutdownException:
            pass

    threading.Thread(target=spin, daemon=True).start()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Panel(ros)
    win.show()
    code = app.exec()
    ros.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
