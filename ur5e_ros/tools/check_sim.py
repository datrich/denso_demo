"""Kiem tra nhanh ket noi Isaac Sim <-> ROS 2 (chay khi sim dang Play).

    pixi run check-sim            # chi doc
    pixi run check-sim -- --move  # them buoc cho khop shoulder_pan quay +/-15 do roi ve cho cu

Kiem tra: /joint_states, /clock, anh mau + do sau, FK cua panel so voi /tf, (tuy chon) bam lenh khop.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panel"))
import ur5e_kinematics as kin  # noqa: E402

try:
    import tf2_ros
except ImportError:  # pragma: no cover
    tf2_ros = None


class Checker(Node):
    def __init__(self):
        super().__init__("ur5e_sim_checker")
        self.q = None
        self.clock = []
        self.rgb = None
        self.depth = None
        self.create_subscription(JointState, "/joint_states", self._js, 10)
        self.create_subscription(Clock, "/clock", lambda m: self.clock.append(m.clock.sec + m.clock.nanosec * 1e-9), 10)
        self.create_subscription(Image, "/camera/color/image_raw", lambda m: setattr(self, "rgb", m), qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/depth/image_rect_raw", lambda m: setattr(self, "depth", m), qos_profile_sensor_data)
        self.cmd = self.create_publisher(JointState, "/joint_command", 10)
        if tf2_ros:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _js(self, msg):
        d = dict(zip(msg.name, msg.position))
        self.all_joints = d
        if all(n in d for n in kin.JOINT_NAMES):
            self.q = np.array([d[n] for n in kin.JOINT_NAMES])

    def spin_for(self, seconds, until=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if until and until():
                return True
        return until() if until else True

    def send(self, q):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(kin.JOINT_NAMES)
        m.position = [float(v) for v in q]
        self.cmd.publish(m)


def report(ok, text):
    print(f"[{'PASS' if ok else 'FAIL'}] {text}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true", help="thu cho robot chuyen dong")
    ap.add_argument("--save-rgb", default="", help="luu 1 anh mau ra file PNG")
    args, _ = ap.parse_known_args()

    rclpy.init()
    n = Checker()
    results = []

    ok = n.spin_for(60, lambda: n.q is not None)
    results.append(report(ok, "/joint_states co du 6 khop UR5e" + (f": {np.round(np.degrees(n.q), 1)} deg" if ok else "")))
    if not ok:
        print("-> Kiem tra: Zenoh router chay truoc Isaac Sim? Sim da Play?")
        return 1

    n.spin_for(1.5)
    adv = len(n.clock) > 5 and n.clock[-1] > n.clock[0]
    results.append(report(adv, f"/clock dang chay ({len(n.clock)} msg / 1.5 s)"))

    n.spin_for(15, lambda: n.rgb is not None and n.depth is not None)
    results.append(report(n.rgb is not None, "anh mau" + (f" {n.rgb.width}x{n.rgb.height} {n.rgb.encoding}" if n.rgb else "")))
    if n.depth is not None:
        d = np.frombuffer(bytes(n.depth.data), np.float32) if n.depth.encoding == "32FC1" else None
        stat = f", do sau {np.nanmin(d[np.isfinite(d)]):.2f}..{np.nanmax(d[np.isfinite(d)]):.2f} m" if d is not None and np.isfinite(d).any() else ""
        results.append(report(True, f"anh do sau {n.depth.width}x{n.depth.height} {n.depth.encoding}{stat}"))
    else:
        results.append(report(False, "anh do sau"))
    if args.save_rgb and n.rgb is not None and n.rgb.encoding in ("rgb8", "rgba8"):
        from PySide6 import QtGui

        ch = 3 if n.rgb.encoding == "rgb8" else 4
        arr = np.frombuffer(bytes(n.rgb.data), np.uint8).reshape(n.rgb.height, n.rgb.step)[:, : n.rgb.width * ch]
        arr = np.ascontiguousarray(arr.reshape(n.rgb.height, n.rgb.width, ch)[:, :, :3])
        QtGui.QImage(arr.data, n.rgb.width, n.rgb.height, 3 * n.rgb.width, QtGui.QImage.Format.Format_RGB888).save(args.save_rgb)
        print(f"      da luu anh mau: {args.save_rgb}")

    if tf2_ros:
        n.spin_for(2.0)
        try:
            t = n.tf_buffer.lookup_transform("base_link", "wrist_3_link", Time(), timeout=Duration(seconds=2.0))
            p_tf = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
            p_fk = kin.fk(n.q)[:3, 3]
            err = np.linalg.norm(p_tf - p_fk) * 1000
            results.append(report(err < 2.0, f"FK panel vs /tf (base_link->wrist_3_link): sai {err:.2f} mm  tf={np.round(p_tf, 4)} fk={np.round(p_fk, 4)}"))
            qx, qy, qz, qw = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
            r_tf = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            rot_err = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r_tf.T @ kin.fk(n.q)[:3, :3]) - 1) / 2))))
            results.append(report(rot_err < 0.5, f"huong FK (tool0) vs /tf wrist_3_link: lech {rot_err:.3f} deg"))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] khong doc duoc /tf base_link->wrist_3_link: {exc}")
            print("       frame co tren /tf:", n.tf_buffer.all_frames_as_string()[:600])

    from ur5e_panel import load_gripper_config  # noqa: E402

    grip = load_gripper_config()
    if grip:
        fj = grip["finger_joint"]
        gj = sorted(k for k in getattr(n, "all_joints", {}) if k not in kin.JOINT_NAMES)
        results.append(report(fj in gj, f"/joint_states co khop gripper ({len(gj)}): {gj}"))
        if tf2_ros:
            frames = n.tf_buffer.all_frames_as_string()
            results.append(report("robotiq_base_link" in frames, "frame TF robotiq_base_link (khong trung base_link cua UR5e)"))
            try:
                t = n.tf_buffer.lookup_transform("wrist_3_link", "robotiq_base_link", Time(), timeout=Duration(seconds=2.0))
                off = np.linalg.norm([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]) * 1000
                results.append(report(off < 1.0, f"gripper gan chat vao wrist_3_link (lech {off:.2f} mm)"))
            except Exception as exc:  # noqa: BLE001
                results.append(report(False, f"khong doc duoc wrist_3_link -> robotiq_base_link: {exc}"))

    if args.move:
        q0 = n.q.copy()
        for target_off in (math.radians(15), 0.0):
            start = n.q.copy()
            goal = q0.copy()
            goal[0] += target_off
            steps = 100
            for i in range(1, steps + 1):  # 2 s @ 50 Hz
                n.send(start + (goal - start) * i / steps)
                n.spin_for(0.02)
            n.spin_for(1.0)
            err = math.degrees(abs(n.q[0] - goal[0]))
            others = math.degrees(np.max(np.abs(n.q[1:] - q0[1:])))
            results.append(report(err < 1.0 and others < 2.0, f"bam lenh shoulder_pan -> {math.degrees(goal[0]):.1f} deg: sai {err:.2f} deg, khop khac lech toi da {others:.2f} deg"))

    print(f"\n{sum(results)}/{len(results)} PASS")
    n.destroy_node()
    rclpy.try_shutdown()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
