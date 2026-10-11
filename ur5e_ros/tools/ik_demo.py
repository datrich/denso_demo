"""Demo IK: Ready -> hinh vuong 15 cm (MoveL) -> ha/nang 10 cm -> Ready.

Dung chinh MotionController cua panel (IK giai tich cho MoveJ, DLS cho MoveL).
Chay khi Isaac Sim dang Play:  pixi run ik-demo   (them "-- --speed 0.5" de nhanh hon)
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panel"))
import ur5e_kinematics as kin  # noqa: E402
from ur5e_control import RosBridge, MotionController  # noqa: E402

READY = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])


def run_until_idle(ctrl, ros, timeout=30.0):
    dt = 1.0 / 50.0
    end = time.monotonic() + timeout
    while ctrl.mode != "idle" and time.monotonic() < end:
        ctrl.tick(dt)
        time.sleep(dt)
    time.sleep(0.6)  # cho sim bam kip
    if ctrl.error:
        print(f"   ! {ctrl.error}")
    return ros.joint_state()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.3, help="0..1, giong thanh Toc do cua panel")
    ap.add_argument("--side", type=float, default=0.15, help="canh hinh vuong (m)")
    args, _ = ap.parse_known_args()

    rclpy.init()
    ros = RosBridge(
        argparse.Namespace(
            joint_command_topic="/joint_command",
            joint_states_topic="/joint_states",
            rgb_topic="/camera/color/image_raw",
            depth_topic="/camera/depth/image_rect_raw",
        )
    )
    threading.Thread(target=lambda: rclpy.spin(ros), daemon=True).start()
    ctrl = MotionController(ros)
    ctrl.speed = args.speed

    t0 = time.monotonic()
    while ros.joint_state()[0] is None:
        if time.monotonic() - t0 > 20:
            print("Khong nhan duoc /joint_states - Isaac Sim da Play chua?")
            return 1
        time.sleep(0.1)

    print("1) MoveJ -> Ready (huong xuong)")
    ctrl.move_joint(READY, ros.joint_state()[0])
    q = run_until_idle(ctrl, ros)
    start = kin.fk(q)
    print(f"   TCP = {np.round(start[:3, 3] * 1000, 1)} mm")

    s = args.side
    corners = [(s, 0, 0), (s, s, 0), (0, s, 0), (0, 0, 0)]
    errors = []
    print(f"2) MoveL hinh vuong {s * 100:.0f} cm trong mat phang XY (giu nguyen huong)")
    for i, (dx, dy, dz) in enumerate(corners, 1):
        target = start.copy()
        target[:3, 3] += [dx, dy, dz]
        ctrl.move_pose_linear(target, ros.joint_state()[0])
        q = run_until_idle(ctrl, ros)
        err = np.linalg.norm(kin.fk(q)[:3, 3] - target[:3, 3]) * 1000
        errors.append(err)
        print(f"   goc {i}: sai {err:.2f} mm")

    print("3) MoveL ha 10 cm roi nang lai")
    for dz in (-0.10, 0.0):
        target = start.copy()
        target[2, 3] += dz
        ctrl.move_pose_linear(target, ros.joint_state()[0])
        q = run_until_idle(ctrl, ros)
        err = np.linalg.norm(kin.fk(q)[:3, 3] - target[:3, 3]) * 1000
        errors.append(err)
        print(f"   z {dz * 100:+.0f} cm: sai {err:.2f} mm")

    rot_err = math.degrees(kin.rotation_angle_between(kin.fk(q), start))
    print(f"\nXong. Sai so vi tri lon nhat {max(errors):.2f} mm, lech huong cuoi {rot_err:.2f} deg")
    ros.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
