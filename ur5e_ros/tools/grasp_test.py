"""Thu gap TestCube bang Robotiq 2F-85 (chay khi Isaac Sim dang Play).

    pixi run grasp-test

Dung MotionController cua panel (MoveJ/MoveL) va vi tri khoi tu /tf (frame TestCube).
TCP = grip_frame cua 2F-85, doc tu config/gripper.json do scene ghi ra.
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panel"))
import ur5e_kinematics as kin  # noqa: E402
from ur5e_control import MotionController, RosBridge, load_gripper_config  # noqa: E402

import tf2_ros  # noqa: E402

READY = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
APPROACH = 0.12  # m tren khoi
LIFT = 0.15
GRASP_ABOVE_CENTER = 0.008  # nang TCP 8 mm so voi tam khoi de dau ngon khong cham ban


def report(results, ok, text):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {text}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.3)
    args, _ = ap.parse_known_args()

    cfg = load_gripper_config()
    if not cfg:
        print("Khong co gripper trong config/gripper.json - chay 'pixi run sim' voi UR5E_GRIPPER=2f85 truoc")
        return 1
    if "test_cube" not in cfg:
        print("Scene dang dung vat BOP HOPE -> dung 'pixi run pick-place' (hoac chay sim voi UR5E_OBJECTS=cube)")
        return 1

    rclpy.init()
    ros = RosBridge(argparse.Namespace(
        joint_command_topic="/joint_command", joint_states_topic="/joint_states",
        rgb_topic="/camera/color/image_raw", depth_topic="/camera/depth/image_rect_raw"))
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, ros)
    threading.Thread(target=lambda: rclpy.spin(ros), daemon=True).start()
    ctrl = MotionController(ros)
    ctrl.speed = args.speed
    ctrl.tcp_z = cfg["tcp_offset_m"][2]
    fj = cfg["finger_joint"]
    cube_frame = cfg["test_cube"]["frame"]
    results = []

    def run(timeout=30.0):
        dt = 1.0 / 50.0
        end = time.monotonic() + timeout
        while ctrl.mode != "idle" and time.monotonic() < end:
            ctrl.tick(dt)
            time.sleep(dt)
        time.sleep(0.5)
        if ctrl.error:
            print(f"   ! {ctrl.error}")
            ctrl.error = ""

    def gripper(rad, wait=1.5):
        ros.send_command([rad], names=[fj])
        time.sleep(wait)
        return ros.joint_value(fj)

    def cube():
        t = tf_buffer.lookup_transform("base_link", cube_frame, Time(), timeout=Duration(seconds=2.0))
        return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])

    def tcp_pose(p, rot):
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = p
        return t

    t0 = time.monotonic()
    while ros.joint_state()[0] is None or ros.joint_value(fj) is None:
        if time.monotonic() - t0 > 20:
            print("Khong nhan duoc /joint_states co finger_joint - Isaac Sim da Play chua?")
            return 1
        time.sleep(0.1)
    report(results, True, f"/joint_states co {fj} = {math.degrees(ros.joint_value(fj)):.1f} deg, TCP offset {ctrl.tcp_z * 1000:.1f} mm")

    try:
        c0 = cube()
    except Exception as exc:  # noqa: BLE001
        report(results, False, f"khong doc duoc /tf base_link -> {cube_frame}: {exc}")
        return 1
    print(f"   khoi tai {np.round(c0 * 1000, 1)} mm (khung base_link)")

    print("1) Mo gripper, MoveJ -> Ready")
    f_open = gripper(cfg["open_rad"])
    ctrl.move_joint(READY, ros.joint_state()[0])
    run()
    down = kin.fk(READY, ctrl.tcp_z)[:3, :3]
    report(results, abs(math.degrees(f_open or 0)) < 2.0 and down[2, 2] < -0.99,
           f"gripper mo ({math.degrees(f_open):.1f} deg), truc TCP huong xuong (z.z = {down[2, 2]:.3f})")

    print("2) Tiep can phia tren khoi, MoveL xuong")
    ctrl.move_pose_joint(tcp_pose(c0 + [0, 0, APPROACH], down), ros.joint_state()[0])
    run()
    grasp_p = c0 + [0, 0, GRASP_ABOVE_CENTER]
    ctrl.move_pose_linear(tcp_pose(grasp_p, down), ros.joint_state()[0])
    run()
    err = np.linalg.norm(kin.fk(ros.joint_state()[0], ctrl.tcp_z)[:3, 3] - grasp_p) * 1000
    report(results, err < 3.0, f"TCP toi vi tri gap, sai {err:.1f} mm")
    moved = np.linalg.norm(cube()[:2] - c0[:2]) * 1000
    report(results, moved < 5.0, f"khoi chua bi xo lech khi tiep can ({moved:.1f} mm)")

    print("3) Dong gripper")
    f_closed = gripper(cfg["closed_rad"], wait=2.0)
    deg = math.degrees(f_closed)
    report(results, 5.0 < deg < math.degrees(cfg["closed_rad"]) - 3.0,
           f"ngon dung o {deg:.1f} deg (< {math.degrees(cfg['closed_rad']):.1f} deg lenh dong) -> dang kep vat")

    print("4) Nang 15 cm")
    ctrl.move_pose_linear(tcp_pose(grasp_p + [0, 0, LIFT], down), ros.joint_state()[0])
    run()
    time.sleep(1.0)
    c1 = cube()
    report(results, c1[2] - c0[2] > LIFT - 0.02, f"khoi duoc nang {1000 * (c1[2] - c0[2]):.1f} mm (lenh {LIFT * 1000:.0f} mm)")
    rel1 = c1 - kin.fk(ros.joint_state()[0], ctrl.tcp_z)[:3, 3]  # khoi so voi TCP truoc khi di ngang

    print("5) Di ngang 10 cm roi quay lai (kiem tra truot)")
    ctrl.move_pose_linear(tcp_pose(grasp_p + [0, 0.10, LIFT], down), ros.joint_state()[0])
    run()
    time.sleep(0.5)
    rel2 = cube() - kin.fk(ros.joint_state()[0], ctrl.tcp_z)[:3, 3]
    slip = np.linalg.norm(rel2 - rel1) * 1000
    report(results, slip < 10.0, f"khoi di theo gripper, truot {slip:.1f} mm")
    ctrl.move_pose_linear(tcp_pose(grasp_p + [0, 0, LIFT], down), ros.joint_state()[0])
    run()

    print("6) Dat lai, mo gripper, rut len")
    ctrl.move_pose_linear(tcp_pose(grasp_p + [0, 0, 0.003], down), ros.joint_state()[0])
    run()
    gripper(cfg["open_rad"])
    ctrl.move_pose_linear(tcp_pose(grasp_p + [0, 0, APPROACH], down), ros.joint_state()[0])
    run()
    time.sleep(1.0)
    c3 = cube()
    report(results, abs(c3[2] - c0[2]) < 0.01, f"khoi nam lai tren ban (lech z {1000 * (c3[2] - c0[2]):.1f} mm)")

    print(f"\n{sum(results)}/{len(results)} PASS")
    ros.destroy_node()
    rclpy.try_shutdown()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
