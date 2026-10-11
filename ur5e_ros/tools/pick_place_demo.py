"""Demo pick & place 5 vat BOP HOPE bang UR5e + Robotiq 2F-85 (chay khi Isaac Sim dang Play).

    pixi run pick-place                 # gap tung vat o hang xa, dat ngay ngan vao hang gan robot
    pixi run pick-place -- --only Yogurt Butter

Voi moi vat: doc pose tu /tf -> chon truc hep nhat theo phuong ngang lam huong kep -> tiep can
tu tren xuong (MoveJ + MoveL) -> dong gripper -> nang -> xoay cho truc hep doc theo X (xep ngay ngan)
-> dat vao o dich -> nha -> kiem tra vi tri / dung thang tu /tf.
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panel"))
import ur5e_kinematics as kin  # noqa: E402
from ur5e_control import MotionController, RosBridge, load_gripper_config  # noqa: E402

OBJECTS_FILE = Path(__file__).resolve().parents[1] / "config" / "objects.json"
# do cao kep tinh tu profile be rong cua mesh (tools/bop_to_usd.py)
HOPE_MANIFEST = Path(__file__).resolve().parents[2] / "datasets" / "bop_hope" / "usd" / "objects.json"
READY = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
MAX_WIDTH = 0.075   # 2F-85 mo 85 mm, chua lai khe ho
GRASP_DEPTH = 0.03  # TCP thap hon dinh vat 3 cm
MIN_TCP_ABOVE_TABLE = 0.025  # dau ngon khong cham ban
TRANSIT_ABOVE_TABLE = 0.26
PLACE_GAP = 0.004   # tha vat cach mat ban 4 mm


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tool_rot(closing_axis, prev_y):
    """Huong TCP: z chi xuong, y (huong kep cua 2F-85) doc theo truc hep cua vat."""
    y = np.array([closing_axis[0], closing_axis[1], 0.0])
    y /= np.linalg.norm(y)
    if prev_y is not None and np.dot(y, prev_y) < 0:
        y = -y  # gripper doi xung -> chon chieu it phai xoay co tay
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


def pose(p, r):
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = p
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--only", nargs="*", help="chi gap cac vat nay")
    args, _ = ap.parse_known_args()

    grip = load_gripper_config()
    if not grip or not OBJECTS_FILE.exists():
        print("Can scene co gripper 2F-85 va vat HOPE (pixi run sim, UR5E_OBJECTS=hope)")
        return 1
    scene = json.loads(OBJECTS_FILE.read_text(encoding="utf-8"))
    table_z = scene["table_z"]
    objects = [o for o in scene["objects"] if not args.only or o["name"] in args.only]
    grasp_h, tapered = {}, {}
    if HOPE_MANIFEST.exists():
        for m in json.loads(HOPE_MANIFEST.read_text(encoding="utf-8"))["objects"]:
            grasp_h[m["name"]] = m.get("grasp_height_m")
            tapered[m["name"]] = m.get("tapered", False)

    rclpy.init()
    ros = RosBridge(argparse.Namespace(
        joint_command_topic="/joint_command", joint_states_topic="/joint_states",
        rgb_topic="/camera/color/image_raw", depth_topic="/camera/depth/image_rect_raw"))
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, ros)
    threading.Thread(target=lambda: rclpy.spin(ros), daemon=True).start()
    ctrl = MotionController(ros)
    ctrl.speed = args.speed
    ctrl.tcp_z = grip["tcp_offset_m"][2]
    fj = grip["finger_joint"]

    def run(timeout=30.0):
        dt = 1.0 / 50.0
        end = time.monotonic() + timeout
        while ctrl.mode != "idle" and time.monotonic() < end:
            ctrl.tick(dt)
            time.sleep(dt)
        time.sleep(0.4)
        err = ctrl.error
        ctrl.error = ""
        return err

    def gripper(rad, wait=1.5):
        ros.send_command([rad], names=[fj])
        time.sleep(wait)
        return ros.joint_value(fj)

    def obj_pose(frame):
        t = tf_buffer.lookup_transform("base_link", frame, Time(), timeout=Duration(seconds=2.0))
        p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        r = quat_to_R([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
        return p, r

    def tcp():
        return kin.fk(ros.joint_state()[0], ctrl.tcp_z)

    t0 = time.monotonic()
    while ros.joint_state()[0] is None or ros.joint_value(fj) is None:
        if time.monotonic() - t0 > 20:
            print("Khong nhan duoc /joint_states - Isaac Sim da Play chua?")
            return 1
        time.sleep(0.1)

    print(f"Gripper 2F-85, TCP {ctrl.tcp_z * 1000:.1f} mm, mat ban z = {table_z * 1000:.1f} mm (base_link)")
    gripper(grip["open_rad"], wait=0.5)
    ctrl.move_joint(READY, ros.joint_state()[0])
    run()

    results = []
    for o in objects:
        name, (sx, sy, sz) = o["name"], o["size_m"]
        print(f"\n=== {name} ({sx * 1000:.0f} x {sy * 1000:.0f} x {sz * 1000:.0f} mm) ===")
        try:
            p, r = obj_pose(o["frame"])
        except Exception as exc:  # noqa: BLE001
            print(f"   ! khong doc duoc /tf {o['frame']}: {exc}")
            results.append((name, False, "khong co TF"))
            continue
        upright = r[2, 2]
        width = min(sx, sy)
        if upright < 0.95 or width > MAX_WIDTH:
            print(f"   ! bo qua: dung thang={upright:.2f}, be rong kep={width * 1000:.0f} mm")
            results.append((name, False, "khong gap duoc"))
            continue
        axis = r[:, 0] if sx <= sy else r[:, 1]
        top, bottom = p[2] + sz / 2, p[2] - sz / 2
        gh = grasp_h.get(name)
        grasp_z = bottom + gh if gh else top - GRASP_DEPTH  # cho vat rong nhat / gan trong tam
        grasp_z = min(max(grasp_z, table_z + MIN_TCP_ABOVE_TABLE), top - (0.008 if tapered.get(name) else 0.012))
        ctrl.speed = min(args.speed, 0.25) if tapered.get(name) else args.speed  # vat thon: van chuyen cham
        hold_below = grasp_z - (p[2] - sz / 2)  # TCP cach day vat
        transit_z = table_z + max(TRANSIT_ABOVE_TABLE, hold_below + 0.10)
        r_pick = tool_rot(axis, tcp()[:3, 1])
        yaw = math.degrees(math.atan2(axis[1], axis[0]))
        print(f"   vat tai {np.round(p * 1000, 1)} mm, truc kep {yaw:+.0f} deg, be rong {width * 1000:.0f} mm, TCP gap z={grasp_z * 1000:.0f} mm")

        gripper(grip["open_rad"], wait=0.3)
        ctrl.move_pose_joint(pose([p[0], p[1], transit_z], r_pick), ros.joint_state()[0])
        e1 = run()
        ctrl.move_pose_linear(pose([p[0], p[1], grasp_z], r_pick), ros.joint_state()[0])
        e2 = run()
        f = gripper(grip["closed_rad"], wait=1.8)
        holding = math.degrees(f) < math.degrees(grip["closed_rad"]) - 3.0
        print(f"   kep: ngon dung o {math.degrees(f):.1f} deg -> {'dang giu vat' if holding else 'KHONG kep duoc'}" + (f" ({e1 or e2})" if e1 or e2 else ""))
        ctrl.move_pose_linear(pose([p[0], p[1], transit_z], r_pick), ros.joint_state()[0])
        run()
        q_lift, r_lift = obj_pose(o["frame"])
        lifted = q_lift[2] - p[2]
        print(f"   nang: vat len {lifted * 1000:.0f} mm, nghieng {math.degrees(math.acos(min(1.0, r_lift[2, 2]))):.1f} deg")
        if lifted < 0.05:
            gripper(grip["open_rad"], wait=0.5)
            results.append((name, False, "khong nang duoc"))
            continue

        # dat: truc hep cua vat doc theo X cua robot -> hang vat ngay ngan
        px, py = o["place_xy"]
        r_place = tool_rot(np.array([1.0, 0.0, 0.0]), tcp()[:3, 1])
        place_z = table_z + hold_below + PLACE_GAP
        ctrl.move_pose_joint(pose([px, py, transit_z], r_place), ros.joint_state()[0])
        run()
        tilt_transit = math.degrees(math.acos(min(1.0, obj_pose(o["frame"])[1][2, 2])))
        ctrl.move_pose_linear(pose([px, py, place_z], r_place), ros.joint_state()[0])
        run()
        tilt_down = math.degrees(math.acos(min(1.0, obj_pose(o["frame"])[1][2, 2])))
        print(f"   van chuyen: nghieng {tilt_transit:.1f} deg, ha xuong (truoc khi nha): {tilt_down:.1f} deg")
        gripper(grip["open_rad"], wait=1.0)
        ctrl.move_pose_linear(pose([px, py, transit_z], r_place), ros.joint_state()[0])
        run()
        time.sleep(0.5)

        q, rq = obj_pose(o["frame"])
        xy_err = np.linalg.norm(q[:2] - [px, py]) * 1000
        tilt = math.degrees(math.acos(min(1.0, rq[2, 2])))
        a2 = rq[:, 0] if sx <= sy else rq[:, 1]
        align = math.degrees(math.acos(min(1.0, abs(a2[0]) / max(1e-9, math.hypot(a2[0], a2[1])))))
        on_table = abs((q[2] - sz / 2) - table_z) < 0.01
        ok = xy_err < 30 and tilt < 10 and on_table
        print(f"   dat: lech {xy_err:.1f} mm, nghieng {tilt:.1f} deg, lech huong {align:.1f} deg, tren ban={on_table} -> {'OK' if ok else 'LOI'}")
        results.append((name, ok, f"lech {xy_err:.0f} mm, nghieng {tilt:.0f} deg"))

    ctrl.move_joint(READY, ros.joint_state()[0])
    run()
    print("\n=== Ket qua ===")
    for name, ok, msg in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:17s} {msg}")
    n_ok = sum(ok for _, ok, _ in results)
    print(f"{n_ok}/{len(results)} vat duoc gap - dat thanh cong")
    ros.destroy_node()
    rclpy.try_shutdown()
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
