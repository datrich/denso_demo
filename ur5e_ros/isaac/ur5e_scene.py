"""Dung scene UR5e + ban + camera D435 va Action Graph ROS 2 trong Isaac Sim 6.0.1.

Chay qua pixi (Zenoh router phai chay truoc):
    pixi run zenoh      # terminal 1
    pixi run sim        # terminal 2 -> isaac-sim.bat --exec isaac\\ur5e_scene.py

Scene goc: ../ur5eworkspace-env/ur5etunewithtable.usd (repo cua Hung) duoc nap lam *sublayer*,
moi chinh sua nam o root layer an danh nen file goc khong bi ghi de.
- Neu co du ur_1/ va d435_import/ trong repo -> dung robot UR5e + Robotiq 3F + D435 that.
- Neu thieu (hien tai) -> tat /ur, thay bang UR5e chuan cua Isaac Sim o cung vi tri, ap gain
  da tune cua Hung, gan camera D435 tam o dung vi tri d435_mount, gan gripper Robotiq 2F-85
  (robotiq/isaacsim_assets, PhysX parallel_grip - xem robotiq_2f85.py).
- Luon them collider cho mat ban + vat de gap: 5 vat BOP HOPE (mac dinh) hoac khoi lap phuong 4 cm.

ROS 2 (mac dinh):
    /joint_states   (pub)  sensor_msgs/JointState (6 khop UR5e + cac khop gripper)
    /joint_command  (sub)  sensor_msgs/JointState  -> Articulation Controller
                           (gripper: ten "finger_joint", 0 = mo, 0.8 rad = dong)
    /clock          (pub)
    /tf             (pub)  cay link cua robot + cac vat tren ban (Butter, ..., hoac TestCube)
    /camera/color/image_raw, /camera/depth/image_rect_raw, /camera/color/camera_info (pub)

Bien moi truong tuy chon:
    UR5E_SCENE_USD       duong dan scene khac
    UR5E_AUTOPLAY=0      khong tu bam Play
    UR5E_NO_CAMERA=1     khong phat camera len ROS 2 (chan doan loi GPU)
    UR5E_GRIPPER=none    khong gan 2F-85 (mac dinh: 2f85)
    UR5E_ROBOTIQ_ASSETS  thu muc clone robotiq/isaacsim_assets (mac dinh ../robotiq_isaacsim_assets)
    UR5E_OBJECTS=cube    dung khoi lap phuong thay cho 5 vat BOP HOPE (mac dinh: hope)
"""

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import carb
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd
import usdrt.Sdf
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

_THIS = Path(globals().get("__file__") or Path.cwd() / "isaac" / "ur5e_scene.py").resolve()
sys.path.insert(0, str(_THIS.parent))
import robotiq_2f85 as rq  # noqa: E402

WORKSPACE = _THIS.parents[2]  # .../VLA_robot_arm
SCENE_USD = Path(os.environ.get("UR5E_SCENE_USD", WORKSPACE / "ur5eworkspace-env" / "ur5etunewithtable.usd"))
UR5E_ASSET = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
ASSETS_FALLBACK = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
ROBOTIQ_ASSETS = Path(os.environ.get("UR5E_ROBOTIQ_ASSETS", WORKSPACE / "robotiq_isaacsim_assets"))
GRIPPER_2F85_USD = ROBOTIQ_ASSETS / "grippers" / "Robotiq_2F_85" / "Robotiq_2F_85.usda"
GRIPPER = os.environ.get("UR5E_GRIPPER", "2f85").strip().lower()
GRIPPER_CONFIG = _THIS.parents[1] / "config" / "gripper.json"  # panel + tools doc file nay

# Vat ly khi co gripper (theo scene demo cua Robotiq: ur5robot_with_2F-85.usda)
PHYSICS_HZ = 120
# demo dung 255/8; PhysX TGS (Isaac 6) canh bao khi > 4 velocity iterations -> 4 (guide §3.1 chi can >= 2)
ARTICULATION_ITERATIONS = (255, 4)
CAM_TICK_HZ = 30.0  # omni:sensor:tickRate (thay cho frameSkipCount da deprecated)

# Camera toan canh de quay video (UR5E_DEMO_CAM=1): /demo_cam/rgb 1280x720
DEMO_CAM = os.environ.get("UR5E_DEMO_CAM", "0") == "1"
DEMO_CAM_PATH = "/DemoCamera"
DEMO_CAM_W, DEMO_CAM_H = 1280, 720
DEMO_CAM_EYE = (0.95, 1.05, 1.40)       # toa do the gioi (m)
DEMO_CAM_TARGET = (-0.05, -0.08, 0.80)  # giua ban, hang vat gap / dat

# Vat tren ban: "hope" = 5 vat BOP HOPE (tools/bop_to_usd.py), "cube" = khoi lap phuong thu
OBJECTS = os.environ.get("UR5E_OBJECTS", "hope").strip().lower()
HOPE_ROOT = WORKSPACE / "datasets" / "bop_hope"
HOPE_MANIFEST = HOPE_ROOT / "usd" / "objects.json"
OBJECTS_CONFIG = _THIS.parents[1] / "config" / "objects.json"  # cho tools/pick_place_demo.py
# Hang vat can gap / hang dich (khung base_link), moi vat 1 goc xoay khac nhau de thu canh huong gap
PICK_X, PLACE_X = 0.52, 0.30
SLOT_Y = (-0.32, -0.18, -0.04, 0.10, 0.24)
PICK_YAW_DEG = (0.0, 35.0, 90.0, -30.0, 15.0)

# Khoi thu gap (guide §3.1): 0.1 kg, contactOffset 2 mm, restOffset 0.5 mm, damping 0.1, vel iters 4
CUBE_PATH = "/TestObjects/TestCube"
CUBE_SIZE = 0.04
CUBE_MASS = 0.1
CUBE_XY_IN_BASE = (0.45, 0.10)  # vi tri tren ban, theo khung base_link cua robot

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Gia tri du phong neu khong doc duoc tu scene cua Hung
DEFAULT_ROBOT_POS = (-0.37301266253241205, -0.04873248472807018, 0.7405869357032596)
# Vi tri d435_mount so voi wrist_3_link (lay tu scene cua Hung)
D435_MOUNT_T = (0.004051560494559514, 0.10352482402905805, 0.042623558882600276)
D435_MOUNT_Q = (0.6345022587886923, -0.30932993825419125, -0.3310448163296172, -0.6262038026657102)  # w, x, y, z
# camera_bottom_screw_frame -> camera_color_frame (realsense2_description, D435)
D435_COLOR_OFFSET = (0.0, 0.0325, 0.0125)
CAM_W, CAM_H, CAM_HFOV_DEG = 640, 480, 69.0  # D435 color

TAG = "[ur5e_scene]"


def log(msg):
    carb.log_warn(f"{TAG} {msg}")
    print(f"{TAG} {msg}", flush=True)


async def _frames(n=1):
    app = omni.kit.app.get_app()
    for _ in range(n):
        await app.next_update_async()


def _assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path

        root = get_assets_root_path()
        if root:
            return root
    except Exception as exc:  # noqa: BLE001
        log(f"get_assets_root_path loi ({exc}), dung S3 mac dinh")
    return ASSETS_FALLBACK


def _read_hung_layer():
    """Doc vi tri robot + gain da tune tu file goc (khong can robot asset)."""
    info = {"pos": DEFAULT_ROBOT_POS, "drives": {}}
    layer = Sdf.Layer.FindOrOpen(str(SCENE_USD))
    if layer is None:
        return info
    ur = layer.GetPrimAtPath("/ur")
    if ur and "xformOp:translate" in ur.attributes:
        info["pos"] = tuple(ur.attributes["xformOp:translate"].default)
    for name in JOINT_NAMES:
        spec = layer.GetPrimAtPath(f"/ur/Physics/{name}")
        if not spec:
            continue
        info["drives"][name] = {
            a.name: a.default for a in spec.attributes if a.name.startswith("drive:angular:physics:")
        }
    return info


def _find(root_prim, predicate):
    for prim in Usd.PrimRange(root_prim):
        if predicate(prim):
            return prim
    return None


def _apply_drives(stage, robot_root, drives):
    """Ap stiffness/damping/maxForce/targetPosition va trang thai ban dau = home."""
    for name in JOINT_NAMES:
        joint = _find(stage.GetPrimAtPath(robot_root), lambda p, n=name: p.GetName() == n and p.IsA(UsdPhysics.RevoluteJoint))
        if joint is None:
            log(f"khong tim thay khop {name}")
            continue
        vals = drives.get(name, {})
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        if "drive:angular:physics:stiffness" in vals:
            drive.CreateStiffnessAttr().Set(float(vals["drive:angular:physics:stiffness"]))
        if "drive:angular:physics:damping" in vals:
            drive.CreateDampingAttr().Set(float(vals["drive:angular:physics:damping"]))
        if "drive:angular:physics:maxForce" in vals:
            drive.CreateMaxForceAttr().Set(float(vals["drive:angular:physics:maxForce"]))
        target = float(vals.get("drive:angular:physics:targetPosition", 0.0))
        drive.CreateTargetPositionAttr().Set(target)
        # trang thai ban dau = home de robot khong "vung" luc bam Play
        joint.AddAppliedSchema("PhysicsJointStateAPI:angular")
        joint.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(target)
        joint.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)


def _add_temp_d435(stage, wrist3_path):
    """Camera tam: dat dung vi tri d435_mount cua Hung, nhin doc +X cua mount (truc nhin cua D435)."""
    mount = UsdGeom.Xform.Define(stage, f"{wrist3_path}/d435_mount_temp")
    mount.ClearXformOpOrder()
    mount.AddTranslateOp().Set(Gf.Vec3d(*D435_MOUNT_T))
    w, x, y, z = D435_MOUNT_Q
    mount.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    color = UsdGeom.Xform.Define(stage, f"{mount.GetPath()}/camera_color_frame")
    color.AddTranslateOp().Set(Gf.Vec3d(*D435_COLOR_OFFSET))
    cam = UsdGeom.Camera.Define(stage, f"{color.GetPath()}/Camera")
    # Camera USD nhin -Z, +Y huong len. Can: -Z_cam = +X_mount, +Y_cam = +Z_mount
    rot = Gf.Matrix3d(0, -1, 0, 0, 0, 1, -1, 0, 0)  # hang = truc x, y, z cua camera trong khung mount
    cam.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rot.ExtractRotation().GetQuat()))
    h_ap = 20.955
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * CAM_H / CAM_W)
    cam.CreateFocalLengthAttr((h_ap / 2) / math.tan(math.radians(CAM_HFOV_DEG / 2)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    # than camera (hop nho) de nhin thay trong viewport
    body = UsdGeom.Cube.Define(stage, f"{mount.GetPath()}/body")
    body.CreateSizeAttr(1.0)
    body.AddTranslateOp().Set(Gf.Vec3d(-0.0125, 0.0175, 0.0125))
    body.AddScaleOp().Set(Gf.Vec3f(0.025, 0.09, 0.025))
    body.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.15, 0.17)])
    return str(cam.GetPath())


def _ensure_environment(stage):
    if not any(p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase) for p in stage.Traverse()):
        dome = UsdLux.DomeLight.Define(stage, "/Environment/DomeLight")
        dome.CreateIntensityAttr(800.0)
        sun = UsdLux.DistantLight.Define(stage, "/Environment/DistantLight")
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 20, 0))
    floor = UsdGeom.Cube.Define(stage, "/Environment/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
    floor.AddScaleOp().Set(Gf.Vec3f(8, 8, 0.01))
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.47)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())


def _set(prim, name, type_name, value):
    prim.CreateAttribute(name, type_name).Set(value)
    return prim.GetAttribute(name).Get()


def _setup_2f85(stage, robot_root, wrist3_path):
    """Gan 2F-85 + kiem tra cau hinh. Tra ve (info | None, checks)."""
    if not GRIPPER_2F85_USD.exists():
        return None, [(False, f"khong tim thay asset {GRIPPER_2F85_USD} (git clone robotiq/isaacsim_assets)")]
    info = rq.mount(stage, wrist3_path, GRIPPER_2F85_USD.as_posix(), f"{robot_root}/joints/robotiq_2f85_mount_joint")
    return info, rq.validate(stage, robot_root, info)


def _tune_physics(stage, articulation_path):
    """120 Hz + stabilization + solver iterations cua articulation (nhu demo Robotiq)."""
    checks = []
    scene = _find(stage.GetPseudoRoot(), lambda p: p.IsA(UsdPhysics.Scene))
    if scene is None:
        scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene").GetPrim()
    scene.AddAppliedSchema("PhysxSceneAPI")
    hz = _set(scene, "physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt, PHYSICS_HZ)
    stab = _set(scene, "physxScene:enableStabilization", Sdf.ValueTypeNames.Bool, True)
    # 1 robot + 255 solver iterations: PhysX CPU nhanh hon GPU nhieu (demo Robotiq cung tat GPU dynamics)
    gpu = _set(scene, "physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, False)
    bp = _set(scene, "physxScene:broadphaseType", Sdf.ValueTypeNames.Token, "MBP")
    checks.append((hz == PHYSICS_HZ and stab and gpu is False and bp == "MBP",
                   f"{scene.GetPath()}: {hz} Hz, enableStabilization={stab}, GPU dynamics={gpu}, broadphase={bp}"))
    art = stage.GetPrimAtPath(articulation_path)
    art.AddAppliedSchema("PhysxArticulationAPI")
    pos = _set(art, "physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int, ARTICULATION_ITERATIONS[0])
    vel = _set(art, "physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, ARTICULATION_ITERATIONS[1])
    checks.append(((pos, vel) == ARTICULATION_ITERATIONS, f"articulation {articulation_path}: solver iterations {pos}/{vel}"))
    return checks


def _add_table_collider(stage):
    """Ban tu S3 chi co hinh, khong co collider -> them hop va cham tinh o mat ban."""
    table = stage.GetPrimAtPath("/EastRural_Table")
    if not table or not table.IsActive():
        return None, None, [(False, "khong tim thay /EastRural_Table")]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(table).ComputeAlignedRange()
    top = rng.GetMax()[2]
    has_col = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(table, Usd.TraverseInstanceProxies()))
    if not has_col:
        box = UsdGeom.Cube.Define(stage, "/Environment/TableTopCollider")
        box.CreateSizeAttr(1.0)
        box.CreatePurposeAttr(UsdGeom.Tokens.guide)  # khong render, chi va cham
        size = rng.GetSize()
        center = rng.GetMidpoint()
        box.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], top - 0.02))
        box.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], 0.04))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    msg = f"mat ban z={top:.4f} m, XY [{rng.GetMin()[0]:.3f}..{rng.GetMax()[0]:.3f}] x [{rng.GetMin()[1]:.3f}..{rng.GetMax()[1]:.3f}]"
    return top, rng, [(True, msg + (" (co san collider)" if has_col else " -> them TableTopCollider"))]


def _add_test_cube(stage, robot_pos, top, rng):
    x = robot_pos[0] + CUBE_XY_IN_BASE[0]
    y = robot_pos[1] + CUBE_XY_IN_BASE[1]
    z = top + CUBE_SIZE / 2 + 0.002
    inside = rng.GetMin()[0] + 0.05 < x < rng.GetMax()[0] - 0.05 and rng.GetMin()[1] + 0.05 < y < rng.GetMax()[1] - 0.05

    mat = UsdShade.Material.Define(stage, "/TestObjects/CubePhysicsMaterial")
    pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pmat.CreateStaticFrictionAttr(1.0)
    pmat.CreateDynamicFrictionAttr(0.9)
    pmat.CreateRestitutionAttr(0.0)

    cube = UsdGeom.Cube.Define(stage, CUBE_PATH)
    cube.CreateSizeAttr(CUBE_SIZE)
    h = CUBE_SIZE / 2
    cube.CreateExtentAttr([Gf.Vec3f(-h, -h, -h), Gf.Vec3f(h, h, h)])
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.25, 0.15)])
    cube.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(CUBE_MASS)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    co = _set(prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, 0.002)
    ro = _set(prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, 0.0005)
    ld = _set(prim, "physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float, 0.1)
    vi = _set(prim, "physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 4)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    ok = (
        inside
        and math.isclose(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), CUBE_MASS, rel_tol=1e-5)  # float32
        and math.isclose(co, 0.002, rel_tol=1e-4)
        and math.isclose(ro, 0.0005, rel_tol=1e-4)
        and co > ro
        and vi == 4
    )
    return CUBE_PATH, [(ok, f"TestCube {CUBE_SIZE * 100:.0f} cm @({x:.3f}, {y:.3f}, {z:.3f}) tren ban={inside}, "
                            f"mass={CUBE_MASS}, contact/rest={co}/{ro}, damping={ld}, velIters={vi}")]


def _add_hope_objects(stage, robot_pos, top, rng):
    """Dat cac vat BOP HOPE (USD da co vat ly) thanh 1 hang tren ban; ghi config/objects.json."""
    if not HOPE_MANIFEST.exists():
        return [], [(False, f"khong tim thay {HOPE_MANIFEST} - chay 'pixi run bop-to-usd' truoc")]
    manifest = json.loads(HOPE_MANIFEST.read_text(encoding="utf-8"))["objects"][: len(SLOT_Y)]
    UsdGeom.Scope.Define(stage, "/Objects")
    checks, paths, cfg = [], [], []
    for o, sy, yaw in zip(manifest, SLOT_Y, PICK_YAW_DEG):
        path = f"/Objects/{o['name']}"
        x, y = robot_pos[0] + PICK_X, robot_pos[1] + sy
        z = top - o["min_m"][2] + 0.002  # goc model BOP o tam hop bao
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference((HOPE_ROOT / o["usd"]).as_posix())
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(x, y, z))
        h = math.radians(yaw) / 2
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(h), 0, 0, math.sin(h)))
        mesh = stage.GetPrimAtPath(f"{path}/mesh")
        approx = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() if mesh else None
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        inside = rng.GetMin()[0] + 0.05 < x < rng.GetMax()[0] - 0.05 and rng.GetMin()[1] + 0.05 < y < rng.GetMax()[1] - 0.05
        ok = (prim.HasAPI(UsdPhysics.RigidBodyAPI) and mesh and mesh.HasAPI(UsdPhysics.CollisionAPI)
              and approx == "convexHull" and mass and inside)
        checks.append((ok, f"{o['name']} (HOPE obj_{o['obj_id']:06d}) {[round(v * 1000) for v in o['size_m']]} mm, "
                           f"{mass} kg, {approx}, yaw {yaw:+.0f} deg, tren ban={inside}"))
        paths.append(path)
        cfg.append({"name": o["name"], "frame": o["name"], "obj_id": o["obj_id"], "size_m": o["size_m"],
                    "pick_xy": [PICK_X, sy], "pick_yaw_deg": yaw, "place_xy": [PLACE_X, sy]})
    OBJECTS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OBJECTS_CONFIG.write_text(json.dumps({"dataset": "BOP HOPE", "table_z": round(top - robot_pos[2], 5),
                                          "objects": cfg}, indent=2), encoding="utf-8")
    return paths, checks


def _asset_commit():
    try:
        out = subprocess.run(["git", "-C", str(ROBOTIQ_ASSETS), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _write_gripper_config(stage, info, wrist3_path):
    GRIPPER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if info is None:
        data = {"gripper": None}
    else:
        pos, _ = rq.grip_frame_in(stage, wrist3_path, info["gripper"])
        data = {
            "gripper": "robotiq_2f85",
            "physics_variant": rq.PHYSICS_VARIANT,
            "asset_repo": "https://github.com/robotiq/isaacsim_assets",
            "asset_commit": _asset_commit(),
            "finger_joint": rq.FINGER_JOINT,
            "open_rad": 0.0,
            "closed_rad": rq.FINGER_CLOSED_RAD,
            "limit_rad": math.radians(rq.FINGER_LIMIT_DEG),
            "tcp_offset_m": [round(v, 6) for v in pos],
        }
        if OBJECTS == "cube":
            data["test_cube"] = {"frame": CUBE_PATH.rsplit("/", 1)[-1], "size_m": CUBE_SIZE, "mass_kg": CUBE_MASS}
    GRIPPER_CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _build_joint_graph(articulation_path, extra_tf_prims=()):
    keys = og.Controller.Keys
    tf_targets = [usdrt.Sdf.Path(p) for p in (articulation_path, *extra_tf_prims)]
    og.Controller.edit(
        {"graph_path": "/ROS2/JointControl", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                # Isaac 6: doc trang thai khop bang node rieng (targetPrim tren PublishJointState da deprecated)
                ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
                # TF: giu targetPrims (deprecated nhung con ho tro). IsaacComputeTransformTree KHONG mo rong
                # cay link khi articulation root nam tren joint prim (/ur5e/root_joint cua asset UR5e)
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
                ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
                ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
                ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
                ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
                ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
                ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
                ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", articulation_path),
                ("ReadJointState.inputs:prim", [usdrt.Sdf.Path(articulation_path)]),
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("SubscribeJointState.inputs:topicName", "joint_command"),
                # ca cay articulation (base_link ... gripper) + TestCube -> /tf, goc la World
                ("PublishTF.inputs:targetPrims", tf_targets),
                ("PublishTF.inputs:topicName", "tf"),
            ],
        },
    )


def _set_camera_rate(stage, camera_path):
    """Gioi han toc do render/publish bang omni:sensor:tickRate (cach cua Isaac 6, nhu RtxCamera)."""
    cam = stage.GetPrimAtPath(camera_path)
    cam.ApplyAPI("OmniSensorAPI")
    rate = _set(cam, "omni:sensor:tickRate", Sdf.ValueTypeNames.Float, CAM_TICK_HZ)
    return [(math.isclose(rate, CAM_TICK_HZ), f"camera {camera_path.rsplit('/', 2)[-2]} tickRate = {rate} Hz")]


def _add_demo_camera(stage):
    """Camera nhin toan canh ban + robot (quay video demo, nhat la khi chay headless)."""
    cam = UsdGeom.Camera.Define(stage, DEMO_CAM_PATH)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*DEMO_CAM_EYE), Gf.Vec3d(*DEMO_CAM_TARGET), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())  # look-at: world->camera, can nghich dao
    h_ap = 20.955
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * DEMO_CAM_H / DEMO_CAM_W)
    cam.CreateFocalLengthAttr(16.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    return str(cam.GetPath())


def _build_camera_graph(camera_path, graph="/ROS2/Camera", width=CAM_W, height=CAM_H, helpers=None):
    keys = og.Controller.Keys
    helpers = helpers or [
        ("RgbPublish", "isaacsim.ros2.bridge.ROS2CameraHelper", "camera/color/image_raw", "rgb"),
        ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper", "camera/depth/image_rect_raw", "depth"),
        ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper", "camera/color/camera_info", None),
    ]
    create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ] + [(n, t) for n, t, _, _ in helpers]
    connect = [
        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
        ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
    ]
    values = [
        ("RenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
        ("RenderProduct.inputs:width", width),
        ("RenderProduct.inputs:height", height),
    ]
    for name, _, topic, kind in helpers:
        connect += [
            ("RenderProduct.outputs:execOut", f"{name}.inputs:execIn"),
            ("RenderProduct.outputs:renderProductPath", f"{name}.inputs:renderProductPath"),
            ("Context.outputs:context", f"{name}.inputs:context"),
        ]
        values += [
            (f"{name}.inputs:topicName", topic),
            (f"{name}.inputs:frameId", "demo_camera" if graph != "/ROS2/Camera" else "camera_color_optical_frame"),
            (f"{name}.inputs:resetSimulationTimeOnStop", True),
        ]
        if kind:
            values.append((f"{name}.inputs:type", kind))
    og.Controller.edit(
        {"graph_path": graph, "evaluator_name": "execution"},
        {keys.CREATE_NODES: create, keys.CONNECT: connect, keys.SET_VALUES: values},
    )


async def build():
    await _frames(5)
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    ctx = omni.usd.get_context()
    await ctx.new_stage_async()
    await _frames(2)
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if not SCENE_USD.exists():
        log(f"KHONG tim thay scene {SCENE_USD}")
        return
    repo = SCENE_USD.parent
    full_robot = (repo / "ur_1" / "payloads" / "base.usda").exists() and (
        repo / "d435_import" / "d435_1" / "completed435ros.usd"
    ).exists()
    # Khi thieu ur_1/, USD se bao "Could not open asset" cho reference/payload cua /ur (bi muted trong Isaac).
    # Day la canh bao dung (file that su thieu), het khi repo co du ur_1/ va d435_import/.
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD.as_posix())
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))
    await _frames(2)
    hung = _read_hung_layer()
    wrist3 = None

    if full_robot:
        log("Dung robot goc cua scene (UR5e + Robotiq 3F + D435)")
        robot_root = "/ur"
        cam = _find(
            stage.GetPrimAtPath(robot_root),
            lambda p: p.IsA(UsdGeom.Camera) and "camera_color_optical_frame" in str(p.GetPath()),
        )
        camera_path = str(cam.GetPath()) if cam else None
    else:
        log("Thieu ur_1/ hoac d435_import/ -> thay bang UR5e chuan cua Isaac Sim (tam thoi)")
        stage.GetPrimAtPath("/ur").SetActive(False)
        robot_root = "/ur5e"
        robot = stage.DefinePrim(robot_root, "Xform")
        robot.GetReferences().AddReference(_assets_root() + UR5E_ASSET)
        robot.GetVariantSet("Physics").SetVariantSelection("PhysX")
        robot.GetVariantSet("Gripper").SetVariantSelection("None")
        xf = UsdGeom.Xformable(robot)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*hung["pos"]))
        await _frames(10)
        _apply_drives(stage, robot_root, hung["drives"])
        wrist3 = _find(robot, lambda p: p.GetName() == "wrist_3_link" and p.IsA(UsdGeom.Xformable))
        camera_path = _add_temp_d435(stage, str(wrist3.GetPath())) if wrist3 else None

    checks = []
    gripper_info = None
    if not full_robot and GRIPPER == "2f85" and wrist3 is not None:
        gripper_info, gchecks = _setup_2f85(stage, robot_root, str(wrist3.GetPath()))
        checks += gchecks
        await _frames(5)
        log(f"Gan Robotiq 2F-85 ({rq.PHYSICS_VARIANT}) tai {gripper_info['edit'] if gripper_info else '-'}")
    elif full_robot:
        log("Robot goc da co gripper 3F -> khong gan 2F-85")

    art = _find(stage.GetPrimAtPath(robot_root), lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI))
    articulation_path = str(art.GetPath()) if art else robot_root
    log(f"robot={robot_root} articulation_root={articulation_path} camera={camera_path}")
    # Asset UR5e co san robot_gripper_joint (body1 rong, variant Gripper=None chi dat jointEnabled=False)
    # nhung PhysX van parse va canh bao "disjointed body transforms" -> tat han prim
    ugj = _find(stage.GetPrimAtPath(robot_root), lambda p: p.GetName() == "robot_gripper_joint")
    if ugj is not None and not full_robot:
        ugj_path = ugj.GetPath()
        ugj.SetActive(False)
        checks.append((not stage.GetPrimAtPath(ugj_path).IsActive(), f"{ugj_path.name} (khop gripper cu cua asset UR5e) da tat"))

    if gripper_info:
        checks += _tune_physics(stage, articulation_path)
    _ensure_environment(stage)
    top, rng, tchecks = _add_table_collider(stage)
    checks += tchecks
    extra_tf = []
    robot_pos = hung["pos"] if not full_robot else DEFAULT_ROBOT_POS
    if top is not None and OBJECTS == "cube":
        cube_path, cchecks = _add_test_cube(stage, robot_pos, top, rng)
        checks += cchecks
        extra_tf.append(cube_path)
    elif top is not None:
        obj_paths, ochecks = _add_hope_objects(stage, robot_pos, top, rng)
        checks += ochecks
        extra_tf += obj_paths
    cfg = _write_gripper_config(stage, gripper_info, str(wrist3.GetPath()) if not full_robot and wrist3 else "")
    log(f"Ghi {GRIPPER_CONFIG.name}: gripper={cfg['gripper']} tcp={cfg.get('tcp_offset_m')} commit={cfg.get('asset_commit')}")

    if camera_path and os.environ.get("UR5E_NO_CAMERA", "0") != "1":
        checks += _set_camera_rate(stage, camera_path)
    demo_cam = _add_demo_camera(stage) if DEMO_CAM else None
    if demo_cam:
        checks += _set_camera_rate(stage, demo_cam)

    for ok, msg in checks:
        log(f"CHECK {'PASS' if ok else 'FAIL'} {msg}")
    n_ok = sum(ok for ok, _ in checks)
    log(f"CONFIG {'OK' if n_ok == len(checks) else 'LOI'} {n_ok}/{len(checks)}")

    _build_joint_graph(articulation_path, extra_tf)
    if os.environ.get("UR5E_NO_CAMERA", "0") == "1":
        log("UR5E_NO_CAMERA=1 -> khong tao graph camera (chan doan)")
    elif camera_path:
        _build_camera_graph(camera_path)
    if demo_cam:
        _build_camera_graph(demo_cam, graph="/ROS2/DemoCamera", width=DEMO_CAM_W, height=DEMO_CAM_H,
                            helpers=[("RgbPublish", "isaacsim.ros2.bridge.ROS2CameraHelper", "demo_cam/rgb", "rgb")])
        log(f"Camera toan canh {DEMO_CAM_W}x{DEMO_CAM_H} -> /demo_cam/rgb")
    else:
        log("Khong tim thay camera -> bo qua graph camera")

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[0.9, 1.1, 1.45], target=[-0.25, 0.0, 0.85])
    except Exception as exc:  # noqa: BLE001
        log(f"khong dat duoc goc nhin viewport: {exc}")

    await _frames(5)
    if os.environ.get("UR5E_AUTOPLAY", "1") != "0":
        omni.timeline.get_timeline_interface().play()
        log("Da bam Play. Panel: pixi run panel")
    else:
        log("San sang. Bam Play roi chay: pixi run panel")


asyncio.ensure_future(build())
