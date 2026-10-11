"""Gan Robotiq 2F-85 (github.com/robotiq/isaacsim_assets) len co tay UR5e - PhysX parallel_grip.

Lam theo grippers/GRIPPER_SIMULATION_GUIDE.md (§2.2-2.4, §3.1) va scene demo da duoc
Robotiq kiem chung grippers/demo/teleop/ur5robot_with_2F-85.usda:
  - prim gripper la con cua wrist_3_link, pose don vi
  - FixedJoint wrist_3_link <-> gripper base_link
  - xoa ArticulationRootAPI cua gripper, tat root_joint (gripper thuoc articulation cua robot)
  - finger_joint: stiffness 50, damping 3, maxForce 10 N.m (guide §3.1)
  - 4 than cua co cau: solver iterations 64/8 (lop tuning cua demo)

Chi dung pxr (khong phu thuoc Isaac) -> test duoc offline bang usd-core.
"""

from __future__ import annotations

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

EDIT_NAME = "Robotiq_2F_85_edit"
PHYSICS_VARIANT = "Physx_parallel_grip"
FINGERTIP_VARIANT = "Standard"
FINGER_JOINT = "finger_joint"
FINGER_LIMIT_DEG = 47.0
FINGER_CLOSED_RAD = 0.8  # ~45.8 deg, gan gioi han 47 deg (guide §5.1: dong ~0.8 rad)
FINGER_DRIVE = {"stiffness": 50.0, "damping": 3.0, "maxForce": 10.0}
LINKAGE_BODIES = ("left_outer_finger", "left_inner_finger", "right_outer_finger", "right_inner_finger")
LINKAGE_ITERATIONS = (64, 8)
# gripper cung co link ten "base_link" -> doi ten frame TF de khong trung base_link cua UR5e
BASE_LINK_TF_NAME = "robotiq_base_link"


def _find(root: Usd.Prim, name: str, predicate=None):
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name and (predicate is None or predicate(prim)):
            return prim
    return None


def _set(prim: Usd.Prim, name: str, type_name, value):
    attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)
    return attr


def _world_anchored(joint: UsdPhysics.Joint) -> bool:
    return not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets()


def mount(stage: Usd.Stage, wrist3_path: str, asset_path: str, mount_joint_path: str) -> dict:
    """Gan gripper; tra ve cac duong dan quan trong."""
    edit_path = f"{wrist3_path}/{EDIT_NAME}"
    edit = stage.DefinePrim(edit_path, "Xform")
    edit.GetPayloads().AddPayload(asset_path)
    spec = stage.GetEditTarget().GetPrimSpecForScenePath(edit.GetPath())
    spec.variantSelections["Physics"] = PHYSICS_VARIANT
    spec.variantSelections["Fingertip"] = FINGERTIP_VARIANT
    xf = UsdGeom.Xformable(edit)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))
    stage.Load(edit.GetPath())

    gripper = stage.GetPrimAtPath(f"{edit_path}/Robotiq_2F_85")
    if not gripper:
        raise RuntimeError(f"Khong nap duoc gripper tu {asset_path}")

    # 1) gripper nhap vao articulation cua robot
    for prim in Usd.PrimRange(gripper):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            prim.RemoveAppliedSchema("PhysxArticulationAPI")
    for prim in Usd.PrimRange(gripper):
        if prim.IsA(UsdPhysics.Joint) and _world_anchored(UsdPhysics.Joint(prim)):
            prim.SetActive(False)  # root_joint: neo gripper vao the gioi

    # 2) fixed joint co tay <-> base_link gripper, giu nguyen pose hien tai
    wrist3 = stage.GetPrimAtPath(wrist3_path)
    base = stage.GetPrimAtPath(f"{gripper.GetPath()}/base_link")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rel = cache.GetLocalToWorldTransform(base) * cache.GetLocalToWorldTransform(wrist3).GetInverse()
    rel.Orthonormalize()
    joint = UsdPhysics.FixedJoint.Define(stage, mount_joint_path)
    joint.CreateBody0Rel().SetTargets([wrist3.GetPath()])
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # 3) finger_joint: tuning theo guide §3.1, mo luc khoi dong
    finger = _find(gripper, FINGER_JOINT, lambda p: p.IsA(UsdPhysics.RevoluteJoint))
    drive = UsdPhysics.DriveAPI.Apply(finger, "angular")
    drive.CreateStiffnessAttr().Set(FINGER_DRIVE["stiffness"])
    drive.CreateDampingAttr().Set(FINGER_DRIVE["damping"])
    drive.CreateMaxForceAttr().Set(FINGER_DRIVE["maxForce"])
    drive.CreateTargetPositionAttr().Set(0.0)

    # 4) solver iterations cho 4 than cua co cau ngon
    for name in LINKAGE_BODIES:
        body = _find(gripper, name)
        if body:
            body.AddAppliedSchema("PhysxRigidBodyAPI")
            _set(body, "physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int, LINKAGE_ITERATIONS[0])
            _set(body, "physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, LINKAGE_ITERATIONS[1])

    # 5) ten frame TF rieng cho base_link cua gripper
    _set(base, "isaac:nameOverride", Sdf.ValueTypeNames.String, BASE_LINK_TF_NAME)

    return {
        "edit": edit_path,
        "gripper": str(gripper.GetPath()),
        "base_link": str(base.GetPath()),
        "finger_joint": str(finger.GetPath()),
        "mount_joint": mount_joint_path,
    }


def grip_frame_in(stage: Usd.Stage, frame_path: str, gripper_path: str):
    """Vi tri + huong grip_frame (TCP giua 2 dau ngon) trong khung frame_path."""
    grip = _find(stage.GetPrimAtPath(gripper_path), "grip_frame")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rel = cache.GetLocalToWorldTransform(grip) * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(frame_path)).GetInverse()
    rel.Orthonormalize()
    return rel.ExtractTranslation(), rel.ExtractRotationQuat()


def validate(stage: Usd.Stage, robot_root: str, info: dict) -> list[tuple[bool, str]]:
    """Kiem tra cau hinh sau khi gan; moi muc la (ok, mo ta)."""
    out = []
    robot = stage.GetPrimAtPath(robot_root)
    gripper = stage.GetPrimAtPath(info["gripper"])
    edit = stage.GetPrimAtPath(info["edit"])

    sel = edit.GetVariantSet("Physics").GetVariantSelection()
    out.append((sel == PHYSICS_VARIANT, f"variant Physics = {sel}"))
    tip = edit.GetVariantSet("Fingertip").GetVariantSelection()
    out.append((tip == FINGERTIP_VARIANT, f"variant Fingertip = {tip}"))

    roots = [str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    out.append((len(roots) == 1 and not roots[0].startswith(info["gripper"]), f"articulation root duy nhat: {roots}"))

    anchored = [
        str(p.GetPath())
        for p in Usd.PrimRange(gripper)
        if p.IsActive() and p.IsA(UsdPhysics.Joint) and _world_anchored(UsdPhysics.Joint(p))
    ]
    out.append((not anchored, f"khong con joint neo gripper vao the gioi: {anchored or 'OK'}"))

    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(info["mount_joint"]))
    b0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
    b1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
    bodies_ok = all(stage.GetPrimAtPath(p).HasAPI(UsdPhysics.RigidBodyAPI) for p in b0 + b1)
    out.append((len(b0) == 1 and len(b1) == 1 and bodies_ok, f"mount joint {b0} -> {b1} (ca 2 la rigid body: {bodies_ok})"))

    finger = stage.GetPrimAtPath(info["finger_joint"])
    drive = UsdPhysics.DriveAPI(finger, "angular")
    vals = (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get())
    want = (FINGER_DRIVE["stiffness"], FINGER_DRIVE["damping"], FINGER_DRIVE["maxForce"])
    out.append((all(math.isclose(a, b) for a, b in zip(vals, want)), f"finger_joint drive (K, C, maxF) = {vals}"))
    lim = UsdPhysics.RevoluteJoint(finger).GetUpperLimitAttr().Get()
    out.append((lim is not None and math.degrees(FINGER_CLOSED_RAD) <= lim, f"finger_joint upperLimit = {lim} deg >= lenh dong {math.degrees(FINGER_CLOSED_RAD):.1f}"))

    mimic = []
    for p in Usd.PrimRange(gripper):
        if not p.IsActive():
            continue
        for r in p.GetRelationships():
            if r.GetName().startswith("physxMimicJoint:") and r.GetName().endswith(":referenceJoint"):
                mimic.append((p.GetName(), [str(t) for t in r.GetTargets()]))
    bad = [m for m in mimic if m[1] != [info["finger_joint"]]]
    out.append((len(mimic) >= 4 and not bad, f"{len(mimic)} mimic joint tro toi finger_joint" + (f", SAI: {bad}" if bad else "")))

    iters = []
    for name in LINKAGE_BODIES:
        b = _find(gripper, name)
        iters.append(b.GetAttribute("physxRigidBody:solverPositionIterationCount").Get() if b else None)
    out.append((all(v == LINKAGE_ITERATIONS[0] for v in iters), f"solver iterations co cau ngon = {iters}"))

    pos, _ = grip_frame_in(stage, str(joint.GetBody0Rel().GetTargets()[0]), info["gripper"])
    out.append((abs(pos[0]) < 1e-3 and abs(pos[1]) < 1e-3 and 0.10 < pos[2] < 0.20, f"grip_frame (TCP) trong wrist_3_link = {Gf.Vec3d(pos)}"))
    return out
