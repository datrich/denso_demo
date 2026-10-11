"""2 bang chuyen (asset NVIDIA ConveyorBelt_A06) phia sau robot + thung KLT hung vat o cuoi moi bang.

Bo tri (nhin tu tren, robot o mep -X cua ban, quay mat ve +X):

        thung 1     thung 2
           ^           ^          bang chay huong -X (ra xa ban)
         bang 1      bang 2
           |     R     |          bang 1 o -Y (vat 1, 2), bang 2 o +Y (vat 3, 4, 5)
    +------------------------+
    |          ban           |

Asset goc la bang chuyen cong nghiep (mat bang cao 1.78 m, rong 0.9 m) -> thu nho DEU de mat bang
bang dung mat ban (giu nguyen hinh dang). Mat bang (/Belt) da la rigid body kinematic trong asset;
van toc be mat do node IsaacConveyor (isaacsim.asset.gen.conveyor) dat, hoac PhysxSurfaceVelocityAPI.

Chi dung pxr (khong phu thuoc Isaac) -> test duoc offline bang usd-core.
"""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

CONVEYOR_ASSET = "/Isaac/Props/Conveyors/ConveyorBelt_A06.usd"
BIN_ASSET = "/Isaac/Props/KLT_Bin/small_KLT.usd"
ROOT = "/Conveyors"
# Moi bang: ten, toa do Y tam bang (khung base_link), chi so vat trong hang gap (0 = vat 1)
LINES = (
    ("Line1", -0.33, (0, 1)),
    ("Line2", 0.33, (2, 3, 4)),
)
TABLE_GAP = 0.01        # dau bang cach mep ban
DROP_FROM_START = 0.18  # diem tha cach dau bang (doc theo bang)
BELT_SPEED = 0.12       # m/s
BIN_RIM_BELOW_BELT = 0.04
BIN_GAP = 0.005         # thanh thung gan bang cach cuoi bang


def _bounds(stage: Usd.Stage, prim: Usd.Prim) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _reference(stage: Usd.Stage, path: str, asset: str) -> Usd.Prim:
    """Prim boc (vi tri/xoay/ti le cua minh) + con 'asset' tham chieu file - giu nguyen xform goc cua asset."""
    wrapper = UsdGeom.Xform.Define(stage, path).GetPrim()
    stage.DefinePrim(f"{path}/asset", "Xform").GetReferences().AddReference(asset)
    return wrapper


def _place(prim: Usd.Prim, translate, yaw_deg: float, scale: float = 1.0):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate))
    xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(yaw_deg)
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale, scale, scale))


def build(stage: Usd.Stage, assets_root: str, robot_pos, table_top: float, table_rng: Gf.Range3d,
          floor_z: float = 0.0) -> tuple[dict, list[tuple[bool, str]]]:
    """Dung 2 bang chuyen + 2 thung. Tra ve (config cho demo - toa do khung base_link, danh sach check)."""
    UsdGeom.Scope.Define(stage, ROOT)
    rx, ry, rz = robot_pos
    x_start = table_rng.GetMin()[0] - TABLE_GAP  # dau bang (phia ban), bang chay ve -X
    lines, checks = [], []
    for name, y_base, obj_idx in LINES:
        y = ry + y_base
        conv = _reference(stage, f"{ROOT}/{name}", assets_root + CONVEYOR_ASSET)
        belt = stage.GetPrimAtPath(f"{conv.GetPath()}/asset/Belt")
        if not belt:
            checks.append((False, f"{name}: asset {CONVEYOR_ASSET} khong co /Belt"))
            continue
        whole, b = _bounds(stage, conv), _bounds(stage, belt)  # khung asset (chua bien doi)
        s = (table_top - floor_z) / (b.GetMax()[2] - whole.GetMin()[2])
        # xoay 180 deg: x_world = tx - s * x_local -> dau bang (x_local nho nhat) o x_start
        tx = x_start + s * b.GetMin()[0]
        ty = y + s * (b.GetMin()[1] + b.GetMax()[1]) / 2
        tz = floor_z - s * whole.GetMin()[2]
        _place(conv, (tx, ty, tz), 180.0, s)
        bw = _bounds(stage, belt)
        belt_len = s * (b.GetMax()[0] - b.GetMin()[0])
        x_end = x_start - belt_len
        belt_w = bw.GetMax()[1] - bw.GetMin()[1]

        # mat bang: rigid body kinematic (asset da co) - dam bao lai, ma sat cao de vat khong truot
        UsdPhysics.RigidBodyAPI.Apply(belt).CreateKinematicEnabledAttr(True)
        top_ok = abs(bw.GetMax()[2] - table_top) < 0.002
        checks.append((top_ok and belt.HasAPI(UsdPhysics.CollisionAPI),
                       f"{name}: ConveyorBelt_A06 x{s:.3f}, mat bang z={bw.GetMax()[2]:.4f} (ban {table_top:.4f}), "
                       f"dai {belt_len:.2f} m, rong {belt_w:.2f} m, x [{x_end:.3f}..{x_start:.3f}], y={y:.3f}"))

        # thung KLT: canh dai nam ngang bang, mieng thung thap hon mat bang
        bin_wrap = _reference(stage, f"{ROOT}/{name}_Bin", assets_root + BIN_ASSET)
        bin_prim = stage.GetPrimAtPath(f"{bin_wrap.GetPath()}/asset")  # goc asset = rigid body
        k = _bounds(stage, bin_wrap)
        yaw = 90.0 if (k.GetMax()[0] - k.GetMin()[0]) > (k.GetMax()[1] - k.GetMin()[1]) else 0.0
        sx, sy = (k.GetMax()[1] - k.GetMin()[1], k.GetMax()[0] - k.GetMin()[0]) if yaw else \
                 (k.GetMax()[0] - k.GetMin()[0], k.GetMax()[1] - k.GetMin()[1])
        h = k.GetMax()[2] - k.GetMin()[2]
        rim = table_top - BIN_RIM_BELOW_BELT
        bin_cx = x_end - BIN_GAP - sx / 2
        _place(bin_wrap, (0, 0, 0), yaw)
        c = _bounds(stage, bin_wrap)  # tam hop bao sau khi xoay
        cx0, cy0 = (c.GetMin()[0] + c.GetMax()[0]) / 2, (c.GetMin()[1] + c.GetMax()[1]) / 2
        _place(bin_wrap, (bin_cx - cx0, y - cy0, rim - c.GetMax()[2]), yaw)
        if bin_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(bin_prim).CreateKinematicEnabledAttr(True)  # thung dung yen
        kb = _bounds(stage, bin_wrap)

        # be do thung (khoi xam tu san len day thung)
        stand_h = kb.GetMin()[2] - floor_z
        stand = UsdGeom.Cube.Define(stage, f"{ROOT}/{name}_BinStand")
        stand.CreateSizeAttr(1.0)
        stand.AddTranslateOp().Set(Gf.Vec3d(bin_cx, y, floor_z + stand_h / 2))
        stand.AddScaleOp().Set(Gf.Vec3f(sx * 0.9, sy * 0.9, stand_h))
        stand.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.36, 0.38)])
        UsdPhysics.CollisionAPI.Apply(stand.GetPrim())

        bin_ok = (bin_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                  and UsdPhysics.RigidBodyAPI(bin_prim).GetKinematicEnabledAttr().Get()
                  and abs(kb.GetMax()[2] - rim) < 0.002 and kb.GetMax()[0] < x_end and stand_h > 0.1)
        checks.append((bool(bin_ok), f"{name}_Bin: KLT {sx * 100:.0f}x{sy * 100:.0f}x{h * 100:.1f} cm, mieng z={kb.GetMax()[2]:.3f} "
                                     f"(thap hon bang {(table_top - kb.GetMax()[2]) * 100:.1f} cm), be do {stand_h:.2f} m"))

        drop = (x_start - DROP_FROM_START - rx, y_base)
        reach = (drop[0] ** 2 + drop[1] ** 2) ** 0.5
        checks.append((0.3 < reach < 0.7, f"{name}: diem tha (base) ({drop[0]:.3f}, {drop[1]:.3f}) cach de {reach:.2f} m, vat {[i + 1 for i in obj_idx]}"))
        lines.append({
            "name": name,
            "belt_prim": str(belt.GetPath()),
            "object_index": list(obj_idx),
            "drop_xy": [round(drop[0], 4), round(drop[1], 4)],
            "belt_z": round(table_top - rz, 5),
            "direction_xy": [-1.0, 0.0],
            "speed_mps": BELT_SPEED,
            "belt_x_range": [round(x_end - rx, 4), round(x_start - rx, 4)],
            "belt_width_m": round(belt_w, 4),
            "bin": {
                "min": [round(kb.GetMin()[0] - rx, 4), round(kb.GetMin()[1] - ry, 4), round(kb.GetMin()[2] - rz, 4)],
                "max": [round(kb.GetMax()[0] - rx, 4), round(kb.GetMax()[1] - ry, 4), round(kb.GetMax()[2] - rz, 4)],
            },
        })
    return {"lines": lines}, checks


def apply_surface_velocity(stage: Usd.Stage, belt_path: str, speed: float) -> bool:
    """Du phong khi khong co node IsaacConveyor: van toc be mat co dinh doc +X cua mat bang (local)."""
    belt = stage.GetPrimAtPath(belt_path)
    belt.AddAppliedSchema("PhysxSurfaceVelocityAPI")
    belt.CreateAttribute("physxSurfaceVelocity:surfaceVelocityEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    belt.CreateAttribute("physxSurfaceVelocity:surfaceVelocityLocalSpace", Sdf.ValueTypeNames.Bool).Set(True)
    belt.CreateAttribute("physxSurfaceVelocity:surfaceVelocity", Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(speed, 0, 0))
    return True
