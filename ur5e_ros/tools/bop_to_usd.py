"""Chuyen model BOP (PLY ASCII, mm, texture PNG) -> USD san sang mo phong cho Isaac Sim (PhysX).

    pixi run fetch-hope                       # tai 5 vat HOPE (~20 MB, chi phan can thiet) + chuyen USD
    pixi run bop-to-usd                       # chi chuyen (model da co trong datasets/bop_hope)
    pixi run bop-to-usd -- --ids 3 5 10 23 28 --fetch

Moi vat -> datasets/bop_hope/usd/<Ten>/<Ten>.usda (+ textures/), gom:
  - Mesh (m), normals, UV, vat lieu UsdPreviewSurface + texture
  - RigidBody + Mass (khoi luong uoc tu hop bao * mat do do choi), collision convexHull
  - PhysX: contactOffset 2 mm, restOffset 0.5 mm, damping, solver iters (guide Robotiq §3.1),
    ma sat xoan (torsionalPatchRadius 10 mm) de vat thon khong xoay quanh 2 diem kep
  - ma sat 1.0 / 0.9
Va manifest datasets/bop_hope/usd/objects.json cho scene + demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[2] / "datasets" / "bop_hope"
HOPE_ZIP_URL = "https://huggingface.co/datasets/bop-benchmark/hope/resolve/main/hope_models.zip"
# Thu tu ten trong HOPE (id theo bang chu cai, da doi chieu bang texture)
HOPE_NAMES = {
    1: "AlphabetSoup", 2: "BBQSauce", 3: "Butter", 4: "Cherries", 5: "ChocolatePudding", 6: "Cookies",
    7: "Corn", 8: "CreamCheese", 9: "GranolaBars", 10: "GreenBeans", 11: "Ketchup", 12: "MacaroniAndCheese",
    13: "Mayo", 14: "Milk", 15: "Mushrooms", 16: "Mustard", 17: "OrangeJuice", 18: "Parmesan", 19: "Peaches",
    20: "PeasAndCarrots", 21: "Pineapple", 22: "Popcorn", 23: "Raisins", 24: "SaladDressing", 25: "Spaghetti",
    26: "TomatoSauce", 27: "Tuna", 28: "Yogurt",
}
DEFAULT_IDS = (3, 5, 10, 23, 28)
DENSITY = 300.0  # kg/m3 tren hop bao - do choi rong ruot
MASS_RANGE = (0.05, 0.15)

# Bo du lieu ho tro. "rest": xoay vat ve tu the dung vung (model BOP cong nghiep khong dat san thang dung)
DATASETS = {
    "hope": {
        "title": "BOP HOPE", "url": HOPE_ZIP_URL, "names": HOPE_NAMES, "ids": DEFAULT_IDS,
        "mass": "bbox", "density": DENSITY, "mass_range": MASS_RANGE, "rest": False,
    },
    "itodd": {
        "title": "BOP ITODD (MVTec, chi tiet co khi)",
        "url": "https://huggingface.co/datasets/bop-benchmark/itodd/resolve/main/itodd_models.zip",
        # ten mo ta (ITODD khong dat ten), chon tu khao sat: dung vung, kep duoc bang 2F-85
        "names": {23: "HexBolt", 24: "SteppedPin", 26: "PipeTee", 9: "ClampBlock", 10: "SlottedClamp"},
        "ids": (23, 24, 26, 9, 10),
        "mass": "volume", "density": 2700.0, "mass_range": (0.02, 0.6), "rest": True,  # nhom
        "collision": "convexDecomposition",
        "colors": {"HexBolt": (0.55, 0.56, 0.58), "SteppedPin": (0.72, 0.6, 0.32), "PipeTee": (0.7, 0.45, 0.28),
                   "ClampBlock": (0.62, 0.66, 0.7), "SlottedClamp": (0.25, 0.28, 0.32)},
    },
}

# ---- tu the dung vung: thu 6 huong up xuong ban, chon huong tam khoi nam sau trong vung tua ----
REST_ROTS = {  # truc HUONG XUONG -> ma tran xoay dua truc do ve -Z
    "-z": np.eye(3), "+z": np.diag([1.0, -1.0, -1.0]),
    "-x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float), "+x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
    "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float), "+y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
}


def volume_centroid(p, f):
    a, b_, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    v = np.einsum("ij,ij->i", a, np.cross(b_, c)) / 6.0
    vol = v.sum()
    if abs(vol) < 1e-12:
        return 0.0, p.mean(0)
    return abs(vol), (v[:, None] * (a + b_ + c) / 4.0).sum(0) / vol


def _hull2d(pts):
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b2):
        return (a[0] - o[0]) * (b2[1] - o[1]) - (a[1] - o[1]) * (b2[0] - o[0])

    lo, up = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def _margin(poly, q):
    if len(poly) < 3:
        return -1.0
    m = np.inf
    for i in range(len(poly)):
        a, c = poly[i], poly[(i + 1) % len(poly)]
        e = c - a
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
        m = min(m, -np.dot(q - a, n))
    return m


def rest_pose(p, f, max_width=0.070, min_height=0.030):
    """Chon huong dat (mm hoac m deu duoc vi chi so sanh tuong doi) -> (ten, R, le_mep)."""
    _, com = volume_centroid(p, f)
    scale = 1.0 if np.ptp(p) < 5 else 0.001  # p tinh bang mm -> doi m de so voi nguong
    cands = []
    for key, R in REST_ROTS.items():
        q, c = p @ R.T * scale, com @ R.T * scale
        zmin = q[:, 2].min()
        margin = _margin(_hull2d(q[q[:, 2] < zmin + 0.0006][:, :2]), c[:2])
        size = q.max(0) - q.min(0)
        w = min(size[:2])
        ok = margin > 0.003 and size[2] >= min_height and 0.010 <= w <= max_width
        cands.append((ok, margin / max(c[2] - zmin, 1e-6), key, R, margin))
    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    ok, _, key, R, margin = cands[0]
    return key, R, margin, ok


_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
              "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
              "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply(path: Path):
    """PLY cua BOP (ASCII hoac binary_little_endian): vertex x y z [nx ny nz] [texture_u texture_v]
    [red green blue], face tam giac. Tra ve dict points/faces (+ normals/uv/colors/texture)."""
    data = path.read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end = data.index(b"\n", end) + 1
    header = data[:end].decode("ascii", "replace").splitlines()
    fmt, texture, elems, cur = "ascii", None, [], None
    for h in header:
        p = h.strip().split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "comment" and len(p) > 2 and p[1].lower() == "texturefile":
            texture = p[2]
        elif p[0] == "element":
            cur = {"name": p[1], "count": int(p[2]), "props": []}
            elems.append(cur)
        elif p[0] == "property":
            cur["props"].append(("list", p[2], p[3], p[4]) if p[1] == "list" else (p[1], p[2]))
    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"{path.name}: dinh dang PLY {fmt} chua ho tro")
    body = data[end:]
    out = {"texture": texture}
    if fmt == "ascii":
        tokens = body.split()
        pos = 0
        for el in elems:
            if el["name"] == "vertex":
                n = len(el["props"])
                arr = np.array(tokens[pos: pos + n * el["count"]], dtype=np.float64).reshape(el["count"], n)
                pos += n * el["count"]
                cols = {pr[1]: arr[:, i] for i, pr in enumerate(el["props"])}
            elif el["name"] == "face":
                k = int(tokens[pos])
                arr = np.array(tokens[pos: pos + (k + 1) * el["count"]], dtype=np.int64).reshape(el["count"], k + 1)
                pos += (k + 1) * el["count"]
                faces = arr[:, 1:]
            else:
                pos += len(el["props"]) * el["count"]
    else:
        pos = 0
        for el in elems:
            if el["name"] == "vertex":
                dt = np.dtype([(pr[1], "<" + _PLY_TYPES[pr[0]]) for pr in el["props"]])
                arr = np.frombuffer(body, dtype=dt, count=el["count"], offset=pos)
                pos += dt.itemsize * el["count"]
                cols = {name: arr[name].astype(np.float64) for name in dt.names}
            elif el["name"] == "face":
                _, ctype, itype, _ = el["props"][0]
                cdt, idt = np.dtype("<" + _PLY_TYPES[ctype]), np.dtype("<" + _PLY_TYPES[itype])
                k = int(np.frombuffer(body, dtype=cdt, count=1, offset=pos)[0])  # BOP: toan tam giac
                dt = np.dtype([("n", cdt), ("v", idt, (k,))])
                arr = np.frombuffer(body, dtype=dt, count=el["count"], offset=pos)
                pos += dt.itemsize * el["count"]
                if not (arr["n"] == k).all():
                    raise ValueError(f"{path.name}: mat khong dong nhat so dinh")
                faces = arr["v"].astype(np.int64)
    if faces.shape[1] != 3:
        raise ValueError(f"{path.name}: chi ho tro mat tam giac")
    out["points"] = np.column_stack([cols["x"], cols["y"], cols["z"]])
    out["faces"] = faces
    if "nx" in cols:
        out["normals"] = np.column_stack([cols["nx"], cols["ny"], cols["nz"]])
    if "texture_u" in cols:
        out["uv"] = np.column_stack([cols["texture_u"], cols["texture_v"]])
    if "red" in cols:
        out["colors"] = np.column_stack([cols["red"], cols["green"], cols["blue"]]) / 255.0
    return out


read_ply_ascii = read_ply  # ten cu


def _section_widths(pts, faces, axis, heights):
    """Be rong mat cat ngang (theo `axis`) tai tung do cao z - cat mat phang qua tam giac,
    dung ca voi mesh CAD it dinh (khong phu thuoc mat do dinh)."""
    tri = pts[faces]                      # (F, 3, 3)
    z = tri[:, :, 2]
    out = []
    for h in heights:
        span = (z.min(1) <= h) & (z.max(1) >= h)
        if not span.any():
            out.append(np.nan)
            continue
        t = tri[span]
        vals = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b_ = t[:, i], t[:, j]
            dz = b_[:, 2] - a[:, 2]
            ok = ((a[:, 2] - h) * (b_[:, 2] - h) <= 0) & (np.abs(dz) > 1e-12)
            s = (h - a[ok, 2]) / dz[ok]
            vals.append(a[ok, axis] + s * (b_[ok, axis] - a[ok, axis]))
        v = np.concatenate(vals)
        out.append(v.max() - v.min() if len(v) else np.nan)
    return np.array(out)


def grasp_height(pts: np.ndarray, faces: np.ndarray | None = None, bin_m: float = 0.003):
    """Chon do cao kep (tinh tu day vat) theo profile be rong doc truc hep.

    Uu tien cho vat rong nhat (vat thuon nhu coc se khong tuot xuong khi nang), neu bang nhau
    thi gan giua chieu cao (gan trong tam). Gioi han: >= 25 mm tren ban, <= dinh - 12 mm.
    """
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    axis = 0 if (hi - lo)[0] <= (hi - lo)[1] else 1  # truc hep (giong demo)
    h = hi[2] - lo[2]
    if faces is not None:  # mat cat chinh xac (mesh CAD it dinh van dung)
        centers = np.arange(bin_m / 2, h, bin_m)
        widths = _section_widths(pts, faces, axis, lo[2] + centers)
        keep = np.isfinite(widths)
        centers, widths = centers[keep], widths[keep]
    else:  # du phong: chia lop theo dinh (mesh quet day dinh)
        z = pts[:, 2] - lo[2]
        edges = np.arange(0.0, h + bin_m, bin_m)
        centers, widths = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = pts[(z >= a) & (z < b), axis]
            if len(sel) >= 20:
                centers.append((a + b) / 2)
                widths.append(sel.max() - sel.min())
        centers, widths = np.array(centers), np.array(widths)
    zmin = max(0.025, 0.3 * h)
    profile = {f"{int(round(zc * 1000))}": round(float(wc) * 1000, 1)
               for zc, wc in zip(centers[:: max(1, len(centers) // 6)], widths[:: max(1, len(centers) // 6)])}
    # Vat thon (coc): pad phang kep vao thanh nghieng se day vat truot len roi lat
    # -> kep o dai rong nhat (thuong la vanh mieng, thanh dung), cho phep sat dinh hon (8 mm)
    band = (centers >= zmin) & (centers <= max(0.025, h - 0.008))
    tapered = bool(band.any() and np.ptp(widths[band]) > 0.004)
    zmax = max(0.025, h - (0.008 if tapered else 0.012))
    ok = (centers >= zmin) & (centers <= zmax)
    if not ok.any():
        return float(min(max(h / 2, 0.025), zmax)), float(hi[axis] - lo[axis]), profile, tapered
    c, w = centers[ok], widths[ok]
    if tapered:
        i = int(np.argmax(w))
        return float(c[i]), float(w[i]), profile, tapered
    near_max = w >= w.max() - 0.0015
    i = np.argmin(np.abs(c[near_max] - h / 2))
    return float(c[near_max][i]), float(w[near_max][i]), profile, tapered


def write_usd(oid: int, name: str, ply: dict, src_dir: Path, out_dir: Path, cfg: dict | None = None) -> dict:
    cfg = cfg or DATASETS["hope"]
    asset_dir = out_dir / name
    asset_dir.mkdir(parents=True, exist_ok=True)
    usd_path = asset_dir / f"{name}.usda"
    pts = ply["points"] * 0.001  # mm -> m
    rest = {}
    if cfg.get("rest"):
        key, R, margin, ok = rest_pose(pts, ply["faces"])
        pts = pts @ R.T
        if "normals" in ply:
            ply = dict(ply, normals=ply["normals"] @ R.T)
        pts = pts - (pts.min(axis=0) + pts.max(axis=0)) / 2  # goc o tam hop bao (nhu BOP)
        rest = {"rest_down_axis": key, "rest_margin_mm": round(float(margin) * 1000, 1), "rest_ok": bool(ok)}
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    if cfg.get("mass") == "volume":
        vol, _ = volume_centroid(pts, ply["faces"])
        mass = float(np.clip(vol * cfg["density"], *cfg["mass_range"]))
    else:
        mass = float(np.clip(np.prod(size) * cfg.get("density", DENSITY), *cfg.get("mass_range", MASS_RANGE)))

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind("component")
    rp = root.GetPrim()
    rp.SetCustomDataByKey("bop", {"dataset": cfg["title"], "obj_id": oid, "source": f"obj_{oid:06d}.ply",
                                  **{k: v for k, v in rest.items() if k != "rest_ok"}})

    # vat ly than cung
    UsdPhysics.RigidBodyAPI.Apply(rp)
    UsdPhysics.MassAPI.Apply(rp).CreateMassAttr(mass)
    rp.AddAppliedSchema("PhysxRigidBodyAPI")
    rp.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(0.1)
    rp.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(0.1)
    rp.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    rp.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)

    # luoi
    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(ply["faces"]), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(ply["faces"].reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])
    if "normals" in ply:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(ply["normals"].astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if "uv" in ply:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(ply["uv"].astype(np.float32)))
    mp = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mp)
    # vat lom (chi tiet co khi) -> convexDecomposition de diem cham khop hinh that; vat do dung -> convexHull
    approx = cfg.get("collision", UsdPhysics.Tokens.convexHull)
    UsdPhysics.MeshCollisionAPI.Apply(mp).CreateApproximationAttr(approx)
    mp.AddAppliedSchema("PhysxCollisionAPI")
    mp.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
    mp.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0005)
    # Ma sat xoan (mac dinh PhysX = 0): pad phang kep vat thon (coc) chi cham 1 diem moi ben ->
    # khong co ma sat xoan thi vat quay tu do quanh truc noi 2 diem cham va bi lat khi nang / dat
    mp.CreateAttribute("physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.01)
    mp.CreateAttribute("physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float).Set(0.005)

    # vat lieu hien thi
    looks = UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    mat = UsdShade.Material.Define(stage, f"{looks.GetPath()}/Material")
    shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    textured = bool(ply.get("texture") and "uv" in ply)
    color = cfg.get("colors", {}).get(name)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6 if textured else 0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0 if textured else 0.75)  # chi tiet kim loai
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if not textured:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(color or (0.6, 0.62, 0.65))))
    if textured:
        tex_name = Path(ply["texture"]).name
        (asset_dir / "textures").mkdir(exist_ok=True)
        shutil.copy2(src_dir / tex_name, asset_dir / "textures" / tex_name)
        reader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex_name}")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    UsdShade.MaterialBindingAPI.Apply(mp).Bind(mat)

    # vat lieu vat ly (ma sat)
    pmat = UsdShade.Material.Define(stage, f"{looks.GetPath()}/PhysicsMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    pm.CreateStaticFrictionAttr(1.0)
    pm.CreateDynamicFrictionAttr(0.9)
    pm.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI(mp).Bind(pmat, UsdShade.Tokens.weakerThanDescendants, "physics")

    stage.GetRootLayer().Save()
    gh, gw, profile, tapered = grasp_height(pts, ply["faces"])
    return {
        "grasp_height_m": round(gh, 4),  # tinh tu day vat
        "grasp_width_m": round(gw, 4),
        "tapered": tapered,              # be rong thay doi > 4 mm theo chieu cao (coc, chai thon)
        "width_profile_mm": profile,     # {do cao mm: be rong mm} theo truc hep
        "name": name,
        "obj_id": oid,
        "usd": str(usd_path.relative_to(out_dir.parent)).replace("\\", "/"),
        "size_m": [round(float(v), 5) for v in size],
        "min_m": [round(float(v), 5) for v in lo],
        "max_m": [round(float(v), 5) for v in hi],
        "mass_kg": round(mass, 4),
        "points": int(len(pts)),
        "faces": int(len(ply["faces"])),
        **rest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hope", choices=sorted(DATASETS))
    ap.add_argument("--ids", type=int, nargs="+", help="mac dinh: danh sach chon san cua bo du lieu")
    ap.add_argument("--src", help="mac dinh datasets/bop_<dataset>/models")
    ap.add_argument("--out", help="mac dinh datasets/bop_<dataset>/usd")
    ap.add_argument("--fetch", action="store_true", help="tai cac file con thieu tu zip BOP tren Hugging Face")
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    root = ROOT.parent / f"bop_{args.dataset}"
    ids = args.ids or list(cfg["ids"])
    src, out = Path(args.src or root / "models"), Path(args.out or root / "usd")
    out.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        from remote_zip import open_remote_zip

        zf = raw = None
        src.mkdir(parents=True, exist_ok=True)
        for n in [f"obj_{oid:06d}.ply" for oid in ids] + ["models_info.json"]:
            if (src / n).exists():
                continue
            if zf is None:
                zf, raw = open_remote_zip(cfg["url"])
            names = set(zf.namelist())
            todo = [n]
            ply_hdr = None
            with zf.open(f"models/{n}") as f:
                (src / n).write_bytes(f.read())
            if n.endswith(".ply"):  # texture di kem (neu co)
                ply_hdr = (src / n).read_bytes()[:400].decode("ascii", "replace")
                for line in ply_hdr.splitlines():
                    if line.lower().startswith("comment texturefile") and f"models/{line.split()[-1]}" in names:
                        tex = line.split()[-1]
                        with zf.open(f"models/{tex}") as f:
                            (src / tex).write_bytes(f.read())
                        todo.append(tex)
            print("tai " + ", ".join(todo))
        if raw is not None:
            print(f"da tai {raw.downloaded / 1e6:.1f} MB / zip {raw.size / 1e6:.1f} MB")
    manifest = []
    for oid in ids:
        name = cfg["names"].get(oid, f"{args.dataset}_obj_{oid:06d}")
        ply = read_ply(src / f"obj_{oid:06d}.ply")
        info = write_usd(oid, name, ply, src, out, cfg)
        if "rest_down_axis" in info:
            print(f"   tu the dung: truc {info['rest_down_axis']} huong xuong, tam khoi cach mep vung tua "
                  f"{info['rest_margin_mm']} mm{'' if info['rest_ok'] else ' (CANH BAO: khong dat tieu chi kep)'}")
        manifest.append(info)
        s = [round(v * 1000, 1) for v in info["size_m"]]
        print(f"OK obj_{oid:06d} -> {info['usd']}  {s} mm, {info['mass_kg']} kg, {info['points']} dinh / {info['faces']} mat")
        print(f"   kep o {info['grasp_height_m'] * 1000:.0f} mm tu day, be rong {info['grasp_width_m'] * 1000:.1f} mm"
              f"{' (THON)' if info['tapered'] else ''}; "
              f"profile (cao: rong) {info['width_profile_mm']}")
    (out / "objects.json").write_text(json.dumps({"dataset": cfg["title"], "objects": manifest}, indent=2), encoding="utf-8")
    print(f"manifest: {out / 'objects.json'}")


if __name__ == "__main__":
    main()
