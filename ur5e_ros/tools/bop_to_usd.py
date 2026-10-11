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


def read_ply_ascii(path: Path):
    """PLY ASCII cua BOP: vertex x y z nx ny nz texture_u texture_v, face tam giac."""
    with path.open("r", encoding="ascii", errors="replace") as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.startswith("end_header"):
                break
        n_v = n_f = 0
        props, texture = [], None
        for h in header:
            if h.startswith("element vertex"):
                n_v = int(h.split()[-1])
            elif h.startswith("element face"):
                n_f = int(h.split()[-1])
            elif h.startswith("property float") or h.startswith("property double"):
                props.append(h.split()[-1])
            elif h.lower().startswith("comment texturefile"):
                texture = h.split()[-1]
        verts = np.loadtxt(f, dtype=np.float64, max_rows=n_v)
        faces = np.loadtxt(f, dtype=np.int64, max_rows=n_f)
    col = {p: i for i, p in enumerate(props)}
    if not (faces[:, 0] == 3).all():
        raise ValueError(f"{path.name}: chi ho tro mat tam giac")
    out = {"points": verts[:, [col["x"], col["y"], col["z"]]], "faces": faces[:, 1:4], "texture": texture}
    if "nx" in col:
        out["normals"] = verts[:, [col["nx"], col["ny"], col["nz"]]]
    if "texture_u" in col:
        out["uv"] = verts[:, [col["texture_u"], col["texture_v"]]]
    return out


def grasp_height(pts: np.ndarray, bin_m: float = 0.003):
    """Chon do cao kep (tinh tu day vat) theo profile be rong doc truc hep.

    Uu tien cho vat rong nhat (vat thuon nhu coc se khong tuot xuong khi nang), neu bang nhau
    thi gan giua chieu cao (gan trong tam). Gioi han: >= 25 mm tren ban, <= dinh - 12 mm.
    """
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    axis = 0 if (hi - lo)[0] <= (hi - lo)[1] else 1  # truc hep (giong demo)
    h = hi[2] - lo[2]
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


def write_usd(oid: int, name: str, ply: dict, src_dir: Path, out_dir: Path) -> dict:
    asset_dir = out_dir / name
    (asset_dir / "textures").mkdir(parents=True, exist_ok=True)
    usd_path = asset_dir / f"{name}.usda"
    pts = ply["points"] * 0.001  # mm -> m
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    mass = float(np.clip(np.prod(size) * DENSITY, *MASS_RANGE))

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root).SetKind("component")
    rp = root.GetPrim()
    rp.SetCustomDataByKey("bop", {"dataset": "hope", "obj_id": oid, "source": f"obj_{oid:06d}.ply"})

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
    UsdPhysics.MeshCollisionAPI.Apply(mp).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
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
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if ply.get("texture") and "uv" in ply:
        tex_name = Path(ply["texture"]).name
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
    gh, gw, profile, tapered = grasp_height(pts)
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=list(DEFAULT_IDS))
    ap.add_argument("--src", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "usd"))
    ap.add_argument("--fetch", action="store_true", help="tai cac file con thieu tu zip HOPE tren Hugging Face")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        need = [f"obj_{oid:06d}.{ext}" for oid in args.ids for ext in ("ply", "png")] + ["models_info.json"]
        need = [n for n in need if not (src / n).exists()]
        if need:
            from remote_zip import open_remote_zip

            zf, raw = open_remote_zip(HOPE_ZIP_URL)
            src.mkdir(parents=True, exist_ok=True)
            for n in need:
                with zf.open(f"models/{n}") as f:
                    (src / n).write_bytes(f.read())
                print(f"tai {n}")
            print(f"da tai {raw.downloaded / 1e6:.1f} MB / zip {raw.size / 1e6:.1f} MB")
    manifest = []
    for oid in args.ids:
        name = HOPE_NAMES[oid]
        ply = read_ply_ascii(src / f"obj_{oid:06d}.ply")
        info = write_usd(oid, name, ply, src, out)
        manifest.append(info)
        s = [round(v * 1000, 1) for v in info["size_m"]]
        print(f"OK obj_{oid:06d} -> {info['usd']}  {s} mm, {info['mass_kg']} kg, {info['points']} dinh / {info['faces']} mat")
        print(f"   kep o {info['grasp_height_m'] * 1000:.0f} mm tu day, be rong {info['grasp_width_m'] * 1000:.1f} mm"
              f"{' (THON)' if info['tapered'] else ''}; "
              f"profile (cao: rong) {info['width_profile_mm']}")
    (out / "objects.json").write_text(json.dumps({"dataset": "BOP HOPE", "objects": manifest}, indent=2), encoding="utf-8")
    print(f"manifest: {out / 'objects.json'}")


if __name__ == "__main__":
    main()
