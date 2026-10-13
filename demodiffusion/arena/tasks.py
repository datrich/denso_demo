"""Task Arena tuong tu DemoDiffusion + expert co thong tin dac quyen (de tong hop demo "nguoi").

Expert gap-dat: gap tu tren xuong (huong tiep can = -z the gioi), phuong kep vuong goc truc dai cua vat (PCA
diem mesh tren mat phang ngang), diem kep = tam lat cat giua vat (chuoi cong: khong lay tam hop bao); vat rong hon
do mo gripper (bat) thi kep o vanh. Quy dao Cartesian min-jerk qua cac diem: tren vat -> kep -> nang -> tren dich
-> ha -> nha -> rut len, IK Franka moi buoc (trong run_demodiffusion.py).
"""

from __future__ import annotations

import numpy as np

from demodiffusion.franka_kinematics import pose, rot_err

TCP_OFFSET = 0.131      # Robotiq base_link -> diem kep (+x), do trong USD DROID (Arena droid.py)
MAX_OPEN = 0.085
ARENA_ENVS = "isaaclab_arena_environments/robolab/tasks"

TASKS = {
    # DemoDiffusion "pick and place the banana on the plate"
    "banana_on_plate": dict(env_spec=f"{ARENA_ENVS}/banana_on_plate.yaml", pick="banana", place="plate",
                            place_lift=0.04),
    # DemoDiffusion "pick and place the bowl" (bep -> thung xam trong Arena)
    "bowl_in_bin": dict(env_spec=f"{ARENA_ENVS}/bowl_in_bin.yaml", pick="bowl", place="grey_bin", rim_grasp=True,
                        place_lift=0.06),
    # DemoDiffusion "pick up the bear doll" (do choi mem -> tuong do choi trong Arena, dat vao thung)
    "toy_in_bin": dict(env_spec=f"{ARENA_ENVS}/toy_in_bin.yaml", pick="lizard_figurine", place="grey_bin",
                       place_lift=0.06),
}


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def _slerp_R(R0, R1, s):
    from demodiffusion.hand_demo import _rotvec

    return _rotvec(s * rot_err(R1, R0)) @ R0


def object_points_local(sim, name):
    """Diem mesh cua vat trong khung goc cua vat (doc tu USD; tu the tuong doi khong doi)."""
    import omni.usd
    from pxr import Usd, UsdGeom

    import re

    stage = omni.usd.get_context().get_stage()
    root_path = sim.env.scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    root_path = re.sub(r"env_(\[\^/\]\+|\.\*)", "env_0", root_path)  # regex prim cua Isaac Lab -> env 0
    root = stage.GetPrimAtPath(root_path)
    assert root, f"khong tim thay prim {root_path}"
    from demodiffusion.franka_kinematics import orthonormalize

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    # Chi khu phan quay + tinh tien cua prim goc (tu the cua vat), GIU ti le (asset theo cm co scale 0.01 o goc)
    M_root = np.array(cache.GetLocalToWorldTransform(root)).T
    R_root, t_root = orthonormalize(M_root[:3, :3]), M_root[:3, 3]
    pts = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            P = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], float)
            if len(P):
                M = np.array(cache.GetLocalToWorldTransform(p)).T
                Pw = P @ M[:3, :3].T + M[:3, 3]
                pts.append((Pw - t_root) @ R_root)
    assert pts, f"vat {name} khong co mesh"
    return np.vstack(pts)


class PickPlaceExpert:
    def __init__(self, cfg, sim, kin, T_bw):
        self.cfg, self.sim, self.kin, self.T_bw = cfg, sim, kin, T_bw
        self.object_names = [cfg["pick"], cfg["place"]]
        self.q_rest = sim.q()
        self.T0 = T_bw @ sim.T_world_ee()

    def _world_pts(self, name):
        P = object_points_local(self.sim, name)
        T = self.T_bw @ self.sim.obj_T(name)  # tu the vat trong khung goc robot
        return P @ T[:3, :3].T + T[:3, 3]

    def _grasp(self):
        P = self._world_pts(self.cfg["pick"])
        xy = P[:, :2] - P[:, :2].mean(0)
        evals, evecs = np.linalg.eigh(np.cov(xy.T))
        long_ax = np.r_[evecs[:, 1], 0.0]
        proj = xy @ evecs[:, 1]
        mid = np.abs(proj - 0.5 * (proj.min() + proj.max())) < 0.012   # lat cat giua theo truc dai
        top = P[:, 2].max()
        if self.cfg.get("rim_grasp"):
            c = P[:, :2].mean(0)
            r = np.linalg.norm(P[:, :2] - c, axis=1).max()
            base_xy = self.T0[:2, 3]
            d = base_xy - c
            d /= np.linalg.norm(d)                                     # vanh phia gan robot
            g = np.r_[c + d * (r - 0.012), top - 0.02]
            close_ax = np.r_[d, 0.0]                                   # kep theo phuong ban kinh
        else:
            sl = P[mid]
            g = np.r_[sl[:, :2].mean(0), 0.5 * (sl[:, 2].min() + sl[:, 2].max())]
            # kep o tam lat cat giua than (vat cao nhu tuong do choi: khong kep dau/duoi manh o dinh)
            g[2] = np.clip(g[2], P[:, 2].min() + 0.015, top - 0.015)
            close_ax = np.cross([0, 0, 1.0], long_ax)                  # vuong goc truc dai, nam ngang
        approach = np.array([0, 0, -1.0])
        y0 = close_ax / np.linalg.norm(close_ax)
        # gripper doi xung (y hoac -y deu kep duoc) -> chon chieu IK dat huong tot nhat (khop 7 gioi han +-166 deg)
        best = None
        for y in (y0, -y0):
            R = np.column_stack([approach, y, np.cross(approach, y)])
            q, ep, er = self.kin.ik(pose(R, g - TCP_OFFSET * approach), self.q_rest, iters=200, null_q=self.q_rest)
            if best is None or ep + 0.05 * er < best[0]:
                best = (ep + 0.05 * er, R, ep, er)
        _, R, ep, er = best
        print(f"DBG IK tu the kep: lech {ep * 1000:.1f} mm, {np.degrees(er):.1f} deg", flush=True)
        return g, R, top, P[:, 2].min()

    def plan(self):
        g, R, top, bottom = self._grasp()
        T_wb = np.linalg.inv(self.T_bw)
        print(f"DBG diem kep (the gioi) {(T_wb @ np.r_[g, 1])[:3].round(3)}, vat cao {top - bottom:.3f} m, "
              f"phuong kep {(T_wb[:3, :3] @ R[:, 1]).round(2)}", flush=True)
        D = self._world_pts(self.cfg["place"])
        dst = np.r_[D[:, :2].mean(0), D[:, 2].max()]
        lift_z = max(top, dst[2]) + 0.15

        def ee(p_grasp):                                               # diem kep -> base_link
            return p_grasp - TCP_OFFSET * R[:, 0]

        # day vat cach mat dich place_lift: diem kep = dst_top + place_lift + (diem kep - day vat)
        place_g = np.r_[dst[:2], dst[2] + self.cfg.get("place_lift", 0.04) + (g[2] - bottom)]
        waypoints = [  # (diem kep dich, gripper, thoi gian s)
            (np.r_[g[:2], lift_z], 0, 2.5),
            (g, 0, 1.6),
            (g, 1, 0.8),
            (np.r_[g[:2], lift_z], 1, 1.5),
            (np.r_[dst[:2], lift_z], 1, 2.2),
            (place_g, 1, 1.4),
            (place_g, 0, 0.8),
            (np.r_[dst[:2], lift_z], 0, 1.2),
        ]
        p_prev, R_prev, first = self.T0[:3, 3], self.T0[:3, :3], True
        for p_g, grip, dur in waypoints:
            p_goal = ee(p_g)
            n = max(1, int(round(dur * 15)))
            for k in range(1, n + 1):
                s = _minjerk(k / n)
                Rk = _slerp_R(R_prev, R, s) if first else R
                yield pose(Rk, p_prev + s * (p_goal - p_prev)), grip
            p_prev, R_prev, first = p_goal, R, False


def make_expert(task, sim, kin, T_bw):
    return PickPlaceExpert(TASKS[task], sim, kin, T_bw)
