"""DemoDiffusion trong Isaac Lab Arena (robot DROID: Franka + Robotiq 2F-85, pi0-DROID jointpos qua openpi).

Hai che do (chay bang python cua venv Arena, server openpi phai dang chay neu dung pi0):

  record  - Ghi "demo nguoi" tong hop: dat canh demo (seed rieng), expert co thong tin dac quyen thuc hien task
            (quy dao Cartesian min-jerk + IK), luu quy dao base_link + gripper, roi tong hop ban tay 21 keypoint
            (khung goc robot) kem sai lech kieu nguoi -> <out>/demo_<task>.npz
  eval    - Chay cac phuong phap tren cac canh test (seed khac, vat dat lech so voi luc demo):
              pi0           : pi0-DROID nguyen ban (s* = 1)
              kinematic     : phat lai quy dao retarget ho mach (s* = 0)
              demodiffusion : khu nhieu quy dao retarget tu s* (mac dinh 0.2 va 0.4)
            Do dai episode = do dai demo (nhu bai bao). Ghi video camera + results.json.

    python demodiffusion/arena/run_demodiffusion.py --mode record --task banana_on_plate --env_spec <yaml>
    python demodiffusion/arena/run_demodiffusion.py --mode eval --task banana_on_plate --env_spec <yaml> \
        --methods pi0 kinematic dd0.2 dd0.4 --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # .../demodiffusion (goi demodiffusion.*)

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402

CONTROL_HZ = 15
H, OPEN_LOOP = 10, 8          # pi0 chunk 10, chay ho mach 8 (bang V cua bai bao)


def parse():
    p = get_isaaclab_arena_cli_parser()
    p.add_argument("--mode", required=True, choices=["record", "eval"])
    p.add_argument("--task", required=True, help="ten task trong demodiffusion/arena/tasks.py")
    p.add_argument("--out", default=os.environ.get("DD_OUT", "dd_out"))
    p.add_argument("--demo_seed", type=int, default=0)
    p.add_argument("--human_noise", default="realistic", choices=["none", "realistic", "plus5cm"])
    p.add_argument("--methods", nargs="*", default=["pi0", "kinematic", "dd0.2", "dd0.4"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--first_seed", type=int, default=100)
    p.add_argument("--policy_host", default="127.0.0.1")
    p.add_argument("--policy_port", type=int, default=8000)
    p.add_argument("--video", type=int, default=3, help="so episode dau moi phuong phap duoc ghi video")
    p.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    p.add_argument("--debug", action="store_true")
    return p


# ---------------------------------------------------------------- tien ich mo phong
def T_from_pq(p, q_xyzw):
    from demodiffusion.franka_kinematics import pose, quat_wxyz_to_R
    x, y, z, w = q_xyzw
    return pose(quat_wxyz_to_R([w, x, y, z]), p)


class SimHandle:
    """Truy cap trang thai robot/vat trong env Arena (env_id 0)."""

    def __init__(self, env):
        import warp as wp

        self.wp = wp
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        names = list(self.robot.data.body_names)
        self.i_link0 = names.index("panda_link0")
        self.i_ee = names.index("base_link")  # Robotiq base_link (cung frame voi quan sat eef_pos/eef_quat)
        jn = list(self.robot.data.joint_names)
        self.i_arm = [jn.index(f"panda_joint{k}") for k in range(1, 8)]

    def _t(self, a):
        return self.wp.to_torch(a)[0].detach().cpu().numpy()

    def q(self):
        return self._t(self.robot.data.joint_pos)[self.i_arm]

    def body_T(self, idx):
        return T_from_pq(self._t(self.robot.data.body_pos_w)[idx], self._t(self.robot.data.body_quat_w)[idx])

    def T_world_base(self):
        return self.body_T(self.i_link0)

    def T_world_ee(self):
        return self.body_T(self.i_ee)

    def obj_T(self, name):
        o = self.env.scene[name]
        return T_from_pq(self._t(o.data.root_pos_w), self._t(o.data.root_quat_w))

    def tcp_world(self):
        """Diem kep ma Arena dinh nghia (ee_frame/end_effector = base_link + 0.131 x)."""
        try:
            return self._t(self.env.scene["ee_frame"].data.target_pos_w)[0]
        except Exception:  # noqa: BLE001
            return np.full(3, np.nan)

    def term(self, name):
        tm = self.env.termination_manager
        return bool(tm.get_term(name)[0]) if name in tm.active_terms else False


def calibrate(sim, kin):
    """Do bien doi link8 -> Robotiq base_link tu mo phong (kiem tra quy uoc quaternion xyzw luon)."""
    T_b = np.linalg.inv(sim.T_world_base()) @ sim.T_world_ee()
    kin.calibrate_tool(sim.q(), T_b)
    return T_b


# ---------------------------------------------------------------- video
class VideoWriter:
    def __init__(self, path, ffmpeg, fps=CONTROL_HZ):
        self.path, self.ffmpeg, self.fps, self.proc = path, ffmpeg, fps, None

    def add(self, frame):
        frame = np.ascontiguousarray(frame[..., :3]).astype(np.uint8)
        if self.proc is None:
            h, w = frame.shape[:2]
            self.proc = subprocess.Popen([self.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
                                          "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(self.path)],
                                         stdin=subprocess.PIPE)
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


def camera_frame(obs, text=None):
    from PIL import Image, ImageDraw

    cam = obs["camera_obs"]
    ext = cam["external_camera_rgb"][0].detach().cpu().numpy()[..., :3]
    wri = cam["wrist_camera_rgb"][0].detach().cpu().numpy()[..., :3]
    h = ext.shape[0]
    wri = np.asarray(Image.fromarray(wri.astype(np.uint8)).resize((int(wri.shape[1] * h / wri.shape[0]), h)))
    img = Image.fromarray(np.concatenate([ext.astype(np.uint8), wri], axis=1))
    if text:
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 9 * len(text) + 12, 22], fill=(0, 0, 0))
        d.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(img)


# ---------------------------------------------------------------- che do record
def run_record(args, env, task):
    from demodiffusion.franka_kinematics import FrankaKinematics
    from demodiffusion.hand_demo import synthesize_hand

    from tasks import make_expert

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=args.demo_seed)
    sim = SimHandle(env)
    kin = FrankaKinematics()
    calibrate(sim, kin)
    T_wb = sim.T_world_base()
    T_bw = np.linalg.inv(T_wb)
    expert = make_expert(task, sim, kin, T_bw)
    video = VideoWriter(out / f"demo_{task}_expert.mp4", args.ffmpeg)
    ee_traj, grip_traj, q = [], [], sim.q()
    success, step = False, 0
    T_e = sim.T_world_ee()
    raw = sim._t(sim.robot.data.body_quat_w)
    print(f"DBG quat tho: panda_link0 {raw[sim.i_link0].round(4)} base_link {raw[sim.i_ee].round(4)}; "
          f"root {sim._t(sim.robot.data.root_quat_w).round(4)}; T_tool R=\n{kin.T_tool[:3, :3].round(3)}", flush=True)
    print(f"DBG base_link (the gioi): x={T_e[:3, 0].round(3)} y={T_e[:3, 1].round(3)} z={T_e[:3, 2].round(3)} "
          f"p={T_e[:3, 3].round(3)} | FK sau hieu chinh lech "
          f"{np.linalg.norm((T_wb @ kin.fk(sim.q()))[:3, 3] - T_e[:3, 3]) * 1000:.2f} mm", flush=True)
    for step, (T_goal_b, grip) in enumerate(expert.plan()):
        q, ep, er = kin.ik(T_goal_b, q, null_q=expert.q_rest)
        import torch

        act = torch.tensor([[*q, float(grip)]], dtype=torch.float32, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(act)
        ee_traj.append(T_bw @ sim.T_world_ee())     # tu the base_link THUC (khung goc robot)
        grip_traj.append(float(grip))
        if args.debug and step % 10 == 0:
            T_act = T_bw @ sim.T_world_ee()
            tcp_arena = sim.tcp_world()
            tcp_mine = (sim.T_world_ee() @ np.r_[0.131, 0, 0, 1])[:3]
            o = sim.obj_T(expert.object_names[0])[:3, 3]
            from demodiffusion.franka_kinematics import rot_err

            print(f"DBG t={step:3d} grip={grip:.0f} gpos={float(obs['policy']['gripper_pos'][0, 0]):.2f} "
                  f"lech dat {np.linalg.norm(T_act[:3, 3] - T_goal_b[:3, 3]) * 1000:5.1f} mm "
                  f"{np.degrees(np.linalg.norm(rot_err(T_goal_b[:3, :3], T_act[:3, :3]))):4.1f} deg "
                  f"IK {ep * 1000:4.1f} mm {np.degrees(er):4.1f} deg | "
                  f"TCP arena {tcp_arena.round(3)} cua minh {tcp_mine.round(3)} | vat {o.round(3)}", flush=True)
        video.add(camera_frame(obs, f"expert demo t={step / CONTROL_HZ:4.1f}s"))
        if sim.term("success"):
            success = True
        if bool(terminated[0]) or bool(truncated[0]):
            break
    video.close()
    ee_traj, grip_traj = np.array(ee_traj), np.array(grip_traj)
    kps = synthesize_hand(ee_traj, grip_traj, rng=args.demo_seed, noise=args.human_noise)
    scene = {n: (T_bw @ sim.obj_T(n)).tolist() for n in expert.object_names}
    np.savez(out / f"demo_{task}.npz", keypoints=kps, ee_traj=ee_traj, grip=grip_traj, T_world_base=T_wb,
             expert_success=success, human_noise=args.human_noise, demo_seed=args.demo_seed,
             prompt=env.unwrapped.get_language_instruction(), scene=json.dumps(scene))
    print(f"RESULT record {task}: expert {'THANH CONG' if success else 'THAT BAI'}, {len(ee_traj)} buoc "
          f"({len(ee_traj) / CONTROL_HZ:.1f} s), tay nguoi '{args.human_noise}' -> {out / f'demo_{task}.npz'}", flush=True)


# ---------------------------------------------------------------- che do eval
def retarget_to_joints(keypoints, kin, q0, q_rest):
    """Keypoint tay (khung goc robot) -> tu the base_link (cong thuc tac gia) -> IK -> q tuyet doi (T, 7) + gripper."""
    from demodiffusion.hand_demo import retarget

    poses, grip, _ = retarget(keypoints)
    qs, q, errs = [], np.array(q0[:7], float), []
    for Tg in poses:
        q, ep, er = kin.ik(Tg, q, null_q=q_rest)
        qs.append(q.copy())
        errs.append(ep)
    return np.array(qs), grip, poses, float(np.max(errs))


def run_eval(args, env, task):
    import torch
    from demodiffusion.franka_kinematics import FrankaKinematics
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    from tasks import TASKS

    out = Path(args.out)
    demo = np.load(out / f"demo_{task}.npz", allow_pickle=True)
    kps, prompt = demo["keypoints"], str(demo["prompt"])
    T_steps = len(kps)
    client = None
    if any(m != "kinematic" for m in args.methods):
        client = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    results = {"task": task, "prompt": prompt, "demo_steps": T_steps, "human_noise": str(demo["human_noise"]),
               "methods": {}}
    for method in args.methods:
        s_star = {"pi0": 1.0, "kinematic": 0.0}.get(method)
        if s_star is None:
            s_star = float(method.replace("dd", ""))
        succ = []
        for ep in range(args.episodes):
            seed = args.first_seed + ep
            obs, _ = env.reset(seed=seed)
            sim = SimHandle(env)
            kin = FrankaKinematics()
            calibrate(sim, kin)
            q_rest = TASKS[task].get("q_rest", sim.q())
            q_ret, g_ret, _, ik_err = retarget_to_joints(kps, kin, sim.q(), q_rest)
            video = VideoWriter(out / f"{task}_{method}_ep{ep}.mp4", args.ffmpeg) if ep < args.video else None
            chunk, k_in_chunk, success, t_success = None, OPEN_LOOP, False, None
            for t in range(T_steps):
                if method == "kinematic":
                    a = np.r_[q_ret[t], g_ret[t]]
                else:
                    if k_in_chunk >= OPEN_LOOP:
                        po = obs["policy"]
                        cam = obs["camera_obs"]
                        req = {
                            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                                cam["external_camera_rgb"][0].detach().cpu().numpy()[..., :3], 224, 224),
                            "observation/wrist_image_left": image_tools.resize_with_pad(
                                cam["wrist_camera_rgb"][0].detach().cpu().numpy()[..., :3], 224, 224),
                            "observation/joint_position": po["joint_pos"][0].detach().cpu().numpy(),
                            "observation/gripper_position": po["gripper_pos"][0].detach().cpu().numpy(),
                            "prompt": prompt,
                            "time": s_star,
                        }
                        if s_star < 1.0:  # demo tai buoc t: vi tri khop t+1..t+H, gripper t..t+H-1 (nhu tac gia)
                            idx = np.minimum(np.arange(t + 1, t + 1 + H), T_steps - 1)
                            gidx = np.minimum(np.arange(t, t + H), T_steps - 1)
                            req["retargeted_actions"] = np.c_[q_ret[idx], g_ret[gidx]].astype(np.float32)
                        chunk = np.asarray(client.infer(req)["actions"])
                        k_in_chunk = 0
                    a = chunk[k_in_chunk].copy()
                    k_in_chunk += 1
                    a[7] = 1.0 if a[7] > 0.5 else 0.0  # nhi phan hoa gripper (nhu tac gia)
                act = torch.tensor(a[None], dtype=torch.float32, device=env.unwrapped.device)
                obs, _, terminated, truncated, _ = env.step(act)
                if video is not None:
                    video.add(camera_frame(obs, f"{method}  ep{ep}  t={t / CONTROL_HZ:4.1f}s"
                                           + ("  THANH CONG" if success else "")))
                if sim.term("success") and not success:
                    success, t_success = True, t / CONTROL_HZ
                if bool(terminated[0]) or bool(truncated[0]):
                    break
            if video is not None:
                video.close()
            succ.append(success)
            print(f"RESULT {task} {method} ep{ep} seed{seed}: {'THANH CONG' if success else 'that bai'}"
                  + (f" luc {t_success:.1f}s" if success else "") + f" (IK retarget lech max {ik_err * 1000:.1f} mm)",
                  flush=True)
        rate = float(np.mean(succ)) * 100
        results["methods"][method] = {"s_star": s_star, "success": succ, "success_rate": rate}
        print(f"RESULT {task} {method}: {sum(succ)}/{len(succ)} = {rate:.0f}%", flush=True)
        (out / f"results_{task}.json").write_text(json.dumps(results, indent=2))


def main():
    parser = parse()
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    with SimulationAppContext(args):
        from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

        sys.path.insert(0, str(HERE.parent))
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args, hydra = parser.parse_known_args()
        args.enable_cameras = True
        builder = get_arena_builder_from_cli(args, hydra_overrides=hydra)
        env = builder.make_registered()
        t0 = time.time()
        (run_record if args.mode == "record" else run_eval)(args, env, args.task)
        print(f"Thoi gian chay {time.time() - t0:.0f} s", flush=True)
        env.close()


if __name__ == "__main__":
    main()
