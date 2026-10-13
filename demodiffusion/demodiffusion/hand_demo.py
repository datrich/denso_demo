"""Demo "nguoi" tong hop (21 keypoint kieu MANO) + kinematic retargeting dung nhu DemoDiffusion.

Tong hop: tu quy dao dau cuoi mong muon (Robotiq base_link, +x = huong tiep can, +y = phuong kep) va do dong
gripper, dung ban tay 21 diem sao cho cong thuc retarget cua tac gia (deploy/preprocess.py) cho lai dung tu the do;
sau do them sai lech "kieu nguoi": lech co dinh (sai khac than hinh / hieu chinh camera), troi cham (tay nguoi
khong di dung quy dao robot), rung tung keypoint (sai so uoc luong tay kieu HaMeR), lech huong, lech thoi diem nam.

Retarget (giu nguyen cong thuc tac gia): EMA alpha=0.1 tren keypoint; A = co tay (0), B = dau ngon cai (4),
C = trung binh dau 4 ngon con lai (8, 12, 16, 20); z = norm((C-A)+(B-A)); x = -norm(cross(C-A, B-A)); y = z x x;
dau cuoi = A - R [0, 0, 0.062]; gripper = 1 - clip(|B-C| / 0.085) -> dong neu > nguong 0.2.
"""

from __future__ import annotations

import numpy as np

MAX_GRIPPER_WIDTH = 0.085          # Robotiq 2F-85 (tac gia dung hang so nay)
WRIST_OFFSET = 0.062               # tac gia: eef = co tay - R [0, 0, 0.062]
TIP_IDX = (4, 8, 12, 16, 20)
# Ban tay (khung tay retarget): z = huong tiep can, y = phuong kep (ngon cai o +y), x = phap tuyen mu ban tay
# R_base_link = R_hand @ M   (base_link: x = tiep can, y = phuong kep, z = x * y)
M_HAND_TO_BASE = np.array([[0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0]])


def hand_frame_from_base(R_base):
    return R_base @ M_HAND_TO_BASE.T


def base_frame_from_hand(R_hand):
    return R_hand @ M_HAND_TO_BASE


def _rotvec(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _smooth_noise(rng, T, dim, sigma, corr_frames):
    """Nhieu Gauss loc thong thap (troi cham) bien do ~sigma, tuong quan ~corr_frames khung."""
    white = rng.normal(size=(T + 4 * corr_frames, dim))
    k = np.exp(-0.5 * (np.arange(-2 * corr_frames, 2 * corr_frames + 1) / max(corr_frames, 1)) ** 2)
    k /= np.sqrt((k ** 2).sum())
    out = np.stack([np.convolve(white[:, d], k, mode="same") for d in range(dim)], axis=1)
    return sigma * out[2 * corr_frames: 2 * corr_frames + T]


HUMAN_NOISE = {
    # lech co dinh (m), troi cham (m), rung keypoint (m), lech huong (rad), lech thoi diem nam (khung, 15 Hz)
    "none": dict(offset=0.0, drift=0.0, jitter=0.0, rot=0.0, grasp_shift=0),
    "realistic": dict(offset=0.01, drift=0.01, jitter=0.004, rot=np.radians(5), grasp_shift=3),
    "plus5cm": dict(offset=0.05, drift=0.01, jitter=0.004, rot=np.radians(5), grasp_shift=3),  # nhu bai bao: +5 cm
}


def synthesize_hand(T_base_list, grip_list, rng=None, noise="realistic", grasp_width=0.03):
    """T_base_list: (T, 4, 4) tu the base_link mong muon; grip_list: (T,) 0 = mo, 1 = dong -> keypoints (T, 21, 3).

    grasp_width: khoang cach ngon cai - cac ngon con lai khi dang cam vat (tay nguoi om quanh vat, khong khep kin).
    """
    rng = np.random.default_rng(rng)
    cfg = HUMAN_NOISE[noise] if isinstance(noise, str) else noise
    T = len(T_base_list)
    grip = np.asarray(grip_list, float)
    if cfg["grasp_shift"]:  # nguoi nam/nha som hoac muon vai khung
        grip = np.roll(grip, int(rng.integers(-cfg["grasp_shift"], cfg["grasp_shift"] + 1)))
    grip = np.convolve(np.r_[np.full(3, grip[0]), grip, np.full(3, grip[-1])], np.ones(4) / 4, mode="same")[3:3 + T]
    offset = rng.normal(scale=cfg["offset"], size=3) if cfg["offset"] else np.zeros(3)
    drift = _smooth_noise(rng, T, 3, cfg["drift"], 15) if cfg["drift"] else np.zeros((T, 3))
    rot_noise = _smooth_noise(rng, T, 3, cfg["rot"], 20) if cfg["rot"] else np.zeros((T, 3))
    kps = np.zeros((T, 21, 3))
    for t in range(T):
        Rb, p = T_base_list[t][:3, :3], T_base_list[t][:3, 3]
        Rh = _rotvec(rot_noise[t]) @ hand_frame_from_base(Rb)
        x, y, z = Rh[:, 0], Rh[:, 1], Rh[:, 2]
        A = p + WRIST_OFFSET * z + offset + drift[t]
        # tay mo rong hon gripper mot chut; khi cam vat thi ngon om quanh vat (~grasp_width)
        h = 0.5 * (grip[t] * grasp_width + (1.0 - grip[t]) * MAX_GRIPPER_WIDTH * 1.15)
        B = A + 0.065 * z + h * y                                       # dau ngon cai
        C = A + 0.085 * z - h * y                                       # trung binh 4 dau ngon con lai
        kps[t, 0] = A
        for j, f in zip((1, 2, 3), (0.3, 0.6, 0.85)):                   # ngon cai: CMC, MCP, IP
            kps[t, j] = A + f * (B - A) + 0.01 * x * (1 - f)
        kps[t, 4] = B
        for k, s in enumerate((0.03, 0.01, -0.01, -0.03)):              # tro, giua, ap ut, ut (mean = C)
            tip = C + s * x
            base = 4 * k + 5
            mcp = A + 0.045 * z + 0.6 * s * x
            kps[t, base] = mcp
            kps[t, base + 1] = mcp + 0.45 * (tip - mcp)
            kps[t, base + 2] = mcp + 0.75 * (tip - mcp)
            kps[t, base + 3] = tip
    if cfg["jitter"]:
        kps += rng.normal(scale=cfg["jitter"], size=kps.shape)
    return kps


def ema(data, alpha=0.1):
    out = np.zeros_like(data)
    out[0] = data[0]
    for t in range(1, len(data)):
        out[t] = alpha * data[t] + (1 - alpha) * out[t - 1]
    return out


def retarget(keypoints, alpha=0.1, gripper_threshold=0.2):
    """Cong thuc cua tac gia -> (T_base (T,4,4), gripper (T,) nhi phan 0/1, gripper_raw (T,))."""
    kps = ema(np.asarray(keypoints, float), alpha) if alpha < 1 else np.asarray(keypoints, float)
    T = len(kps)
    poses = np.zeros((T, 4, 4))
    g_raw = np.zeros(T)
    for t in range(T):
        A, B = kps[t, 0], kps[t, 4]
        C = kps[t, [8, 12, 16, 20]].mean(0)
        z = (C - A) + (B - A)
        z /= np.linalg.norm(z) + 1e-8
        x = -np.cross(C - A, B - A)
        x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        Rh = np.column_stack([x, y, z])
        poses[t] = np.eye(4)
        poses[t, :3, :3] = base_frame_from_hand(Rh)
        poses[t, :3, 3] = A - Rh @ np.array([0, 0, WRIST_OFFSET])
        g_raw[t] = 1 - np.clip(np.linalg.norm(B - C) / MAX_GRIPPER_WIDTH, 0, 1)
    return poses, (g_raw > gripper_threshold).astype(float), g_raw
