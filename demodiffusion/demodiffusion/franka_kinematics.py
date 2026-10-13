"""FK / IK so (numpy) cho Franka Panda (tham so DH Craig cua Franka Emika) + khau gan tool tuy y.

Dung cho DemoDiffusion: doi quy dao dau cuoi (retarget tu tay nguoi) sang vi tri khop tuyet doi
(khong gian hanh dong cua pi0-DROID jointpos). Bien doi link8 -> dau cuoi (Robotiq base_link) duoc hieu
chinh tu mo phong (calibrate_tool).
"""

from __future__ import annotations

import numpy as np

# Modified DH (Craig): a_{i-1}, d_i, alpha_{i-1}  (https://frankaemika.github.io/docs/control_parameters.html)
DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
FLANGE_D = 0.107
Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1.0]])


def fk_link8(q):
    """T (4x4) panda_link0 -> panda_link8 (mat bich)."""
    T = np.eye(4)
    for (a, d, al), th in zip(DH, q[:7]):
        T = T @ _mdh(a, d, al, th)
    flange = np.eye(4)
    flange[2, 3] = FLANGE_D
    return T @ flange


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def pose(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rot_err(R_goal, R):
    Re = R_goal @ R.T
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


class FrankaKinematics:
    """FK/IK trong khung goc robot (panda_link0) cho dau cuoi = link8 @ T_tool."""

    def __init__(self, T_tool=None):
        self.T_tool = np.eye(4) if T_tool is None else T_tool

    def calibrate_tool(self, q, T_base_ee):
        """T_tool = inv(FK_link8(q)) @ T_base_ee, voi T_base_ee do tu mo phong o cung cau hinh q."""
        self.T_tool = np.linalg.inv(fk_link8(q)) @ T_base_ee
        return self.T_tool

    def fk(self, q):
        return fk_link8(q) @ self.T_tool

    def jacobian(self, q, eps=1e-6):
        T0 = self.fk(q)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.array(q, float)
            dq[i] += eps
            T = self.fk(dq)
            J[:3, i] = (T[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = rot_err(T[:3, :3], T0[:3, :3]) / eps
        return J

    def ik(self, T_goal, q0, iters=100, damping=0.05, w_rot=0.5, tol=(5e-4, 5e-3), null_q=None, null_gain=0.05):
        """DLS co gioi han khop; null_q keo ve tu the uu tien (tranh khop 7 quay vo han)."""
        q = np.array(q0[:7], float)
        for _ in range(iters):
            T = self.fk(q)
            e = np.concatenate([T_goal[:3, 3] - T[:3, 3], w_rot * rot_err(T_goal[:3, :3], T[:3, :3])])
            if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) / w_rot < tol[1]:
                break
            J = self.jacobian(q)
            J[3:] *= w_rot
            JJt = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            if null_q is not None:
                N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
                dq += N @ (null_gain * (np.asarray(null_q[:7]) - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), Q_LO + 1e-3, Q_HI - 1e-3)
        T = self.fk(q)
        return q, float(np.linalg.norm(T_goal[:3, 3] - T[:3, 3])), float(np.linalg.norm(rot_err(T_goal[:3, :3], T[:3, :3])))
