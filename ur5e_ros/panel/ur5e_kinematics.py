"""Dong hoc thuan / nghich giai tich cho UR5e.

Quy uoc:
- Goc khop trung voi ur_description (ROS) va asset UR5e cua Isaac Sim.
- Pose tra ve / nhan vao duoc bieu dien trong khung ``base_link`` (khung goc cua
  robot trong URDF/USD). Khung DH ``base`` xoay 180 do quanh Z so voi ``base_link``.
- Khung cuoi la ``tool0`` (mat bich), cong them TCP offset doc truc Z neu co.
- IK theo Hawkins, "Analytic Inverse Kinematics for the Universal Robots UR-5/UR-10".
"""

from __future__ import annotations

import math

import numpy as np

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Thong so DH cua UR5e (m), theo Universal Robots
D1 = 0.1625
A2 = -0.425
A3 = -0.3922
D4 = 0.1333
D5 = 0.0997
D6 = 0.0996

_DH_A = (0.0, A2, A3, 0.0, 0.0, 0.0)
_DH_D = (D1, 0.0, 0.0, D4, D5, D6)
_DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)

# Gioi han khop cua UR5e (rad): elbow +-pi, cac khop khac +-2pi
JOINT_LIMITS = np.array(
    [
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
        [-math.pi, math.pi],
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
        [-2 * math.pi, 2 * math.pi],
    ]
)

# base_link -> base (khung DH): xoay pi quanh Z
_T_BASELINK_BASE = np.diag([-1.0, -1.0, 1.0, 1.0])


def _dh(i: int, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(_DH_ALPHA[i]), math.sin(_DH_ALPHA[i])
    a, d = _DH_A[i], _DH_D[i]
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _tcp(tcp_offset_z: float) -> np.ndarray:
    t = np.eye(4)
    t[2, 3] = tcp_offset_z
    return t


def fk(q, tcp_offset_z: float = 0.0) -> np.ndarray:
    """Ma tran 4x4 cua TCP trong khung base_link."""
    t = _T_BASELINK_BASE.copy()
    for i in range(6):
        t = t @ _dh(i, float(q[i]))
    return t @ _tcp(tcp_offset_z)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def ik_all(pose: np.ndarray, tcp_offset_z: float = 0.0, q6_hint: float = 0.0) -> list[np.ndarray]:
    """Tat ca nghiem IK (toi da 8), moi goc nam trong [-pi, pi)."""
    t06 = np.linalg.inv(_T_BASELINK_BASE) @ pose @ np.linalg.inv(_tcp(tcp_offset_z))
    p = t06[:3, 3]
    a_axis = t06[:3, 2]

    p05 = p - D6 * a_axis
    r = math.hypot(p05[0], p05[1])
    if r < abs(D4):
        return []
    psi = math.atan2(p05[1], p05[0])
    phi = math.acos(D4 / r)

    t60 = np.linalg.inv(t06)
    sols = []
    for q1 in (psi + phi + math.pi / 2, psi - phi + math.pi / 2):
        s1, c1 = math.sin(q1), math.cos(q1)
        c5 = (p[0] * s1 - p[1] * c1 - D4) / D6
        if abs(c5) > 1.0 + 1e-9:
            continue
        c5 = max(-1.0, min(1.0, c5))
        for q5 in (math.acos(c5), -math.acos(c5)):
            s5 = math.sin(q5)
            if abs(s5) < 1e-9:
                q6 = q6_hint  # ky di co tay: q6 tu do
            else:
                q6 = math.atan2(
                    (-t60[1, 0] * s1 + t60[1, 1] * c1) / s5,
                    (t60[0, 0] * s1 - t60[0, 1] * c1) / s5,
                )
            t14 = np.linalg.inv(_dh(0, q1)) @ t06 @ np.linalg.inv(_dh(4, q5) @ _dh(5, q6))
            p13 = t14 @ np.array([0.0, -D4, 0.0, 1.0])
            p13n2 = p13[0] ** 2 + p13[1] ** 2
            c3 = (p13n2 - A2**2 - A3**2) / (2 * A2 * A3)
            if abs(c3) > 1.0 + 1e-9:
                continue
            c3 = max(-1.0, min(1.0, c3))
            p13n = math.sqrt(p13n2)
            for q3 in (math.acos(c3), -math.acos(c3)):
                q2 = -math.atan2(p13[1], -p13[0]) + math.asin(
                    max(-1.0, min(1.0, A3 * math.sin(q3) / p13n))
                )
                t34 = np.linalg.inv(_dh(1, q2) @ _dh(2, q3)) @ t14
                q4 = math.atan2(t34[1, 0], t34[0, 0])
                sols.append(np.array([_wrap(v) for v in (q1, q2, q3, q4, q5, q6)]))
    # Chi giu nghiem thuc su khop pose (loai sai so so hoc / nhanh khong hop le)
    return [q for q in sols if pose_error(fk(q, tcp_offset_z), pose) < 1e-6]


def jacobian(q, tcp_offset_z: float = 0.0) -> np.ndarray:
    """Jacobian hinh hoc 6x6 ([v; w]) trong khung base_link."""
    t = _T_BASELINK_BASE.copy()
    origins, axes = [], []
    for i in range(6):
        origins.append(t[:3, 3].copy())
        axes.append(t[:3, 2].copy())
        t = t @ _dh(i, float(q[i]))
    pe = (t @ _tcp(tcp_offset_z))[:3, 3]
    jac = np.zeros((6, 6))
    for i in range(6):
        jac[:3, i] = np.cross(axes[i], pe - origins[i])
        jac[3:, i] = axes[i]
    return jac


def twist_error(t_cur: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    """Sai lech [dp; dw] (dw la vector truc-goc) tu t_cur toi t_target, trong khung base_link."""
    dp = t_target[:3, 3] - t_cur[:3, 3]
    r = t_target[:3, :3] @ t_cur[:3, :3].T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        dw = np.zeros(3)
    elif math.pi - angle < 1e-6:
        w, v = np.linalg.eigh(r + r.T)
        dw = v[:, np.argmax(w)] * angle
    else:
        dw = angle / (2 * math.sin(angle)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return np.concatenate([dp, dw])


def ik_dls(target: np.ndarray, q_init, tcp_offset_z: float = 0.0, iterations: int = 10, damping: float = 0.02):
    """IK lap damped least-squares tu q_init: on dinh khi gan ky di (dung cho MoveL/jog).

    Tra ve (q, sai_so_vi_tri_m, sai_so_goc_rad).
    """
    q = np.asarray(q_init, dtype=float).copy()
    for _ in range(iterations):
        err = twist_error(fk(q, tcp_offset_z), target)
        if np.linalg.norm(err[:3]) < 1e-6 and np.linalg.norm(err[3:]) < 1e-5:
            break
        jac = jacobian(q, tcp_offset_z)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    err = twist_error(fk(q, tcp_offset_z), target)
    return q, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))


def pose_error(t_a: np.ndarray, t_b: np.ndarray) -> float:
    dp = np.linalg.norm(t_a[:3, 3] - t_b[:3, 3])
    dr = np.linalg.norm(t_a[:3, :3] - t_b[:3, :3])
    return float(dp + dr)


def _closest_branch(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray | None:
    """Doi tung khop +-2pi de gan q_ref nhat ma van trong gioi han."""
    out = np.empty(6)
    for i in range(6):
        best = None
        for k in (-1, 0, 1):
            v = q[i] + 2 * math.pi * k
            if JOINT_LIMITS[i, 0] - 1e-9 <= v <= JOINT_LIMITS[i, 1] + 1e-9:
                if best is None or abs(v - q_ref[i]) < abs(best - q_ref[i]):
                    best = v
        if best is None:
            return None
        out[i] = best
    return out


def ik_closest(pose: np.ndarray, q_ref, tcp_offset_z: float = 0.0) -> np.ndarray | None:
    """Nghiem IK gan q_ref nhat (theo tong |dq| co trong so), hoac None neu khong toi duoc."""
    q_ref = np.asarray(q_ref, dtype=float)
    weights = np.array([1.0, 1.0, 1.0, 0.6, 0.6, 0.4])
    best, best_cost = None, math.inf
    for q in ik_all(pose, tcp_offset_z, q6_hint=float(q_ref[5])):
        qb = _closest_branch(q, q_ref)
        if qb is None:
            continue
        cost = float(np.sum(weights * np.abs(qb - q_ref)))
        if cost < best_cost:
            best, best_cost = qb, cost
    return best


# ---------- tien ich xoay ----------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (quy uoc ROS)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(r: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def make_pose(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(*rpy)
    t[:3, 3] = xyz
    return t


def interpolate_pose(t_a: np.ndarray, t_b: np.ndarray, s: float) -> np.ndarray:
    """Noi suy tuyen tinh vi tri + slerp huong (s trong [0, 1])."""
    t = np.eye(4)
    t[:3, 3] = (1 - s) * t_a[:3, 3] + s * t_b[:3, 3]
    r_rel = t_a[:3, :3].T @ t_b[:3, :3]
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r_rel) - 1) / 2)))
    if angle < 1e-9:
        t[:3, :3] = t_a[:3, :3]
    else:
        axis = np.array([r_rel[2, 1] - r_rel[1, 2], r_rel[0, 2] - r_rel[2, 0], r_rel[1, 0] - r_rel[0, 1]])
        if np.linalg.norm(axis) < 1e-9:  # angle ~ pi
            w, v = np.linalg.eigh(r_rel + r_rel.T)
            axis = v[:, np.argmax(w)]
        t[:3, :3] = t_a[:3, :3] @ axis_angle(axis, s * angle)
    return t


def rotation_angle_between(t_a: np.ndarray, t_b: np.ndarray) -> float:
    r_rel = t_a[:3, :3].T @ t_b[:3, :3]
    return math.acos(max(-1.0, min(1.0, (np.trace(r_rel) - 1) / 2)))
