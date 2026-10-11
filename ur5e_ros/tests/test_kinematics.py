import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panel"))

import ur5e_kinematics as kin  # noqa: E402


def test_fk_zero_pose_matches_ur_description():
    # Tai q = 0, tool0 trong base_link (ur_description): x = -(a2 + a3), y = -(d4 + d6)... sau khi xoay pi:
    # khung DH base: p = (a2 + a3, -(d4 + d6), d1 - d5) -> base_link: (-(a2 + a3), d4 + d6, d1 - d5)
    t = kin.fk(np.zeros(6))
    expected = np.array([-(kin.A2 + kin.A3), kin.D4 + kin.D6, kin.D1 - kin.D5])
    np.testing.assert_allclose(t[:3, 3], expected, atol=1e-9)


def test_fk_home_pose_reach_is_reasonable():
    q = np.radians([0, -51.4, 66.5, -67.3, 0, 0])
    p = kin.fk(q)[:3, 3]
    assert 0.2 < np.linalg.norm(p) < 0.9


@pytest.mark.parametrize("seed", range(200))
def test_ik_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-math.pi, math.pi, 6)
    q[2] = rng.uniform(-2.8, 2.8)
    q[4] = rng.choice([-1, 1]) * rng.uniform(0.15, math.pi - 0.15)  # tranh ky di co tay
    tcp = 0.0 if seed % 2 == 0 else 0.15
    pose = kin.fk(q, tcp)
    sols = kin.ik_all(pose, tcp)
    assert sols, "IK khong tra ve nghiem"
    for s in sols:
        assert kin.pose_error(kin.fk(s, tcp), pose) < 1e-6
    # Nghiem goc phai nam trong tap nghiem (sai khac 2pi)
    diffs = [np.max(np.abs(np.angle(np.exp(1j * (s - q))))) for s in sols]
    assert min(diffs) < 1e-6


def test_ik_closest_returns_reference_for_own_pose():
    q_ref = np.radians([10, -60, 80, -100, -90, 30])
    pose = kin.fk(q_ref)
    q = kin.ik_closest(pose, q_ref)
    np.testing.assert_allclose(q, q_ref, atol=1e-6)


def test_ik_small_step_stays_on_branch():
    q_ref = np.radians([10, -60, 80, -100, -90, 30])
    pose = kin.fk(q_ref)
    pose[:3, 3] += [0.01, -0.01, 0.01]
    q = kin.ik_closest(pose, q_ref)
    assert q is not None
    assert np.max(np.abs(q - q_ref)) < 0.1


def test_unreachable_pose_returns_none():
    pose = kin.make_pose([2.0, 0.0, 0.5], [0, 0, 0])
    assert kin.ik_closest(pose, np.zeros(6)) is None


def test_rpy_roundtrip():
    for rpy in [(0.1, -0.4, 2.0), (-2.5, 1.2, -0.3), (math.pi, 0.0, 0.0)]:
        r = kin.rpy_to_matrix(*rpy)
        np.testing.assert_allclose(kin.rpy_to_matrix(*kin.matrix_to_rpy(r)), r, atol=1e-9)


def test_interpolate_pose_endpoints():
    a = kin.make_pose([0.3, 0.1, 0.4], [math.pi, 0, 0])
    b = kin.make_pose([0.4, -0.2, 0.3], [math.pi, 0.3, 1.0])
    np.testing.assert_allclose(kin.interpolate_pose(a, b, 0.0), a, atol=1e-9)
    np.testing.assert_allclose(kin.interpolate_pose(a, b, 1.0), b, atol=1e-9)
    mid = kin.interpolate_pose(a, b, 0.5)
    assert abs(kin.rotation_angle_between(a, mid) - kin.rotation_angle_between(mid, b)) < 1e-9


def test_jacobian_matches_finite_difference():
    q = np.radians([15, -70, 85, -100, -80, 40])
    jac = kin.jacobian(q, 0.1)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        err = kin.twist_error(kin.fk(q, 0.1), kin.fk(q + dq, 0.1)) / eps
        np.testing.assert_allclose(jac[:, i], err, atol=1e-5)


def test_dls_moves_out_of_wrist_singularity():
    # Home cua scene co wrist_2 = 0 -> ky di co tay; IK giai tich se nhay q4/q6
    q = np.radians([0, -51.4, 66.5, -67.3, 0, 0])
    start = kin.fk(q)
    goal = start.copy()
    goal[:3, 3] += [0.0, 0.0, -0.05]
    for s in np.linspace(0, 1, 101)[1:]:
        target = kin.interpolate_pose(start, goal, s)
        q_new, _, _ = kin.ik_dls(target, q)
        assert np.max(np.abs(q_new - q)) < 0.2, "buoc khop qua lon"
        q = q_new
    for _ in range(20):
        q, pos_err, rot_err = kin.ik_dls(goal, q)
    assert pos_err < 1e-4 and rot_err < 1e-3
