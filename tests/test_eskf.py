"""ESKF checks 1 and 2: noise-free propagation against an analytic trajectory, and finite-difference Jacobians."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eskf import (GRAVITY, State, error_jacobians, exp_so3, nominal_step, quat_conj, quat_exp,  # noqa: E402
                  quat_log, quat_mul, quat_to_R)

W = np.array([0.3, -0.2, 0.5])        # constant body angular rate (rad/s)
A = np.array([1.0, -0.5, 0.3])        # world acceleration a_w(t) = A sin(OM t)
OM = 1.3
Q0 = quat_exp(np.array([0.4, -0.3, 0.8]))
P0, V0 = np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.2, 0.1])
BA, BG = np.array([0.1, -0.05, 0.08]), np.array([0.01, 0.02, -0.01])


def propagate_analytic(dt, T):
    """Integrate with IMU samples = mid-interval specific force (what the filter assumes a sample means)."""
    s = State(P0.copy(), V0.copy(), Q0.copy(), BA.copy(), BG.copy())
    for k in range(int(round(T / dt))):
        tm = (k + 0.5) * dt
        Rm = quat_to_R(quat_mul(Q0, quat_exp(W * tm)))
        f = Rm.T @ (A * np.sin(OM * tm) - GRAVITY)
        s = nominal_step(s, f + BA, W + BG, dt)
    return s


def truth(T):
    p = P0 + V0 * T + A * (T / OM - np.sin(OM * T) / OM**2)
    v = V0 + A * (1 - np.cos(OM * T)) / OM
    return p, v, quat_mul(Q0, quat_exp(W * T))


def errors(dt, T=5.0):
    s = propagate_analytic(dt, T)
    p, v, q = truth(T)
    return (np.linalg.norm(s.p - p), np.linalg.norm(s.v - v),
            np.linalg.norm(quat_log(quat_mul(quat_conj(s.q), q))))


def test_propagation_matches_analytic_trajectory():
    ep, ev, eth = errors(0.005)
    assert ep < 1e-4 and ev < 1e-4, (ep, ev)
    assert eth < 1e-10, eth                      # constant body rate: attitude integration is exact


def test_propagation_error_is_second_order_in_dt():
    e1, e2 = errors(0.01), errors(0.005)
    for a, b in zip(e1[:2], e2[:2]):
        assert 3.5 < a / b < 4.5, (a, b)         # halving dt divides the error by ~4


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(0)
    s = State(rng.normal(size=3), rng.normal(size=3), quat_exp(rng.normal(size=3)),
              0.1 * rng.normal(size=3), 0.02 * rng.normal(size=3))
    a_m, w_m, dt, eps = np.array([0.3, -1.2, 9.5]), np.array([0.4, -0.7, 0.9]), 0.005, 1e-6
    Fd, Gn = error_jacobians(s, a_m, w_m, dt)
    s1 = nominal_step(s, a_m, w_m, dt)
    Fn = np.zeros((15, 15))
    for j in range(15):
        d = np.zeros(15)
        d[j] = eps
        Fn[:, j] = (nominal_step(s.boxplus(d), a_m, w_m, dt).boxminus(s1)
                    - nominal_step(s.boxplus(-d), a_m, w_m, dt).boxminus(s1)) / (2 * eps)
    assert np.abs(Fd - Fn).max() < 1e-7, np.abs(Fd - Fn).max()
    Gnum = np.zeros((15, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        Gnum[:, j] = (nominal_step(s, a_m + d[:3], w_m + d[3:], dt).boxminus(s1)
                      - nominal_step(s, a_m - d[:3], w_m - d[3:], dt).boxminus(s1)) / (2 * eps)
    assert np.abs(Gn - Gnum).max() < 1e-7, np.abs(Gn - Gnum).max()


def test_rotation_helpers_are_consistent():
    phi = np.array([0.3, -1.1, 0.7])
    assert np.allclose(quat_to_R(quat_exp(phi)), exp_so3(phi), atol=1e-12)
    assert np.allclose(quat_log(quat_exp(phi)), phi, atol=1e-12)


# ---------------------------------------------------------------- checks 3 and 4 on real data (skipped without it)
import pytest  # noqa: E402

from consistency import nis_coverage  # noqa: E402
from eskf import ImuNoise, run_eskf  # noqa: E402
from euroc import interp_groundtruth, load_euroc  # noqa: E402
from vis_synth import synth_vision  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "MH_01_easy"
needs_data = pytest.mark.skipif(not (DATA / "mav0").exists(), reason="EuRoC MH_01_easy not in data/")


@pytest.fixture(scope="module")
def euroc():
    d = load_euroc(DATA)
    gt, valid = interp_groundtruth(d.gt, d.imu.t)
    return d, gt, valid


def _state(gt, i, bias=False):
    z = np.zeros(3)
    return State(gt["p"][i].copy(), gt["v"][i].copy(), gt["q"][i].copy(),
                 gt["b_a"][i].copy() if bias else z.copy(), gt["b_w"][i].copy() if bias else z.copy())


@needs_data
def test_imu_only_drift_within_predicted_sigma(euroc):
    """Stationary segment, zero initial biases: the actual drift must be explained by the filter's covariance."""
    d, gt, valid = euroc
    still = np.flatnonzero(valid & (np.linalg.norm(gt["v"], axis=1) < 0.02))
    run = max(np.split(still, np.flatnonzero(np.diff(still) > 1) + 1), key=len)
    i0 = run[0]
    i1 = min(run[-1], i0 + 1000)                   # first 5 s
    P0 = np.diag([0.01**2] * 9 + [0.2**2] * 3 + [0.05**2] * 3)
    r = run_eskf(d.imu, i0, i1, np.array([], dtype=int), np.zeros((0, 3)), None, (_state(gt, i0), P0))
    err = np.linalg.norm(r["p"][-1] - gt["p"][i1])
    assert err < 2 * np.sqrt(3) * r["sig_p"][-1, 0], (err, r["sig_p"][-1])      # drift is real and covered
    assert err > 1.0                                                           # ... and large without vision


@needs_data
def test_oracle_filter_is_consistent_on_real_imu(euroc):
    d, gt, valid = euroc
    j0, j1 = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])
    ci = np.searchsorted(d.imu.t, d.cam_t)
    sel = (ci > j0) & (ci <= j1)
    ci, ct = ci[sel], d.cam_t[sel]
    P0 = np.diag([0.1**2] * 6 + [0.05**2] * 3 + [0.2**2] * 3 + [0.05**2] * 3)
    nis, nees = [], []
    for seed in range(3):
        vis = synth_vision(ct, gt["p"][ci], seed)
        r = run_eskf(d.imu, j0, j1, ci, vis["z"], lambda k: vis["sigma"][k] ** 2, (_state(gt, j0), P0),
                     ImuNoise(white_scale=10.0), truth=gt)
        keep = ct[r["used"]] - d.imu.t[j0] > 10.0
        nis.append(r["nis"][keep])
        nees.append(r["nees"][keep])
    nis, nees = np.concatenate(nis), np.concatenate(nees)
    assert 2.8 < nis.mean() < 3.2, nis.mean()          # ideal 3
    assert 0.93 < nis_coverage(nis, 3) < 0.97
    assert 6.0 < nees.mean() < 11.0, nees.mean()       # ideal 9


@needs_data
def test_gravity_and_imu_frame(euroc):
    """At rest R(q_gt)(a_m - b_a) = -g, and GT angular velocity matches the gyro without any frame rotation."""
    d, gt, valid = euroc
    still = np.flatnonzero(valid & (np.linalg.norm(gt["v"], axis=1) < 0.02))
    run = max(np.split(still, np.flatnonzero(np.diff(still) > 1) + 1), key=len)
    f_w = np.mean([quat_to_R(gt["q"][i]) @ (d.imu.accel[i] - gt["b_a"][i]) for i in run], axis=0)
    assert np.linalg.norm(f_w + GRAVITY) < 0.05, f_w
    v = np.flatnonzero(valid)[1:-1]
    w_gt = np.array([quat_log(quat_mul(quat_conj(gt["q"][i - 1]), gt["q"][i + 1])) for i in v]) \
        / (d.imu.t[v + 1] - d.imu.t[v - 1])[:, None]
    w_imu = d.imu.gyro[v] - gt["b_w"][v]
    U, _, Vt = np.linalg.svd(w_imu.T @ w_gt)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    assert np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) < 0.5
