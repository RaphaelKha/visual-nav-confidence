"""3D error-state Kalman filter (ESKF): IMU propagation + 3D position updates, R supplied by the caller.

Conventions follow Sola, "Quaternion kinematics for the error-state Kalman filter" (2017):
- Nominal state: position p, velocity v (world frame), attitude q (Hamilton, (w,x,y,z), body -> world),
  accelerometer bias b_a, gyro bias b_g. Gravity g is known and constant in the world frame.
- Error state (15): dx = [dp, dv, dtheta, db_a, db_g] with  true = nominal (+) dx, i.e.
  p_t = p + dp,  v_t = v + dv,  q_t = q * Exp(dtheta)  (LOCAL attitude error, in the body frame),  b_t = b + db.
- IMU sample k is assumed to represent the interval [t_k, t_k + dt] (constant over the interval). Nominal
  propagation uses the mid-interval rotation R * Exp(w dt / 2). The error-state transition matrix is the exact
  Jacobian of this discrete propagation (checked by finite differences in tests/test_eskf.py).
"""
import warnings
from dataclasses import dataclass

import numpy as np

# numpy 2.0.x + Apple Accelerate raises spurious FP RuntimeWarnings on every matmul (finite inputs included);
# run_eskf checks that the result is finite instead.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")

GRAVITY = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------- SO(3) / quaternion helpers
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(phi):
    """Rotation matrix of the rotation vector phi (Rodrigues)."""
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-6:
        return np.eye(3) + (1 - th**2 / 6) * K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def right_jacobian(phi):
    """Jr(phi), such that Exp(phi + d) ~ Exp(phi) Exp(Jr(phi) d)."""
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-4:
        A, B = 0.5 - th**2 / 24, 1 / 6 - th**2 / 120
    else:
        A, B = (1 - np.cos(th)) / th**2, (th - np.sin(th)) / th**3
    return np.eye(3) - A * K + B * K @ K


def quat_exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-8:
        return np.array([1.0, *(0.5 * phi)]) / np.sqrt(1 + 0.25 * th**2)
    return np.array([np.cos(th / 2), *(np.sin(th / 2) * phi / th)])


def quat_log(q):
    """Rotation vector of a unit quaternion (shortest rotation)."""
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return 2 * q[1:]
    return 2 * np.arctan2(s, q[0]) * q[1:] / s


def quat_mul(a, b):
    aw, av, bw, bv = a[0], a[1:], b[0], b[1:]
    return np.array([aw * bw - av @ bv, *(aw * bv + bw * av + np.cross(av, bv))])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- state and noise
@dataclass
class State:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    ba: np.ndarray
    bg: np.ndarray

    def copy(self):
        return State(self.p.copy(), self.v.copy(), self.q.copy(), self.ba.copy(), self.bg.copy())

    def boxplus(self, dx):
        """true = nominal (+) error."""
        q = quat_mul(self.q, quat_exp(dx[6:9]))
        return State(self.p + dx[0:3], self.v + dx[3:6], q / np.linalg.norm(q), self.ba + dx[9:12], self.bg + dx[12:15])

    def boxminus(self, other):
        """error = self (-) other, i.e. the dx such that other (+) dx = self."""
        dth = quat_log(quat_mul(quat_conj(other.q), self.q))
        return np.concatenate([self.p - other.p, self.v - other.v, dth, self.ba - other.ba, self.bg - other.bg])


@dataclass
class ImuNoise:
    """Continuous noise densities (ADIS16448 values from EuRoC imu0/sensor.yaml) and tuning scales."""
    acc_nd: float = 2.0e-3        # m/s^2/sqrt(Hz)
    acc_rw: float = 3.0e-3        # m/s^3/sqrt(Hz)
    gyr_nd: float = 1.6968e-4     # rad/s/sqrt(Hz)
    gyr_rw: float = 1.9393e-5     # rad/s^2/sqrt(Hz)
    white_scale: float = 1.0      # multiplies the white-noise densities (std)
    rw_scale: float = 1.0         # multiplies the random-walk densities (std)


# ---------------------------------------------------------------- propagation
def nominal_step(s, a_m, w_m, dt, g=GRAVITY):
    a, w = a_m - s.ba, w_m - s.bg
    E = exp_so3(0.5 * w * dt)
    aw = quat_to_R(s.q) @ E @ a + g
    q = quat_mul(s.q, quat_exp(w * dt))
    return State(s.p + s.v * dt + 0.5 * aw * dt**2, s.v + aw * dt, q / np.linalg.norm(q), s.ba.copy(), s.bg.copy())


def error_jacobians(s, a_m, w_m, dt):
    """Exact Jacobians of nominal_step: Fd (15x15, wrt the error state) and Gn (15x6, wrt [n_a, n_g]
    = derivative wrt the measured [a_m, w_m])."""
    a, w = a_m - s.ba, w_m - s.bg
    E = exp_so3(0.5 * w * dt)
    Rn = quat_to_R(s.q)
    Rm = Rn @ E
    Jr = right_jacobian(w * dt)
    Ea_x = skew(E @ a)
    a_x = skew(a)
    I, Z = np.eye(3), np.zeros((3, 3))

    F_vth = -Rn @ Ea_x * dt
    F_vba = -Rm * dt
    F_vbg = Rm @ a_x @ right_jacobian(0.5 * w * dt) * (0.5 * dt**2)
    Fd = np.block([
        [I, I * dt, 0.5 * dt * F_vth, 0.5 * dt * F_vba, 0.5 * dt * F_vbg],
        [Z, I, F_vth, F_vba, F_vbg],
        [Z, Z, exp_so3(-w * dt), Z, -Jr * dt],
        [Z, Z, Z, I, Z],
        [Z, Z, Z, Z, I],
    ])
    C = -Rm @ a_x @ right_jacobian(0.5 * w * dt) * (0.5 * dt)    # d(world accel) / d(w_m)
    Gn = np.zeros((15, 6))
    Gn[0:3, 0:3] = 0.5 * dt**2 * Rm
    Gn[3:6, 0:3] = dt * Rm
    Gn[0:3, 3:6] = 0.5 * dt**2 * C
    Gn[3:6, 3:6] = dt * C
    Gn[6:9, 3:6] = Jr * dt
    return Fd, Gn


# ---------------------------------------------------------------- filter
class ESKF:
    def __init__(self, state, P, noise=None, g=GRAVITY):
        self.s, self.P, self.noise, self.g = state, P.copy(), noise or ImuNoise(), g

    def predict(self, a_m, w_m, dt):
        n = self.noise
        Fd, Gn = error_jacobians(self.s, a_m, w_m, dt)
        # discrete white-noise std on a sample of duration dt: density / sqrt(dt)
        Wd = np.diag(np.repeat([n.white_scale * n.acc_nd, n.white_scale * n.gyr_nd], 3) ** 2 / dt)
        Q = Gn @ Wd @ Gn.T
        Q[9:12, 9:12] += np.eye(3) * (n.rw_scale * n.acc_rw) ** 2 * dt
        Q[12:15, 12:15] += np.eye(3) * (n.rw_scale * n.gyr_rw) ** 2 * dt
        self.s = nominal_step(self.s, a_m, w_m, dt, self.g)
        self.P = Fd @ self.P @ Fd.T + Q

    def update_position(self, z, r, gate=None):
        """Position measurement z (3,), isotropic variance r (m^2) per axis. Returns (innovation, S, accepted).
        If ``gate`` is given, the update is skipped when the NIS exceeds it (innovation gating)."""
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        nu = z - self.s.p
        S = H @ self.P @ H.T + r * np.eye(3)
        if gate is not None and nu @ np.linalg.solve(S, nu) > gate:
            return nu, S, False
        K = np.linalg.solve(S, H @ self.P).T                  # P H^T S^-1
        IKH = np.eye(15) - K @ H
        P = IKH @ self.P @ IKH.T + r * K @ K.T                # Joseph form
        dx = K @ nu
        self.s = self.s.boxplus(dx)
        G = np.eye(15)                                        # reset: error expressed in the new nominal frame
        G[6:9, 6:9] -= skew(0.5 * dx[6:9])
        self.P = G @ P @ G.T
        return nu, S, True


def run_eskf(imu, i0, i1, vis_idx, z, R_fn, init, noise=None, truth=None, gate=None, recover_s=None):
    """Run the filter on IMU samples i0..i1 (inclusive), with position updates at IMU indices ``vis_idx``.

    init: (State, P0) valid at IMU index i0. R_fn(k) -> variance (m^2 per axis) of measurement k.
    truth: optional dict with 'p', 'v', 'q' arrays at IMU times (for NEES, post-update, dof = 9: p, v, theta).
    gate: optional NIS threshold; measurements above it are rejected (NIS is still recorded, NEES is not).
    recover_s: if set, a measurement is accepted regardless of the gate once no measurement has been accepted for
        this long (prevents the divergence spiral of an overconfident filter that rejects everything).
    Returns dict with trajectories at every IMU index in [i0, i1] and NIS/NEES at each used measurement.
    """
    state, P0 = init
    f = ESKF(state, P0, noise)
    n = i1 - i0 + 1
    out = {k: np.zeros((n, d)) for k, d in (("p", 3), ("v", 3), ("q", 4), ("ba", 3), ("bg", 3))}
    out["sig_p"] = np.zeros((n, 3))
    vis_pos = {int(i): k for k, i in enumerate(vis_idx)}
    used, nis, nees, nees_p, R_used, accepted, forced = [], [], [], [], [], [], []
    t_last = imu.t[i0]

    def record(i):
        j = i - i0
        s = f.s
        out["p"][j], out["v"][j], out["q"][j], out["ba"][j], out["bg"][j] = s.p, s.v, s.q, s.ba, s.bg
        out["sig_p"][j] = np.sqrt(np.diag(f.P)[:3])

    record(i0)
    for i in range(i0 + 1, i1 + 1):
        f.predict(imu.accel[i - 1], imu.gyro[i - 1], imu.t[i] - imu.t[i - 1])
        if i in vis_pos:
            k = vis_pos[i]
            r = R_fn(k)
            force = recover_s is not None and imu.t[i] - t_last > recover_s
            nu, S, acc = f.update_position(z[k], r, None if force else gate)
            if acc:
                t_last = imu.t[i]
            used.append(k)
            R_used.append(r)
            accepted.append(acc)
            forced.append(force)
            nis.append(nu @ np.linalg.solve(S, nu))
            if truth is not None and acc:
                e = np.concatenate([truth["p"][i] - f.s.p, truth["v"][i] - f.s.v,
                                    quat_log(quat_mul(quat_conj(f.s.q), truth["q"][i]))])
                idx = np.r_[0:9]
                nees.append(e @ np.linalg.solve(f.P[np.ix_(idx, idx)], e))
                nees_p.append(e[:3] @ np.linalg.solve(f.P[:3, :3], e[:3]))
        record(i)
    if not (np.isfinite(f.P).all() and np.isfinite(out["p"]).all()):
        raise FloatingPointError("ESKF diverged (non-finite state or covariance)")
    out.update(used=np.array(used, dtype=int), nis=np.array(nis), nees=np.array(nees), nees_p=np.array(nees_p),
               R=np.array(R_used), accepted=np.array(accepted, dtype=bool), forced=np.array(forced, dtype=bool))
    return out
