"""IMU + visual position Kalman filter (2D), with R supplied by a function.

State: x = [p(2), v(2), b(2)]  (position, velocity, accelerometer bias)
Prediction: at 100 Hz with the IMU acceleration (corrected by the estimated bias).
Update: at 10 Hz with the visual position, of variance R_k (isotropic scalar).

This is a linear Kalman filter (2D, no attitude). The next step is a 3D ESKF.
"""
import numpy as np


def run_filter(sim, R_fn):
    """R_fn(k) -> variance (m^2) of visual measurement k. Returns a dict of trajectories and statistics."""
    dt = sim.dt
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    F = np.block([[I2, dt * I2, -0.5 * dt**2 * I2],
                  [Z2, I2, -dt * I2],
                  [Z2, Z2, I2]])
    Bu = np.vstack([0.5 * dt**2 * I2, dt * I2, Z2])          # input: measured acceleration
    G = np.vstack([0.5 * dt**2 * I2, dt * I2, Z2])           # accelerometer noise -> state
    Q = sim.sigma_a**2 * (G @ G.T)
    Q[4:, 4:] += sim.sigma_b**2 * dt * I2                    # bias random walk
    H = np.hstack([I2, Z2, Z2])

    N = len(sim.t)
    x = np.concatenate([sim.p[0] + np.random.default_rng(0).normal(0, 1, 2),
                        sim.v[0] + np.random.default_rng(1).normal(0, 0.5, 2),
                        np.zeros(2)])
    P = np.diag([1.0, 1.0, 0.25, 0.25, 0.05**2, 0.05**2])

    x_all = np.zeros((N, 6))
    sig_p = np.zeros((N, 2))
    vis_pos = {int(i): k for k, i in enumerate(sim.vis_idx)}
    K = len(sim.vis_idx)
    nis = np.zeros(K)
    nees = np.zeros(K)
    R_used = np.zeros(K)

    for i in range(N):
        if i in vis_pos:
            k = vis_pos[i]
            R = R_fn(k)
            R_used[k] = R
            S = H @ P @ H.T + R * I2
            nu = sim.z[k] - H @ x
            Kg = P @ H.T @ np.linalg.inv(S)
            x = x + Kg @ nu
            IKH = np.eye(6) - Kg @ H
            P = IKH @ P @ IKH.T + R * (Kg @ Kg.T)           # Joseph form
            nis[k] = nu @ np.linalg.solve(S, nu)
            e = x[:4] - np.concatenate([sim.p[i], sim.v[i]])
            nees[k] = e @ np.linalg.solve(P[:4, :4], e)
        x_all[i] = x
        sig_p[i] = np.sqrt(np.diag(P)[:2])
        x = F @ x + Bu @ sim.u[i]
        P = F @ P @ F.T + Q

    return dict(x=x_all, sig_p=sig_p, nis=nis, nees=nees, R=R_used,
                err_p=x_all[:, :2] - sim.p)
