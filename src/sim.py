"""2D simulator: trajectory, IMU (with bias) and a visual position whose quality varies.

The scene quality q(t) in [0, 1] is latent (1 = well-textured urban area, 0 = desert / night / blur).
The visual measurement noise depends on q; the filter only sees noisy indicators of q
(number of matches, matching score, texture, brightness).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Sim:
    dt: float
    t: np.ndarray            # (N,)
    p: np.ndarray            # (N,2) true position
    v: np.ndarray            # (N,2) true velocity
    b: np.ndarray            # (N,2) true accelerometer bias
    u: np.ndarray            # (N,2) acceleration measured by the IMU
    vis_idx: np.ndarray      # (K,) indices (into N) of the visual measurements
    z: np.ndarray            # (K,2) position measured by vision
    q: np.ndarray            # (K,) latent scene quality
    sigma: np.ndarray        # (K,) true standard deviation of the visual noise
    feats: np.ndarray        # (K,4) observable indicators
    sigma_a: float
    sigma_b: float


def vision_sigma(q):
    """Standard deviation (m) of the visual error as a function of scene quality."""
    return 0.5 + 10.0 * (1.0 - q) ** 2


def generate(seed, T=200.0, dt=0.01, vis_every=10, sigma_a=0.05, sigma_b=0.005):
    rng = np.random.default_rng(seed)
    N = int(T / dt)

    p = np.zeros((N, 2))
    v = np.zeros((N, 2))
    b = np.zeros((N, 2))
    a = np.zeros((N, 2))
    p[0] = rng.normal(0, 10, 2)
    v[0] = rng.normal(0, 2, 2)
    b[0] = rng.normal(0, 0.05, 2)

    tau, s_ou = 5.0, 0.8                     # commanded acceleration ~ OU process, stationary std 0.8 m/s^2
    ou = rng.normal(0, s_ou, 2)
    for k in range(N):
        ou = ou - ou / tau * dt + s_ou * np.sqrt(2 / tau * dt) * rng.normal(size=2)
        a[k] = ou - 0.15 * v[k]              # light damping: velocity stays bounded
        if k < N - 1:
            p[k + 1] = p[k] + v[k] * dt + 0.5 * a[k] * dt**2
            v[k + 1] = v[k] + a[k] * dt
            b[k + 1] = b[k] + sigma_b * np.sqrt(dt) * rng.normal(size=2)

    u = a + b + rng.normal(0, sigma_a, (N, 2))

    # --- vision at 10 Hz ---
    vis_idx = np.arange(vis_every, N, vis_every)
    K = len(vis_idx)
    dtv = dt * vis_every
    tau_q, s_z = 10.0, 1.0
    zq = np.zeros(K)
    zq0 = rng.normal(0, s_z)
    prev = zq0
    for i in range(K):
        prev = prev - prev / tau_q * dtv + s_z * np.sqrt(2 / tau_q * dtv) * rng.normal()
        zq[i] = prev
    q = 1.0 / (1.0 + np.exp(-(1.5 * zq + 0.5)))

    sigma = vision_sigma(q)
    z = p[vis_idx] + sigma[:, None] * rng.normal(size=(K, 2))

    # observable indicators, noisy (brightness is a weak / almost-distractor indicator)
    n_matches = rng.poisson(10 + 190 * q)
    score = np.clip(0.3 + 0.6 * q + rng.normal(0, 0.08, K), 0, 1)
    texture = np.clip(q**0.7 + rng.normal(0, 0.12, K), 0, 1)
    bright = np.clip(0.3 + 0.4 * q + rng.normal(0, 0.15, K), 0, 1)
    feats = np.column_stack([np.log1p(n_matches), score, texture, bright])

    return Sim(dt, np.arange(N) * dt, p, v, b, u, vis_idx, z, q, sigma, feats, sigma_a, sigma_b)
