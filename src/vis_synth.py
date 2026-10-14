"""Provisional synthetic "visual" position for real sequences (until real image-based measurements, step 3).

Ground-truth position + Gaussian noise whose std depends on a latent scene quality q (same model as the 2D
simulator: OU latent, sigma(q) = 0.5 + 10 (1-q)^2 m). Observable indicators are noisy functions of q.
The IMU data are real; the visual noise is simulated.
"""
import numpy as np

from sim import vision_sigma


def synth_vision(t, p, seed):
    """t: (K,) camera times (s), p: (K,3) true positions. Returns dict(z, q, sigma, feats)."""
    rng = np.random.default_rng(seed)
    K = len(t)
    dtv = float(np.median(np.diff(t)))
    tau_q, s_z = 10.0, 1.0
    zq = np.zeros(K)
    prev = rng.normal(0, s_z)
    for i in range(K):
        prev = prev - prev / tau_q * dtv + s_z * np.sqrt(2 / tau_q * dtv) * rng.normal()
        zq[i] = prev
    q = 1.0 / (1.0 + np.exp(-(1.5 * zq + 0.5)))
    sigma = vision_sigma(q)
    z = p + sigma[:, None] * rng.normal(size=(K, 3))

    n_matches = rng.poisson(10 + 190 * q)
    score = np.clip(0.3 + 0.6 * q + rng.normal(0, 0.08, K), 0, 1)
    texture = np.clip(q**0.7 + rng.normal(0, 0.12, K), 0, 1)
    bright = np.clip(0.3 + 0.4 * q + rng.normal(0, 0.15, K), 0, 1)
    feats = np.column_stack([np.log1p(n_matches), score, texture, bright])
    return dict(z=z, q=q, sigma=sigma, feats=feats)
