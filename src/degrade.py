"""Synthetic image degradations for the query images: blur and darkening (with sensor noise).

The reference map stays built from clean images (like a reference map captured in good conditions); only the
live view degrades. Severity varies slowly in time: two independent latent OU processes (blur, darkness), so the
quality indicators have to tell them apart.
"""
import cv2
import numpy as np

READ_NOISE_DN = 2.0     # additive Gaussian sensor noise (8-bit digital numbers), applied after the gain


def ou(t, tau, rng):
    """Stationary OU process (unit variance) sampled at times t."""
    z = np.zeros(len(t))
    z[0] = rng.normal()
    for i in range(1, len(t)):
        a = np.exp(-(t[i] - t[i - 1]) / tau)
        z[i] = a * z[i - 1] + np.sqrt(1 - a * a) * rng.normal()
    return z


def degradation_schedule(t, seed, tau=10.0, blur_max=6.0, gain_min=0.08):
    """Per-frame blur sigma (px) and brightness gain in (gain_min, 1]. Roughly half the frames are mildly or
    not degraded, the rest moderately to strongly."""
    rng = np.random.default_rng(seed)
    sb = 1 / (1 + np.exp(-(1.5 * ou(t, tau, rng) - 0.5)))
    sd = 1 / (1 + np.exp(-(1.5 * ou(t, tau, rng) - 0.5)))
    blur = blur_max * sb**2
    gain = 1 - (1 - gain_min) * sd**2
    return blur, gain


def degrade(img, blur, gain, rng):
    """Gaussian blur (sigma in px), then brightness gain, then read noise, quantised to uint8."""
    x = img.astype(np.float32)
    if blur > 0.3:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    x = x * gain + rng.normal(0, READ_NOISE_DN, x.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)
