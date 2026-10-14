"""Sanity check of the EuRoC loader on a real sequence: durations, rates, alignment, trajectory plot.

Usage: python check_euroc.py [data/MH_01_easy]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from euroc import interp_groundtruth, load_euroc  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/MH_01_easy")
d = load_euroc(root)
rate = lambda t: 1 / np.median(np.diff(t))  # noqa: E731

print(f"IMU : {len(d.imu.t)} samples, {d.imu.t[-1]:.1f} s, {rate(d.imu.t):.1f} Hz")
print(f"GT  : {len(d.gt.t)} samples, [{d.gt.t[0]:.2f}, {d.gt.t[-1]:.2f}] s, {rate(d.gt.t):.1f} Hz")
print(f"cam0: {len(d.cam_t)} frames, [{d.cam_t[0]:.2f}, {d.cam_t[-1]:.2f}] s, {rate(d.cam_t):.1f} Hz")
print(f"max IMU gap {np.diff(d.imu.t).max() * 1e3:.1f} ms, max GT gap {np.diff(d.gt.t).max() * 1e3:.1f} ms")

gt, valid = interp_groundtruth(d.gt, d.cam_t)
print(f"cam frames inside GT span: {valid.sum()}/{len(valid)}")
print(f"mean |accel| (IMU) {np.linalg.norm(d.imu.accel, axis=1).mean():.2f} m/s^2 (expect ~9.8)")

# Consistency: integrating the ground-truth velocity must give the ground-truth position.
dp = np.diff(d.gt.p, axis=0)
dpv = 0.5 * (d.gt.v[1:] + d.gt.v[:-1]) * np.diff(d.gt.t)[:, None]
print(f"GT self-consistency, ||sum(dp) - sum(v dt)|| = {np.linalg.norm(dp.sum(0) - dpv.sum(0)):.3f} m "
      f"(path length {np.linalg.norm(dp, axis=1).sum():.1f} m)")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(d.gt.p[:, 0], d.gt.p[:, 1])
ax[0].set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Ground truth, top view")
ax[1].plot(d.gt.t, np.linalg.norm(d.gt.v, axis=1))
ax[1].set(xlabel="t (s)", ylabel="|v| (m/s)", title="Ground-truth speed")
fig.tight_layout()
out = Path("results") / "euroc_check.png"
fig.savefig(out, dpi=120)
print("saved", out)
