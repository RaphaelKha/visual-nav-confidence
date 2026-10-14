"""ESKF checks 3 and 4 on a real EuRoC sequence (checks 1-2 are in tests/test_eskf.py).

3. IMU-only drift on the stationary segment: actual position error vs the filter's own predicted sigma.
4. Oracle filter (true visual variance, synthetic vision) consistency: NIS ~ 3, NEES ~ 9.

Usage: python check_eskf.py [data/MH_01_easy] [white_scale] [rw_scale]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from consistency import frac_below_95, nis_coverage  # noqa: E402
from eskf import ImuNoise, State, run_eskf  # noqa: E402
from euroc import interp_groundtruth, load_euroc  # noqa: E402
from vis_synth import synth_vision  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/MH_01_easy")
noise = ImuNoise(white_scale=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0,
                 rw_scale=float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
d = load_euroc(root)
gt, valid = interp_groundtruth(d.gt, d.imu.t)


def init_state(i, with_bias):
    ba, bg = (gt["b_a"][i], gt["b_w"][i]) if with_bias else (np.zeros(3), np.zeros(3))
    return State(gt["p"][i].copy(), gt["v"][i].copy(), gt["q"][i].copy(), ba.copy(), bg.copy())


# ---------------------------------------------------------------- check 3: IMU-only drift while stationary
speed = np.linalg.norm(gt["v"], axis=1)
still = np.flatnonzero(valid & (speed < 0.02))
runs = np.split(still, np.flatnonzero(np.diff(still) > 1) + 1)
run = max(runs, key=len)                       # longest run of stillness
i0, i1 = run[0], run[-1]
print(f"stationary segment: t = {d.imu.t[i0]:.1f}..{d.imu.t[i1]:.1f} s ({d.imu.t[i1] - d.imu.t[i0]:.1f} s)")
no_meas = np.array([], dtype=int)
for with_bias in (True, False):
    P0 = np.diag([0.01**2] * 3 + [0.01**2] * 3 + [0.01**2] * 3 + [0.2**2] * 3 + [0.05**2] * 3)
    r = run_eskf(d.imu, i0, i1, no_meas, np.zeros((0, 3)), None, (init_state(i0, with_bias), P0), noise)
    t = d.imu.t[i0:i1 + 1] - d.imu.t[i0]
    err = np.linalg.norm(r["p"] - gt["p"][i0:i1 + 1], axis=1)
    print(f"  IMU only, init biases {'= GT' if with_bias else '= 0 (sigma_ba 0.2, sigma_bg 0.05)'}:")
    for T in (1, 5, 10, min(20, t[-1])):
        j = np.searchsorted(t, T)
        print(f"    t+{T:4.1f}s: |position error| = {err[j]:.3f} m, filter sigma_p (1 axis) = {r['sig_p'][j, 0]:.3f} m")

# ---------------------------------------------------------------- check 4: oracle consistency
BURN = 10.0   # s
j0 = int(np.flatnonzero(valid)[0])
j1 = int(np.flatnonzero(valid)[-1])
cam_idx = np.searchsorted(d.imu.t, d.cam_t)
sel = (cam_idx > j0) & (cam_idx <= j1)
cam_idx, cam_t = cam_idx[sel], d.cam_t[sel]
P0 = np.diag([0.1**2] * 3 + [0.1**2] * 3 + [0.05**2] * 3 + [0.2**2] * 3 + [0.05**2] * 3)
nis_all, nees_all, q_all, rmse = [], [], [], []
for seed in range(10):
    vis = synth_vision(cam_t, gt["p"][cam_idx], seed)
    r = run_eskf(d.imu, j0, j1, cam_idx, vis["z"], lambda k: vis["sigma"][k] ** 2,
                 (init_state(j0, False), P0), noise, truth=gt)
    keep = cam_t[r["used"]] - d.imu.t[j0] > BURN
    nis_all.append(r["nis"][keep]); nees_all.append(r["nees"][keep]); q_all.append(vis["q"][r["used"]][keep])
    e = r["p"][cam_idx - j0] - gt["p"][cam_idx]
    rmse.append(np.sqrt((np.linalg.norm(e, axis=1)[cam_t > d.imu.t[j0] + BURN] ** 2).mean()))
nis, nees, q = map(np.concatenate, (nis_all, nees_all, q_all))
print(f"\noracle filter, {len(rmse)} noise seeds, {len(nis)} updates after {BURN:.0f} s burn-in, noise scales "
      f"white={noise.white_scale}, rw={noise.rw_scale}")
print(f"  position RMSE {np.mean(rmse):.3f} m")
print(f"  NIS  mean {nis.mean():.3f} (ideal 3), inside 95% interval {nis_coverage(nis, 3):.3f} (ideal 0.95)")
print(f"  NEES mean {nees.mean():.3f} (ideal 9), below chi2_9 95% {frac_below_95(nees, 9):.3f} (ideal 0.95)")
for name, m in (("poor q<0.4", q < 0.4), ("medium", (q >= 0.4) & (q <= 0.7)), ("good q>0.7", q > 0.7)):
    print(f"  NEES {name:11s}: {nees[m].mean():.2f}  (n={m.sum()})")
