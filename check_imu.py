"""IMU / ground-truth frame checks and IMU-only drift decomposition on the stationary segment.

1. Gravity: at rest, R(q_gt) (a_m - b_a) must equal -g = (0, 0, +9.81) in the world frame.
2. Gyro at rest: w_m must equal the ground-truth gyro bias.
3. Frame / T_BS: the angular velocity derived from the ground-truth quaternions must match w_m - b_g in the IMU
   frame. A best-fit rotation between the two (Kabsch) must be ~identity, otherwise the ground-truth attitude is
   not the IMU frame. Also checks the time offset (cross-correlation lag).
4. IMU-only propagation while stationary, with each bias set to its ground-truth value or to zero, compared with
   the analytic drift: 1/2 |b_a| t^2 (accel bias) and g |b_g,tilt| t^3 / 6 (gyro bias tilting gravity).

Usage: python check_imu.py [data/MH_01_easy]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from eskf import GRAVITY, State, nominal_step, quat_conj, quat_log, quat_mul, quat_to_R  # noqa: E402
from euroc import interp_groundtruth, load_euroc  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/MH_01_easy")
d = load_euroc(root)
gt, valid = interp_groundtruth(d.gt, d.imu.t)
still = np.flatnonzero(valid & (np.linalg.norm(gt["v"], axis=1) < 0.02))
run = max(np.split(still, np.flatnonzero(np.diff(still) > 1) + 1), key=len)
i0, i1 = run[0], run[-1]
s = slice(i0, i1 + 1)
print(f"stationary segment t = {d.imu.t[i0]:.1f}..{d.imu.t[i1]:.1f} s; GT speed max {np.linalg.norm(gt['v'][s], axis=1).max():.4f} m/s, "
      f"GT position spread {np.ptp(gt['p'][s], axis=0).round(4)} m")

# 1. gravity
f_w = np.array([quat_to_R(gt["q"][i]) @ (d.imu.accel[i] - gt["b_a"][i]) for i in range(i0, i1 + 1)])
print("\n1. gravity at rest: mean R (a_m - b_a) =", f_w.mean(0).round(4), "m/s^2 (expect [0, 0, 9.81]); "
      f"|mean| = {np.linalg.norm(f_w.mean(0)):.4f}, residual vs -GRAVITY = {np.linalg.norm(f_w.mean(0) + GRAVITY):.4f}")
print("   raw accel mean in body frame:", d.imu.accel[s].mean(0).round(3), "-> body x points up")

# 2. gyro at rest
print("\n2. gyro at rest: mean w_m =", d.imu.gyro[s].mean(0).round(5), " GT b_g =", gt["b_w"][s].mean(0).round(5), "rad/s")

# 3. frame and time offset from the angular velocity over the whole flight
v = np.flatnonzero(valid)
v = v[(v > v[0]) & (v < v[-1])]
dt = d.imu.t[v + 1] - d.imu.t[v - 1]
w_gt = np.array([quat_log(quat_mul(quat_conj(gt["q"][i - 1]), gt["q"][i + 1])) for i in v]) / dt[:, None]
w_imu = d.imu.gyro[v] - gt["b_w"][v]
H = w_imu.T @ w_gt
U, _, Vt = np.linalg.svd(H)
Rfit = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T      # w_gt ~ Rfit w_imu
ang = np.degrees(np.arccos(np.clip((np.trace(Rfit) - 1) / 2, -1, 1)))
res = w_gt - w_imu
print(f"\n3. angular velocity GT vs gyro (whole flight, {len(v)} samples): rms |w| = {np.sqrt((w_gt**2).sum(1).mean()):.3f} rad/s, "
      f"rms residual = {np.sqrt((res**2).sum(1).mean()):.4f} rad/s")
print(f"   best-fit rotation between the two frames: {ang:.2f} deg (expect ~0 if GT attitude = IMU frame, T_BS = I)")
lags = np.arange(-20, 21)
a, b = np.linalg.norm(w_gt, axis=1), np.linalg.norm(d.imu.gyro[v] - gt["b_w"][v], axis=1)
cc = [np.corrcoef(a[20:-20], b[20 + k:len(b) - 20 + k])[0, 1] for k in lags]
print(f"   time offset (max cross-correlation of |w|): {lags[int(np.argmax(cc))] * 5} ms")

# 4. drift decomposition
print("\n4. IMU-only propagation while stationary (initial p, v, q from GT):")
tt = d.imu.t[s] - d.imu.t[i0]
ba0, bg0 = gt["b_a"][i0], gt["b_w"][i0]
up_b = quat_to_R(gt["q"][i0]).T @ np.array([0, 0, 1.0])               # world up in the body frame
bg_tilt = np.linalg.norm(bg0 - (bg0 @ up_b) * up_b)                   # gyro bias component that tilts gravity
for label, ba, bg in (("both biases = GT", ba0, bg0), ("b_a = 0, b_g = GT", 0 * ba0, bg0),
                      ("b_a = GT, b_g = 0", ba0, 0 * bg0), ("both = 0", 0 * ba0, 0 * bg0)):
    st = State(gt["p"][i0].copy(), gt["v"][i0].copy(), gt["q"][i0].copy(), ba.copy(), bg.copy())
    err = [0.0]
    for i in range(i0, i1):
        st = nominal_step(st, d.imu.accel[i], d.imu.gyro[i], d.imu.t[i + 1] - d.imu.t[i])
        err.append(np.linalg.norm(st.p - gt["p"][i + 1]))
    err = np.array(err)
    row = "  ".join(f"t+{T:>2}s {err[np.searchsorted(tt, T)]:7.3f} m" for T in (1, 5, 10))
    print(f"   {label:18s} {row}")
print(f"   GT biases: |b_a| = {np.linalg.norm(ba0):.3f} m/s^2, b_g = {bg0.round(4)} rad/s, tilting part |b_g,tilt| = {bg_tilt:.4f} rad/s")
for T in (1, 5, 10):
    print(f"   analytic t+{T:>2}s: accel bias 1/2|b_a|t^2 = {0.5 * np.linalg.norm(ba0) * T**2:6.3f} m, "
          f"gyro bias g|b_g,tilt|t^3/6 = {9.81 * bg_tilt * T**3 / 6:7.3f} m")
