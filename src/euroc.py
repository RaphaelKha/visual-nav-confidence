"""EuRoC MAV loader (ASL format): IMU, ground truth and camera timestamps on a common time axis.

Expected layout (as unzipped from e.g. MH_01_easy.zip): ``<root>/mav0/{imu0,state_groundtruth_estimate0,cam0}/data.csv``.
Only the CSV files are read here, images are not loaded.

Time is returned in seconds relative to the first IMU sample (computed on integer nanoseconds, so no
float precision is lost on the ~1.4e18 ns epoch timestamps). Quaternions are Hamilton, order (w, x, y, z),
as in the dataset. The ground truth is expressed in the world frame R, the IMU in the body frame S.

Dataset: M. Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 35(10), 2016
(ETH ASL, CC BY-NC-SA 4.0 -- check the dataset page).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class Imu:
    t: np.ndarray        # (N,) s
    gyro: np.ndarray     # (N,3) rad/s, body frame
    accel: np.ndarray    # (N,3) m/s^2, body frame (specific force, includes gravity)


@dataclass
class GroundTruth:
    t: np.ndarray        # (M,) s
    p: np.ndarray        # (M,3) m, world frame
    q: np.ndarray        # (M,4) (w, x, y, z)
    v: np.ndarray        # (M,3) m/s, world frame
    b_w: np.ndarray      # (M,3) gyro bias
    b_a: np.ndarray      # (M,3) accelerometer bias


@dataclass
class EuRoC:
    imu: Imu
    gt: GroundTruth
    cam_t: np.ndarray    # (C,) s, cam0 frame timestamps
    t0_ns: int           # timestamp (ns) of t = 0


def _read_csv(path, ncols):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if arr.shape[1] < ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def _read_ns(path):
    """First column (integer ns) read exactly, as int64."""
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)


def load_euroc(root):
    """Load IMU, ground truth and camera timestamps from an EuRoC sequence directory."""
    mav = Path(root) / "mav0"
    imu_ns = _read_ns(mav / "imu0" / "data.csv")
    gt_ns = _read_ns(mav / "state_groundtruth_estimate0" / "data.csv")
    cam_ns = np.loadtxt(mav / "cam0" / "data.csv", delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
    t0 = int(imu_ns[0])

    def sec(ns):
        return (ns - t0).astype(np.float64) * 1e-9

    a = _read_csv(mav / "imu0" / "data.csv", 7)
    g = _read_csv(mav / "state_groundtruth_estimate0" / "data.csv", 17)
    imu = Imu(sec(imu_ns), a[:, 1:4], a[:, 4:7])
    gt = GroundTruth(sec(gt_ns), g[:, 1:4], g[:, 4:8], g[:, 8:11], g[:, 11:14], g[:, 14:17])
    for name, t in (("imu", imu.t), ("groundtruth", gt.t), ("cam0", sec(cam_ns))):
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{name} timestamps are not strictly increasing")
    return EuRoC(imu, gt, sec(cam_ns), t0)


def load_groundtruth(root, t0_ns):
    """Ground truth and cam0 timestamps only (no IMU needed), times in s relative to ``t0_ns``.
    Returns (GroundTruth, cam_ns int64 array, cam_t float array)."""
    mav = Path(root) / "mav0"
    gt_ns = _read_ns(mav / "state_groundtruth_estimate0" / "data.csv")
    cam_ns = _read_ns(mav / "cam0" / "data.csv")
    g = _read_csv(mav / "state_groundtruth_estimate0" / "data.csv", 17)
    sec = lambda ns: (ns - t0_ns).astype(np.float64) * 1e-9  # noqa: E731
    gt = GroundTruth(sec(gt_ns), g[:, 1:4], g[:, 4:8], g[:, 8:11], g[:, 11:14], g[:, 14:17])
    return gt, cam_ns, sec(cam_ns)


def interp_groundtruth(gt, t):
    """Ground truth at times ``t``: linear for p, v, biases, slerp for q.

    Returns (dict of arrays, valid mask). Times outside the ground-truth span are marked invalid
    (their values are clamped to the nearest end and must not be used).
    """
    t = np.asarray(t, dtype=np.float64)
    valid = (t >= gt.t[0]) & (t <= gt.t[-1])
    tc = np.clip(t, gt.t[0], gt.t[-1])

    def lin(x):
        return np.stack([np.interp(tc, gt.t, x[:, i]) for i in range(x.shape[1])], axis=1)

    rot = Rotation.from_quat(gt.q[:, [1, 2, 3, 0]])  # scipy wants (x, y, z, w)
    q_xyzw = Slerp(gt.t, rot)(tc).as_quat()
    out = {
        "p": lin(gt.p),
        "v": lin(gt.v),
        "q": q_xyzw[:, [3, 0, 1, 2]],
        "b_w": lin(gt.b_w),
        "b_a": lin(gt.b_a),
    }
    return out, valid
