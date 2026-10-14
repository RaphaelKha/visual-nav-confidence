"""Visual relocalisation against a prior map: real cam0 image -> absolute 3D body position + quality indicators.

Map: built from reference sequences (stereo cam0/cam1 SIFT matches, triangulated with the calibrated extrinsics,
placed in the world frame with the ground-truth pose of each keyframe). Query: SIFT on a single cam0 image,
nearest-neighbour matching against all map descriptors (no position prior), PnP + RANSAC, LM refinement.

This is the same principle as map-based visual navigation (match the current view against a geo-referenced
reference), at indoor scale. The EuRoC Machine Hall sequences share one world frame (checked visually), so a
map from MH_01/MH_02 can localise MH_03..MH_05.
"""
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eskf import quat_to_R
from euroc import interp_groundtruth


# ---------------------------------------------------------------- calibration and images
@dataclass
class Camera:
    K: np.ndarray        # 3x3
    dist: np.ndarray     # radial-tangential (k1, k2, p1, p2)
    T_BC: np.ndarray     # 4x4, camera -> body (IMU)
    size: tuple          # (width, height)


def _floats(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def load_camera(yaml_path):
    """Minimal parser for the EuRoC camera sensor.yaml (no pyyaml dependency)."""
    s = Path(yaml_path).read_text()
    T = np.array(_floats(re.search(r"data:\s*\[([^\]]*)\]", s).group(1))).reshape(4, 4)
    fu, fv, cu, cv = _floats(re.search(r"intrinsics:\s*\[([^\]]*)\]", s).group(1))
    dist = np.array(_floats(re.search(r"distortion_coefficients:\s*\[([^\]]*)\]", s).group(1)))
    w, h = (int(x) for x in _floats(re.search(r"resolution:\s*\[([^\]]*)\]", s).group(1)))
    return Camera(np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]]), dist, T, (w, h))


class ImageSource:
    """Reads EuRoC PNG frames directly from the sequence zip (images are not unzipped)."""

    def __init__(self, zip_path):
        self.z = zipfile.ZipFile(zip_path)

    def read(self, cam, ts_ns):
        buf = np.frombuffer(self.z.read(f"mav0/{cam}/data/{int(ts_ns)}.png"), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def body_pose(p, q):
    """4x4 body -> world from ground-truth position and (w,x,y,z) quaternion."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_R(q), p
    return T


# ---------------------------------------------------------------- map building
@dataclass
class Map:
    X: np.ndarray        # (N,3) world points
    desc: np.ndarray     # (N,128) float32 SIFT descriptors (from cam0)
    kf: np.ndarray       # (N,) keyframe id of each point


def stereo_landmarks(img0, img1, cam0, cam1, sift, epi_px=2.0, zmin=0.3, zmax=20.0):
    """SIFT stereo matches triangulated in the cam0 frame. Returns (X_c0 (n,3), desc (n,128))."""
    k0, d0 = sift.detectAndCompute(img0, None)
    k1, d1 = sift.detectAndCompute(img1, None)
    if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
        return np.zeros((0, 3)), np.zeros((0, 128), np.float32)
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2) if m.distance < 0.8 * n.distance]
    if len(good) < 10:
        return np.zeros((0, 3)), np.zeros((0, 128), np.float32)
    u0 = np.float64([k0[m.queryIdx].pt for m in good])
    u1 = np.float64([k1[m.trainIdx].pt for m in good])
    x0 = cv2.undistortPoints(u0[:, None], cam0.K, cam0.dist)[:, 0]
    x1 = cv2.undistortPoints(u1[:, None], cam1.K, cam1.dist)[:, 0]
    T10 = np.linalg.inv(cam1.T_BC) @ cam0.T_BC                  # cam0 -> cam1
    R, t = T10[:3, :3], T10[:3, 3]
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    h0, h1 = np.c_[x0, np.ones(len(x0))], np.c_[x1, np.ones(len(x1))]
    lines = h0 @ E.T                                            # epipolar lines in cam1
    epi = np.abs(np.sum(h1 * lines, 1)) / np.linalg.norm(lines[:, :2], axis=1)
    ok = epi < epi_px / cam1.K[0, 0]
    Xh = cv2.triangulatePoints(np.eye(3, 4), T10[:3], x0[ok].T, x1[ok].T)
    X = (Xh[:3] / Xh[3]).T
    z1 = (X @ R.T + t)[:, 2]
    keep = (X[:, 2] > zmin) & (X[:, 2] < zmax) & (z1 > zmin)
    desc = np.float32([d0[m.queryIdx] for m, o in zip(good, ok) if o])
    return X[keep], desc[keep]


def build_map(sequences, cam0, cam1, n_features=1500, min_move=0.25, min_rot_deg=10.0, log=print):
    """sequences: list of (zip_path, GroundTruth, cam_ns, cam_t). Keyframes are taken whenever the body moved
    more than ``min_move`` m or rotated more than ``min_rot_deg`` since the previous keyframe."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    Xs, Ds, Ks = [], [], []
    kf_id = 0
    for zip_path, gt, cam_ns, cam_t in sequences:
        src = ImageSource(zip_path)
        g, valid = interp_groundtruth(gt, cam_t)
        last = None
        for i in np.flatnonzero(valid):
            T_WB = body_pose(g["p"][i], g["q"][i])
            if last is not None:
                dR = last[:3, :3].T @ T_WB[:3, :3]
                ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
                if np.linalg.norm(T_WB[:3, 3] - last[:3, 3]) < min_move and ang < min_rot_deg:
                    continue
            last = T_WB
            X, d = stereo_landmarks(src.read("cam0", cam_ns[i]), src.read("cam1", cam_ns[i]), cam0, cam1, sift)
            T_WC = T_WB @ cam0.T_BC
            Xs.append(X @ T_WC[:3, :3].T + T_WC[:3, 3])
            Ds.append(d)
            Ks.append(np.full(len(X), kf_id))
            kf_id += 1
        log(f"  map: {Path(zip_path).stem}: {kf_id} keyframes so far, {sum(map(len, Xs))} points")
    return Map(np.concatenate(Xs).astype(np.float32), np.concatenate(Ds), np.concatenate(Ks))


# ---------------------------------------------------------------- localisation
FEATURES = ("log_matches", "inlier_ratio", "log_inliers", "reproj_rms", "coverage", "brightness", "sharpness")


class Relocalizer:
    def __init__(self, m, cam0, n_features=1000, ratio=0.8, same_point_m=0.15, min_inliers=12):
        self.m, self.cam = m, cam0
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.flann.add([m.desc])
        self.flann.train()
        self.ratio, self.same_point_m, self.min_inliers = ratio, same_point_m, min_inliers

    def locate(self, img):
        """Returns dict: ok, p (body position in world, NaN if failed), and the quality indicators."""
        h, w = img.shape
        out = dict(ok=False, p=np.full(3, np.nan), brightness=img.mean() / 255.0,
                   sharpness=np.log(cv2.Laplacian(img, cv2.CV_64F).var() + 1.0),
                   n_kp=0, n_matches=0, n_inliers=0, reproj_rms=np.nan, coverage=0.0)
        kp, d = self.sift.detectAndCompute(img, None)
        out["n_kp"] = len(kp)
        if d is None or len(kp) < self.min_inliers:
            return out
        X, uv = [], []
        for m, n in self.flann.knnMatch(d, k=2):
            # ratio test, except when the two nearest neighbours are the same physical point seen from two keyframes
            if m.distance < self.ratio * n.distance or \
                    np.linalg.norm(self.m.X[m.trainIdx] - self.m.X[n.trainIdx]) < self.same_point_m:
                X.append(self.m.X[m.trainIdx])
                uv.append(kp[m.queryIdx].pt)
        out["n_matches"] = len(X)
        if len(X) < self.min_inliers:
            return out
        X, uv = np.float64(X), np.float64(uv)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, self.cam.K, self.cam.dist, iterationsCount=1000,
                                                 reprojectionError=3.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.min_inliers:
            return out
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], uv[inl], self.cam.K, self.cam.dist, rvec, tvec)
        proj, _ = cv2.projectPoints(X[inl], rvec, tvec, self.cam.K, self.cam.dist)
        R, _ = cv2.Rodrigues(rvec)
        T_WC = np.eye(4)
        T_WC[:3, :3], T_WC[:3, 3] = R.T, -R.T @ tvec[:, 0]
        T_WB = T_WC @ np.linalg.inv(self.cam.T_BC)
        cells = {(int(4 * u / w), int(4 * v / h)) for u, v in uv[inl]}
        out.update(ok=True, p=T_WB[:3, 3], n_inliers=len(inl),
                   reproj_rms=float(np.sqrt(np.mean(np.sum((proj[:, 0] - uv[inl]) ** 2, 1)))),
                   coverage=len(cells) / 16.0)
        return out


def feature_vector(r):
    """Indicator vector (see FEATURES) from a locate() result."""
    nm, ni = r["n_matches"], r["n_inliers"]
    return np.array([np.log1p(nm), ni / nm if nm else 0.0, np.log1p(ni),
                     r["reproj_rms"] if r["ok"] else np.nan, r["coverage"], r["brightness"], r["sharpness"]])


def localize_sequence(rl, zip_path, gt, cam_ns, cam_t, step=2, log=None, transform=None):
    """Localise every ``step``-th cam0 frame with valid ground truth. Returns a dict of arrays.
    transform(img, n) -> img: optional per-frame image transformation (n = index among the localised frames)."""
    g, valid = interp_groundtruth(gt, cam_t)
    idx = np.flatnonzero(valid)[::step]
    src = ImageSource(zip_path)
    z, feats, ok = [], [], []
    for n, i in enumerate(idx):
        img = src.read("cam0", cam_ns[i])
        r = rl.locate(img if transform is None else transform(img, n))
        z.append(r["p"]), feats.append(feature_vector(r)), ok.append(r["ok"])
        if log and n % 500 == 0:
            log(f"{n}/{len(idx)}")
    z = np.array(z)
    return dict(cam_ns=cam_ns[idx], cam_t=cam_t[idx], z=z, ok=np.array(ok), feats=np.array(feats),
                p_gt=g["p"][idx], err=np.linalg.norm(z - g["p"][idx], axis=1), feature_names=np.array(FEATURES))
