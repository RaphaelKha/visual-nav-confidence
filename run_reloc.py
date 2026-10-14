"""Build the visual map from MH_01+MH_02 and relocalise every 2nd cam0 frame (10 Hz) of MH_03..MH_05.

Images are read from the original EuRoC zips (path below); ground truth and calibration from data/<seq>/mav0.
Outputs: data/cache/map_MH01_MH02.npz (map cache), results/reloc/<seq>.npz (measurements, indicators, errors).

Usage: python run_reloc.py [path/to/machine_hall]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from euroc import interp_groundtruth, load_groundtruth  # noqa: E402
from reloc import FEATURES, ImageSource, Map, Relocalizer, build_map, feature_vector, load_camera  # noqa: E402

ZIPS = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Downloads" / "machine_hall")
DATA = Path("data")
MAP_SEQS = ["MH_01_easy", "MH_02_easy"]
QUERY_SEQS = ["MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
STEP = 2   # every 2nd frame -> 10 Hz

cam0 = load_camera(DATA / "MH_01_easy/mav0/cam0/sensor.yaml")
cam1 = load_camera(DATA / "MH_01_easy/mav0/cam1/sensor.yaml")


def seq(name):
    gt, cam_ns, _ = load_groundtruth(DATA / name, 0)
    t0 = int(cam_ns[0])
    gt, cam_ns, cam_t = load_groundtruth(DATA / name, t0)
    return ZIPS / name / f"{name}.zip", gt, cam_ns, cam_t


cache = DATA / "cache" / "map_MH01_MH02.npz"
if cache.exists():
    c = np.load(cache)
    m = Map(c["X"], c["desc"], c["kf"])
    print(f"map loaded from cache: {len(m.X)} points, {m.kf.max() + 1} keyframes")
else:
    t = time.time()
    m = build_map([seq(s) for s in MAP_SEQS], cam0, cam1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=m.X, desc=m.desc, kf=m.kf)
    print(f"map built in {time.time() - t:.0f} s: {len(m.X)} points")

rl = Relocalizer(m, cam0)
out_dir = Path("results") / "reloc"
out_dir.mkdir(parents=True, exist_ok=True)
for name in QUERY_SEQS:
    zip_path, gt, cam_ns, cam_t = seq(name)
    g, valid = interp_groundtruth(gt, cam_t)
    idx = np.flatnonzero(valid)[::STEP]
    src = ImageSource(zip_path)
    t = time.time()
    z, feats, ok, nkp = [], [], [], []
    for n, i in enumerate(idx):
        r = rl.locate(src.read("cam0", cam_ns[i]))
        z.append(r["p"]), feats.append(feature_vector(r)), ok.append(r["ok"]), nkp.append(r["n_kp"])
        if n % 200 == 0:
            print(f"  {name}: {n}/{len(idx)} ({time.time() - t:.0f} s)", flush=True)
    z, ok = np.array(z), np.array(ok)
    err = np.linalg.norm(z - g["p"][idx], axis=1)
    np.savez(out_dir / f"{name}.npz", cam_ns=cam_ns[idx], cam_t=cam_t[idx], z=z, ok=ok, feats=np.array(feats),
             n_kp=np.array(nkp), p_gt=g["p"][idx], err=err, feature_names=np.array(FEATURES))
    e = err[ok]
    print(f"{name}: {len(idx)} frames, success {ok.mean():.2f}, error median {np.median(e):.3f} m, "
          f"p90 {np.percentile(e, 90):.3f} m, p99 {np.percentile(e, 99):.2f} m, max {e.max():.1f} m "
          f"({time.time() - t:.0f} s)", flush=True)
