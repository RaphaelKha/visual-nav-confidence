"""Leave-one-out relocalisation measurements for all five Machine Hall sequences, with nested folds.

Outer fold (test sequence s): map built from the 4 other sequences, query s.
Inner folds (for training the learned R without leakage): for each training sequence u != s, its measurements
come from a map built WITHOUT s and u (the 3 remaining sequences). The map excluding {a, b} serves both
(query a, outer b) and (query b, outer a): 10 three-sequence maps + 5 four-sequence maps, 25 query runs.

All maps are of the same room (Machine Hall) from different flights: easier than a real deployment where the
reference map (e.g. satellite imagery) differs much more from the live view. The strict MH_01+MH_02 map results
(run_reloc.py, results/reloc/) are kept as a stress test.

With --degrade, the query images (never the map) are blurred and darkened with a slowly varying, per-sequence
schedule (src/degrade.py; seed = sequence index, so a sequence has the same degradations in every fold).

Output: results/reloc_loo[_degraded]/excl-<A>[+<B>]__query-<Q>.npz
Usage: python run_reloc_loo.py [--degrade] [path/to/machine_hall] [n_workers]
"""
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from euroc import load_groundtruth  # noqa: E402
from degrade import degradation_schedule, degrade  # noqa: E402
from euroc import interp_groundtruth  # noqa: E402
from reloc import Relocalizer, build_map, load_camera, localize_sequence  # noqa: E402

DEGRADE = "--degrade" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--degrade"]
ZIPS = Path(ARGS[0] if len(ARGS) > 0 else Path.home() / "Downloads" / "machine_hall")
WORKERS = int(ARGS[1]) if len(ARGS) > 1 else 4
DATA = Path("data")
OUT = Path("results") / ("reloc_loo_degraded" if DEGRADE else "reloc_loo")
SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
SHORT = {s: s[:5] for s in SEQS}


def seq(name):
    _, cam_ns, _ = load_groundtruth(DATA / name, 0)
    gt, cam_ns, cam_t = load_groundtruth(DATA / name, int(cam_ns[0]))
    return ZIPS / name / f"{name}.zip", gt, cam_ns, cam_t


def out_path(excluded, query):
    return OUT / f"excl-{'+'.join(SHORT[e] for e in excluded)}__query-{SHORT[query]}.npz"


def job(excluded):
    todo = [q for q in excluded if not out_path(excluded, q).exists()]
    if not todo:
        return f"{excluded}: cached"
    cam0 = load_camera(DATA / "MH_01_easy/mav0/cam0/sensor.yaml")
    cam1 = load_camera(DATA / "MH_01_easy/mav0/cam1/sensor.yaml")
    t = time.time()
    m = build_map([seq(s) for s in SEQS if s not in excluded], cam0, cam1, log=lambda *_: None)
    rl = Relocalizer(m, cam0)
    msg = []
    for q in todo:
        zip_path, gt, cam_ns, cam_t = seq(q)
        extra, transform = {}, None
        if DEGRADE:
            _, valid = interp_groundtruth(gt, cam_t)
            t_loc = cam_t[np.flatnonzero(valid)[::2]]             # frames localised by localize_sequence (step 2)
            blur, gain = degradation_schedule(t_loc, seed=SEQS.index(q))
            rng = np.random.default_rng(100 + SEQS.index(q))
            transform = lambda img, n: degrade(img, blur[n], gain[n], rng)  # noqa: E731
            extra = dict(blur=blur, gain=gain)
        r = localize_sequence(rl, zip_path, gt, cam_ns, cam_t, transform=transform)
        np.savez(out_path(excluded, q), **r, **extra)
        e = r["err"][r["ok"]]
        msg.append(f"query {SHORT[q]}: success {r['ok'].mean():.2f}, median {np.median(e):.3f} m, p99 {np.percentile(e, 99):.2f} m")
    return f"excl {[SHORT[e] for e in excluded]} ({len(m.X)} pts, {time.time() - t:.0f} s): " + "; ".join(msg)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s,) for s in SEQS] + list(itertools.combinations(SEQS, 2))
    with Pool(WORKERS) as pool:
        for line in pool.imap_unordered(job, jobs):
            print(line, flush=True)
