"""EuRoC experiment: fixed R vs learned R in the 3D ESKF, real IMU + real visual relocalisation measurements.

Protocol (nested leave-one-out over the five Machine Hall sequences, no leakage):
- Test sequence s: its measurements come from a map built WITHOUT s (results/reloc_loo/excl-s__query-s).
- Training data for s: for each other sequence u, measurements from a map built without s AND u
  (excl-{s,u}__query-u). Everything below is fitted on these only:
  * R models: fixed constant and learned log-linear, Gaussian NLL on all training residuals.
  * Inflation factor alpha (per model): the relocalisation errors are correlated in time (neighbouring views see
    the same map landmarks), which a Kalman filter fed with white-noise R cannot know. Each filter uses
    alpha * r_k, with alpha chosen so that the position NEES on the training sequences is closest to 3.
  * IMU noise scale: oracle synthetic vision on the training sequences, NEES closest to 9.
- Outlier detector ('+flag' variants): logistic regression on the training flights' indicators predicting fixes
  with error > 0.2 m, threshold at 80 % recall on training; flagged fixes are dropped before the filter. Alpha is
  re-tuned for each variant the same way.
- All filters use the same innovation gate (NIS > chi2_3(0.999) rejected) and the same recovery rule (after 1 s
  without any accepted measurement, the next one is accepted regardless of the gate).
- Initialisation: at the end of the first stationary window (>= 3 s, detected with the ground-truth speed as a
  stand-in for a zero-velocity detector): p and q from the ground truth, v = 0, gyro bias = mean gyro over the
  window, accelerometer bias = 0.
- Stress test: test measurements from the strict MH_01+MH_02 map (results/reloc/) for MH_03..MH_05, with the same
  per-fold models (trained on dense leave-one-out maps): a distribution shift (sparser map, more failures).

All maps are of the same room (different flights): easier than a real deployment.
With --degraded, the query images (training and test) are blurred and darkened (results/reloc_loo_degraded,
see run_reloc_loo.py --degrade); outputs go to results/euroc_exp_degraded and there is no stress test.
Usage: python run_euroc_experiment.py [--degraded] [n_workers]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from consistency import frac_below_95, nis_coverage  # noqa: E402
from eskf import ImuNoise, State, run_eskf  # noqa: E402
from euroc import interp_groundtruth, load_euroc  # noqa: E402
from learn_R import LogVarModel  # noqa: E402
from run_failure_prediction import logreg  # noqa: E402
from vis_synth import synth_vision  # noqa: E402

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"]
SHORT = {s: s[:5] for s in SEQS}
MODELS = ("fixed", "learned", "fixed+flag", "learned+flag")
OUTLIER_M = 0.2          # outlier definition for the detector (same as run_failure_prediction.py)
FLAG_RECALL = 0.8        # detector threshold: 80 % recall of outliers on the training flights
DEGRADED = "--degraded" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--degraded"]
LOO = ROOT / "results" / ("reloc_loo_degraded" if DEGRADED else "reloc_loo")
STRICT = None if DEGRADED else ROOT / "results" / "reloc"
OUT = ROOT / "results" / ("euroc_exp_degraded" if DEGRADED else "euroc_exp")
GATE = float(stats.chi2.ppf(0.999, 3))
RECOVER_S = 1.0
WHITE_SCALES = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
ALPHAS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
BURN = 10.0          # s after initialisation, excluded from metrics
LV_CLIP = (-14.0, 6.0)
P0 = np.diag([0.05**2] * 3 + [0.05**2] * 3 + [0.02**2] * 3 + [0.2**2] * 3 + [0.005**2] * 3)


# ---------------------------------------------------------------- data
def prepare(name):
    d = load_euroc(ROOT / "data" / name)
    gt, valid = interp_groundtruth(d.gt, d.imu.t)
    still = np.flatnonzero(valid & (np.linalg.norm(gt["v"], axis=1) < 0.02))
    runs = [r for r in np.split(still, np.flatnonzero(np.diff(still) > 1) + 1) if d.imu.t[r[-1]] - d.imu.t[r[0]] >= 3.0]
    win = runs[0]
    i0, i1 = int(win[-1]), int(np.flatnonzero(valid)[-1])
    init = State(gt["p"][i0].copy(), np.zeros(3), gt["q"][i0].copy(), np.zeros(3), d.imu.gyro[win].mean(0))
    imu_ns = d.t0_ns + np.round(d.imu.t * 1e9).astype(np.int64)
    return dict(name=name, d=d, gt=gt, i0=i0, i1=i1, init=init, imu_ns=imu_ns)


_SEQ_CACHE = {}


def get_seq(name):
    if name not in _SEQ_CACHE:
        _SEQ_CACHE[name] = prepare(name)
    return _SEQ_CACHE[name]


def load_meas(path, seq):
    r = np.load(path)
    idx = np.clip(np.searchsorted(seq["imu_ns"], r["cam_ns"]), 0, len(seq["imu_ns"]) - 1)
    assert np.abs(seq["imu_ns"][idx] - r["cam_ns"]).max() < 1_000_000, "camera frames not on the IMU grid"
    # degradation severity in [0, 1] (0 = clean): the worse of blur and darkening
    sev = np.maximum(r["blur"] / 6.0, (1 - r["gain"]) / 0.92) if "blur" in r else np.zeros(len(r["ok"]))
    return dict(idx=idx, z=r["z"], ok=r["ok"], feats=np.nan_to_num(r["feats"]), err=r["z"] - r["p_gt"],
                names=list(r["feature_names"]), sev=sev)


def loo_path(excluded, query):
    return LOO / f"excl-{'+'.join(SHORT[e] for e in sorted(excluded, key=SEQS.index))}__query-{SHORT[query]}.npz"


# ---------------------------------------------------------------- models (fitted on training residuals only)
def fit_model(model, test):
    train = [u for u in SEQS if u != test]
    tr = [load_meas(loo_path([test, u], u), get_seq(u)) for u in train]
    F = np.concatenate([m["feats"][m["ok"]] for m in tr])
    E = np.concatenate([m["err"][m["ok"]] for m in tr])
    if model.split("+")[0] == "fixed":
        r = float((E**2).sum(1).mean() / 3)
        return lambda f: np.full(len(f), r)
    return LogVarModel(lv_clip=LV_CLIP).fit(F, E).predict_var


def fit_flag(model, test):
    """Outlier detector for '+flag' models: logistic regression on the training flights' fixes (all indicators),
    threshold at FLAG_RECALL recall on training. Returns f(feats) -> bool array (True = drop), or None."""
    if not model.endswith("+flag"):
        return None
    tr = [load_meas(loo_path([test, u], u), get_seq(u)) for u in SEQS if u != test]
    F = np.concatenate([m["feats"][m["ok"]] for m in tr])
    y = np.concatenate([np.linalg.norm(m["err"][m["ok"]], axis=1) > OUTLIER_M for m in tr]).astype(float)
    score = logreg(F, y)
    thr = np.quantile(score(F)[y == 1], 1 - FLAG_RECALL)
    return lambda f: score(f) >= thr


def white_scale(test, onees):
    train = [u for u in SEQS if u != test]
    score = {sc: np.mean([abs(np.log(onees[f"{u}|{sc}"] / 9)) for u in train]) for sc in WHITE_SCALES}
    return min(score, key=score.get)


# ---------------------------------------------------------------- filter runs
def run(seq, meas, r_pred, noise, alpha, flag=None):
    t = seq["d"].imu.t
    window = (meas["idx"] > seq["i0"]) & (meas["idx"] <= seq["i1"])
    drop = flag(meas["feats"]) & meas["ok"] if flag is not None else np.zeros(len(meas["ok"]), bool)
    ok = meas["ok"] & window & ~drop
    vis_idx, z, r = meas["idx"][ok], meas["z"][ok], alpha * r_pred[ok]
    out = run_eskf(seq["d"].imu, seq["i0"], seq["i1"], vis_idx, z, lambda k: r[k], (seq["init"].copy(), P0),
                   noise, truth=seq["gt"], gate=GATE, recover_s=RECOVER_S)
    keep = t[vis_idx[out["used"]]] - t[seq["i0"]] > BURN
    acc = out["accepted"]
    # position error at every 10 Hz frame (including relocalisation failures and rejected measurements)
    frames = meas["idx"][window]
    frames = frames[t[frames] - t[seq["i0"]] > BURN]
    e = np.linalg.norm(out["p"][frames - seq["i0"]] - seq["gt"]["p"][frames], axis=1)
    nees, nees_p = out["nees"][keep[acc]], out["nees_p"][keep[acc]]
    ninl = meas["feats"][ok][out["used"]][keep & acc][:, meas["names"].index("log_inliers")]
    sev = meas["sev"][ok][out["used"]][keep & acc]
    big = np.linalg.norm(meas["err"][ok][out["used"]], axis=1) > OUTLIER_M
    n_out = int((meas["ok"] & window & (np.linalg.norm(np.nan_to_num(meas["err"]), axis=1) > OUTLIER_M)).sum())
    return dict(rmse=float(np.sqrt(np.mean(e**2))), median_err=float(np.median(e)), max_err=float(e.max()),
                nis_mean=float(out["nis"][keep & acc].mean()), nis_cov=nis_coverage(out["nis"][keep & acc], 3),
                nees_p_mean=float(nees_p.mean()), nees_p_below=frac_below_95(nees_p, 3),
                nees_mean=float(nees.mean()), nees_below=frac_below_95(nees, 9),
                accept=float(acc[keep].mean()), forced=float(out["forced"][keep].mean()), n_meas=int(keep.sum()),
                reloc_success=float(meas["ok"][window].mean()), dropped=float(drop[window & meas["ok"]].mean()),
                outliers_total=n_out, outliers_accepted=int((big & acc & keep).sum()),
                _nees_p=nees_p, _ninl=ninl, _sev=sev)


def oracle_nees(args):
    """Mean NEES of the filter with synthetic oracle vision (IMU-noise tuning)."""
    name, scale = args
    seq = get_seq(name)
    d = seq["d"]
    cam_idx = np.searchsorted(d.imu.t, d.cam_t)
    sel = (cam_idx > seq["i0"]) & (cam_idx <= seq["i1"])
    ci = cam_idx[sel]
    vis = synth_vision(d.cam_t[sel], seq["gt"]["p"][ci], 0)
    out = run_eskf(d.imu, seq["i0"], seq["i1"], ci, vis["z"], lambda k: vis["sigma"][k] ** 2,
                   (seq["init"].copy(), P0), ImuNoise(white_scale=scale), truth=seq["gt"])
    keep = d.imu.t[ci[out["used"]]] - d.imu.t[seq["i0"]] > BURN
    return f"{name}|{scale}", float(out["nees"][keep].mean())


def alpha_task(args):
    """Position NEES on a training sequence u (inner map, no s, no u) for model fitted for test fold s."""
    test, u, model, scale = args
    r_fn, flag = fit_model(model, test), fit_flag(model, test)
    m = load_meas(loo_path([test, u], u), get_seq(u))
    res = {}
    for a in ALPHAS:
        res[a] = run(get_seq(u), m, r_fn(m["feats"]), ImuNoise(white_scale=scale), a, flag)["nees_p_mean"]
    return f"{test}|{u}|{model}", res


def test_task(args):
    test, model, scale, alpha = args
    r_fn, flag = fit_model(model, test), fit_flag(model, test)
    seq = get_seq(test)
    out = {}
    m = load_meas(loo_path([test], test), seq)
    out["loo"] = run(seq, m, r_fn(m["feats"]), ImuNoise(white_scale=scale), alpha, flag)
    sp = STRICT / f"{test}.npz" if STRICT else None
    if sp is not None and sp.exists():
        st = load_meas(sp, seq)
        out["stress"] = run(seq, st, r_fn(st["feats"]), ImuNoise(white_scale=scale), alpha, flag)
    return test, model, out


def ci95(x):
    x = np.asarray(x)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def fmt(m):
    return (f"RMSE {m['rmse']:.3f} m, median {m['median_err']:.3f}, max {m['max_err']:.2f} | NIS {m['nis_mean']:.2f} "
            f"cov {m['nis_cov']:.3f} | NEES_p {m['nees_p_mean']:.2f} below {m['nees_p_below']:.3f} | NEES_9 "
            f"{m['nees_mean']:.1f} | accepted {m['accept']:.3f}, forced {m['forced']:.3f} of {m['n_meas']} "
            f"(reloc success {m['reloc_success']:.2f}, dropped by flag {m['dropped']:.3f}, outliers accepted "
            f"{m['outliers_accepted']}/{m['outliers_total']})")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    workers = int(ARGS[0]) if ARGS else 6
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEQS:
        q = get_seq(s)
        print(f"{SHORT[s]}: init at t = {q['d'].imu.t[q['i0']]:.1f} s, gyro bias error at init "
              f"{np.linalg.norm(q['init'].bg - q['gt']['b_w'][q['i0']]):.4f} rad/s")

    with Pool(workers) as pool:
        cache = OUT / "oracle_nees.json"
        onees = json.loads(cache.read_text()) if cache.exists() else {}
        todo = [(s, sc) for s in SEQS for sc in WHITE_SCALES if f"{s}|{sc}" not in onees]
        onees.update(dict(pool.map(oracle_nees, todo)))
        cache.write_text(json.dumps(onees, indent=1))
        scales = {s: white_scale(s, onees) for s in SEQS}

        cache = OUT / "alpha_nees_p.json"
        anees = json.loads(cache.read_text()) if cache.exists() else {}
        todo = [(s, u, mdl, scales[s]) for s in SEQS for u in SEQS if u != s for mdl in MODELS
                if f"{s}|{u}|{mdl}" not in anees]
        for key, res in pool.imap_unordered(alpha_task, todo):
            anees[key] = {str(a): v for a, v in res.items()}
            cache.write_text(json.dumps(anees, indent=1))
        alphas = {}
        for s in SEQS:
            for mdl in MODELS:
                score = {a: np.mean([abs(np.log(anees[f"{s}|{u}|{mdl}"][str(a)] / 3)) for u in SEQS if u != s])
                         for a in ALPHAS}
                alphas[s, mdl] = min(score, key=score.get)

        res = {}
        for test, mdl, out in pool.imap_unordered(test_task, [(s, m, scales[s], alphas[s, m]) for s in SEQS for m in MODELS]):
            res[test, mdl] = out

    for s in SEQS:
        print(f"\nfold test={SHORT[s]}: IMU scale x{scales[s]:g}, alpha " +
              ", ".join(f"{m} x{alphas[s, m]:g}" for m in MODELS))
        print("   training position NEES by alpha (mean over training sequences, ideal 3):")
        for mdl in MODELS:
            row = "  ".join(f"x{a:g}: {np.mean([anees[f'{s}|{u}|{mdl}'][str(a)] for u in SEQS if u != s]):.1f}" for a in ALPHAS)
            print(f"     {mdl:12s} {row}")
        for mdl in MODELS:
            print(f"  {mdl:12s}: {fmt(res[s, mdl]['loo'])}")
        for mdl in MODELS:
            if "stress" in res[s, mdl]:
                print(f"  STRESS {mdl:12s}: {fmt(res[s, mdl]['stress'])}")

    print("\n=== leave-one-out, 5 test sequences: means, and 95% t-intervals of paired differences (n = 5, small) ===")
    summary = {}
    pairs = (("diff", "learned", "fixed"), ("flag_effect_fixed", "fixed+flag", "fixed"),
             ("flag_effect_learned", "learned+flag", "learned"))
    for key in ("rmse", "median_err", "max_err", "nis_cov", "nees_p_mean", "nees_p_below", "nees_mean", "accept",
                "forced", "dropped", "outliers_accepted"):
        v = {m: [res[s, m]["loo"][key] for s in SEQS] for m in MODELS}
        summary[key] = {m: ci95(v[m]) for m in MODELS}
        summary[key].update({name: ci95(np.subtract(v[a], v[b])) for name, a, b in pairs})
        print(f"  {key:17s} " + "  ".join(f"{m} {summary[key][m][0]:7.3f}" for m in MODELS))
        print(" " * 20 + "  ".join(f"{name} {summary[key][name][0]:+.3f} [{summary[key][name][1]:+.3f}, "
                                    f"{summary[key][name][2]:+.3f}]" for name, _, _ in pairs))
    print("\nposition NEES (ideal 3) by number of inliers (pooled terciles over the 5 test sequences):")
    by_inliers, by_severity = {}, {}
    for mdl in MODELS:
        nees = np.concatenate([res[s, mdl]["loo"]["_nees_p"] for s in SEQS])
        ninl = np.concatenate([res[s, mdl]["loo"]["_ninl"] for s in SEQS])
        q1, q2 = np.quantile(ninl, [1 / 3, 2 / 3])
        bins = [ninl <= q1, (ninl > q1) & (ninl <= q2), ninl > q2]
        by_inliers[mdl] = [float(nees[b].mean()) for b in bins]
        print(f"  {mdl:12s}: few {nees[bins[0]].mean():.2f}  mid {nees[bins[1]].mean():.2f}  many {nees[bins[2]].mean():.2f}")

    if DEGRADED:
        print("\nposition NEES (ideal 3) by degradation severity of the accepted measurements (pooled):")
        for mdl in MODELS:
            nees = np.concatenate([res[s, mdl]["loo"]["_nees_p"] for s in SEQS])
            sev = np.concatenate([res[s, mdl]["loo"]["_sev"] for s in SEQS])
            bins = [("clean <0.2", sev < 0.2), ("moderate", (sev >= 0.2) & (sev < 0.5)), ("strong >=0.5", sev >= 0.5)]
            by_severity[mdl] = [float(nees[b].mean()) for _, b in bins]
            print(f"  {mdl:12s}: " + "  ".join(f"{n} {nees[b].mean():.2f} (n={b.sum()})" for n, b in bins))

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    (OUT / "metrics.json").write_text(json.dumps(dict(
        protocol=dict(gate=GATE, recover_s=RECOVER_S, burn_s=BURN, alphas=ALPHAS, white_scales=WHITE_SCALES,
                      outlier_m=OUTLIER_M, flag_recall=FLAG_RECALL),
        folds={s: dict(white_scale=scales[s], alpha={m: alphas[s, m] for m in MODELS}) for s in SEQS},
        summary=summary, nees_p_by_inliers_tercile=by_inliers, nees_p_by_severity=by_severity,
        per_sequence={s: {m: strip(res[s, m]["loo"]) for m in MODELS} for s in SEQS},
        stress={s: {m: strip(res[s, m]["stress"]) for m in MODELS} for s in SEQS if "stress" in res[s, "fixed"]}),
        indent=1))
    print("saved", OUT / "metrics.json")
