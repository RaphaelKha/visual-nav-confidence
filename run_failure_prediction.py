"""Do the quality indicators predict relocalisation failures and outliers? (degraded images, nested leave-one-out)

Two questions, same no-leakage protocol as run_euroc_experiment.py (train on the inner-map measurements of the four
other sequences, test on the outer-map measurements of the held-out sequence):
1. Failure (PnP gives no pose) from indicators available BEFORE the pose: image only (brightness, sharpness), and
   image + number of descriptor matches.
2. Outlier (successful pose with error > 0.2 m) among successes, from all indicators. Compared with what the
   filters' innovation gate actually rejected in run_euroc_experiment.py --degraded.

Model: L2-regularised logistic regression on standardised features. Metric: ROC AUC (0.5 = chance, 1 = perfect),
per test sequence and pooled, plus recall/precision at the threshold giving 80 % recall on the training data.
Usage: python run_failure_prediction.py [--clean]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
CLEAN = "--clean" in sys.argv
LOO = ROOT / "results" / ("reloc_loo" if CLEAN else "reloc_loo_degraded")
OUT = ROOT / "results" / ("failure_prediction_clean.json" if CLEAN else "failure_prediction_degraded.json")
SEQS = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]
OUTLIER_M = 0.2


def path(excluded, query):
    return LOO / f"excl-{'+'.join(sorted(excluded))}__query-{query}.npz"


def load(p):
    r = np.load(p)
    names = list(r["feature_names"])
    return dict(F=np.nan_to_num(r["feats"]), ok=r["ok"], err=r["err"], names=names)


def logreg(X, y, l2=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w_pos = 0.5 / max(y.mean(), 1e-9)          # class balancing (rare positives)
    w_neg = 0.5 / max(1 - y.mean(), 1e-9)
    sw = np.where(y == 1, w_pos, w_neg)

    def f(w):
        s = Z @ w
        p = 1 / (1 + np.exp(-s))
        val = -np.mean(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) + l2 * np.sum(w[1:] ** 2)
        g = Z.T @ (sw * (p - y)) / len(y)
        g[1:] += 2 * l2 * w[1:]
        return val, g

    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w


def auc(score, y):
    """ROC AUC via the Mann-Whitney rank statistic."""
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def task(name, select, target, cols):
    per, pooled_s, pooled_y = {}, [], []
    for s in SEQS:
        tr = [load(path([s, u], u)) for u in SEQS if u != s]
        te = load(path([s], s))
        idx = [tr[0]["names"].index(c) for c in cols]
        Xtr = np.concatenate([m["F"][select(m)][:, idx] for m in tr])
        ytr = np.concatenate([target(m)[select(m)] for m in tr]).astype(float)
        Xte, yte = te["F"][select(te)][:, idx], target(te)[select(te)].astype(float)
        model = logreg(Xtr, ytr)
        thr = np.quantile(model(Xtr)[ytr == 1], 0.2)           # 80 % recall on training
        st = model(Xte)
        flag = st >= thr
        per[s] = dict(auc=auc(st, yte), positives=int(yte.sum()), n=len(yte),
                      recall=float(flag[yte == 1].mean()) if yte.sum() else float("nan"),
                      precision=float(yte[flag].mean()) if flag.any() else float("nan"),
                      flagged=float(flag.mean()))
        pooled_s.append(st)
        pooled_y.append(yte)
    y, sc = np.concatenate(pooled_y), np.concatenate(pooled_s)
    pooled = dict(auc=auc(sc, y), positives=int(y.sum()), n=len(y), base_rate=float(y.mean()))
    print(f"\n{name}  (features: {', '.join(cols)})")
    print(f"  pooled AUC {pooled['auc']:.3f}  ({pooled['positives']} positives / {pooled['n']}, base rate {pooled['base_rate']:.3f})")
    for s in SEQS:
        p = per[s]
        print(f"  {s}: AUC {p['auc']:.3f}  positives {p['positives']:4d}/{p['n']}  at train-80%-recall threshold: "
              f"recall {p['recall']:.2f}, precision {p['precision']:.2f}, flagged {p['flagged']:.2f}")
    return dict(pooled=pooled, per_sequence=per, features=cols)


if __name__ == "__main__":
    allf = lambda m: np.ones(len(m["ok"]), bool)  # noqa: E731
    succ = lambda m: m["ok"]  # noqa: E731
    fail = lambda m: ~m["ok"]  # noqa: E731
    outl = lambda m: np.nan_to_num(m["err"], nan=0.0) > OUTLIER_M  # err is already a norm; noqa: E731
    print(f"data: {LOO.name}")
    res = dict(
        failure_image_only=task("1a. FAILURE from image only", allf, fail, ["brightness", "sharpness"]),
        failure_with_matches=task("1b. FAILURE from image + matches", allf, fail, ["brightness", "sharpness", "log_matches"]),
        outlier_all=task(f"2a. OUTLIER (> {OUTLIER_M} m) among successes, all indicators", succ, outl,
                         ["log_matches", "inlier_ratio", "log_inliers", "reproj_rms", "coverage", "brightness", "sharpness"]),
        outlier_inliers_only=task(f"2b. OUTLIER (> {OUTLIER_M} m), number of inliers only", succ, outl, ["log_inliers"]),
    )
    OUT.write_text(json.dumps(res, indent=1))
    print("\nsaved", OUT)
