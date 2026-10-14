"""Experiment: fixed R vs learned R vs oracle R, on simulated flights never seen during training."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from consistency import frac_below_95, nis_coverage  # noqa: E402
from kf import run_filter  # noqa: E402
from learn_R import LogVarModel  # noqa: E402
from sim import generate  # noqa: E402

TRAIN_SEEDS = range(0, 12)
TEST_SEEDS = range(100, 120)
BURN = 100                      # first 10 s ignored in the metrics (warm-up)
BINS = [("poor (q<0.4)", 0.0, 0.4), ("medium (0.4-0.7)", 0.4, 0.7), ("good (q>0.7)", 0.7, 1.01)]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ---------- training ----------
train = [generate(s) for s in TRAIN_SEEDS]
F_tr = np.concatenate([s.feats for s in train])
E_tr = np.concatenate([s.z - s.p[s.vis_idx] for s in train])
model = LogVarModel().fit(F_tr, E_tr)
R_fixed = float((E_tr**2).sum(1).mean() / 2)           # best constant variance (same NLL criterion)
print(f"Fixed R (learned on train) = {R_fixed:.2f} m^2  (sigma = {np.sqrt(R_fixed):.2f} m)")

# ---------- test ----------
names = ["fixed", "learned", "oracle"]
res = {n: dict(nis=[], nees=[], q=[], err_vis=[], err_full=[]) for n in names}
example = None
for s in TEST_SEEDS:
    sim = generate(s)
    Rl = model.predict_var(sim.feats)
    fns = {"fixed": lambda k: R_fixed,
           "learned": lambda k, Rl=Rl: Rl[k],
           "oracle": lambda k, sim=sim: sim.sigma[k] ** 2}
    runs = {n: run_filter(sim, fns[n]) for n in names}
    for n in names:
        r = runs[n]
        res[n]["nis"].append(r["nis"][BURN:])
        res[n]["nees"].append(r["nees"][BURN:])
        res[n]["q"].append(sim.q[BURN:])
        res[n]["err_vis"].append(r["err_p"][sim.vis_idx][BURN:])
        res[n]["err_full"].append(r["err_p"][BURN * 10:])
    if example is None:
        example = (sim, runs)
for n in names:
    for k in res[n]:
        res[n][k] = np.concatenate(res[n][k])

# ---------- metrics ----------
summary = {}
for n in names:
    r = res[n]
    e = np.linalg.norm(r["err_vis"], axis=1)
    row = dict(
        rmse_pos=float(np.sqrt(np.mean(e**2))),
        rmse_pos_poor=float(np.sqrt(np.mean(e[r["q"] < 0.4] ** 2))),
        p95_err=float(np.percentile(e, 95)),
        mean_nis=float(r["nis"].mean()),
        mean_nees=float(r["nees"].mean()),
        nis_cov95=nis_coverage(r["nis"]),
        nees_below95=frac_below_95(r["nees"], 4),
    )
    for label, lo, hi in BINS:
        m = (r["q"] >= lo) & (r["q"] < hi)
        row[f"nees[{label}]"] = float(r["nees"][m].mean())
        row[f"n[{label}]"] = int(m.sum())
    summary[n] = row
json.dump(summary, open(OUT / "metrics.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'':8}{'RMSE':>7}{'RMSE poor':>11}{'p95':>7}{'NIS':>7}{'NEES':>7}{'NIS in95':>10}{'NEES<q95':>10}")
for n in names:
    m = summary[n]
    print(f"{n:8}{m['rmse_pos']:7.2f}{m['rmse_pos_poor']:11.2f}{m['p95_err']:7.2f}"
          f"{m['mean_nis']:7.2f}{m['mean_nees']:7.2f}{m['nis_cov95']:10.2f}{m['nees_below95']:10.2f}")
print("(ideal: NIS=2, NEES=4, NIS in95=0.95, NEES<q95=0.95)")
print("\nMean NEES by scene quality (ideal 4):")
for label, _, _ in BINS:
    print(f"  {label:20}" + "".join(f"{n}={summary[n][f'nees[{label}]']:6.1f}  " for n in names)
          + f"(n={summary['fixed'][f'n[{label}]']})")

# ---------- figures ----------
COL = {"fixed": "#c0392b", "learned": "#1f77b4", "oracle": "#7f8c8d"}

# 1) x error over time, with the +-2 sigma band announced by the filter
sim, runs = example
fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 2, 2]))
tv = sim.t[sim.vis_idx]
ax[0].plot(tv, sim.q, color="k", lw=1)
ax[0].set_ylabel("scene quality q\n(latent)")
ax[0].set_ylim(0, 1)
for a, n, title in [(ax[1], "fixed", "Fixed R: same confidence everywhere"),
                    (ax[2], "learned", "Learned R: confidence predicted at every measurement")]:
    r = runs[n]
    a.fill_between(sim.t, -2 * r["sig_p"][:, 0], 2 * r["sig_p"][:, 0], color=COL[n], alpha=0.25,
                   label="± 2σ announced by the filter")
    a.plot(sim.t, r["err_p"][:, 0], color=COL[n], lw=0.8, label="actual error (x axis)")
    a.set_ylabel("x error (m)")
    a.set_title(title, loc="left", fontsize=10)
    a.legend(loc="upper right", fontsize=8)
    a.set_ylim(-6, 6)
ax[2].set_xlabel("time (s)")
fig.suptitle("An honest filter: the ± 2σ band should contain the error ~95 % of the time", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_error_vs_time.png", dpi=140)
plt.close(fig)

# 2) calibration: NIS CDF + NEES by scene quality
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs = np.linspace(0, 15, 400)
ax[0].plot(xs, stats.chi2.cdf(xs, 2), "k--", label="theory (χ², 2 dof)")
for n in names:
    v = np.sort(res[n]["nis"])
    ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[n], label=n)
ax[0].set_xlim(0, 15)
ax[0].set_xlabel("NIS")
ax[0].set_ylabel("cumulative distribution")
ax[0].set_title("NIS: the closer to theory, the more honest", fontsize=10)
ax[0].legend(fontsize=8)

w = 0.26
for j, n in enumerate(names):
    vals = [summary[n][f"nees[{label}]"] for label, _, _ in BINS]
    ax[1].bar(np.arange(3) + (j - 1) * w, vals, w, color=COL[n], label=n)
ax[1].axhline(4, color="k", ls="--", lw=1, label="ideal (4)")
ax[1].set_xticks(range(3))
ax[1].set_xticklabels([b[0] for b in BINS], fontsize=8)
ax[1].set_ylabel("mean NEES")
ax[1].set_title("NEES by scene quality", fontsize=10)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_calibration.png", dpi=140)
plt.close(fig)
print(f"\nFigures and metrics.json written to {OUT}")
