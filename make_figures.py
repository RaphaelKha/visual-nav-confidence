"""Figures for the EuRoC results (reads the JSON outputs of run_euroc_experiment.py and run_failure_prediction.py).

fig3_euroc_nees_by_inliers.png: position NEES of the fixed / learned filter by number of PnP inliers (ideal 3).
fig4_euroc_failure_prediction.png: ROC AUC of failure / outlier prediction from the quality indicators.
Usage: python make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parent / "results"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated: CVD and contrast checks pass)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def bars(ax, groups, series, colors, ref=None, ref_label=None, fmt="{:.2f}"):
    x = np.arange(len(groups))
    w = 0.36
    for j, ((name, vals), c) in enumerate(zip(series.items(), colors)):
        pos = x + (j - (len(series) - 1) / 2) * (w + 0.03)      # small surface gap between adjacent bars
        ax.bar(pos, vals, w, color=c, label=name, zorder=2)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.04 * ax.get_ylim()[1], fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    if ref is not None:
        ax.axhline(ref, color=INK, lw=1, ls="--", zorder=3, label=ref_label)
    ax.set_xticks(x, groups)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (d, title) in zip(axes, (("euroc_exp", "Clean images"), ("euroc_exp_degraded", "Blurred / darkened images"))):
    m = json.loads((R / d / "metrics.json").read_text())["nees_p_by_inliers_tercile"]
    ax.set_ylim(0, 5.5)
    bars(ax, ["few inliers", "medium", "many inliers"], {"fixed R": m["fixed"], "learned R": m["learned"]},
         [BLUE, ORANGE], ref=3, ref_label="ideal = 3 (honest)")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
axes[0].set_ylabel("position NEES (mean)")
axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle("EuRoC, real IMU + real relocalisation, 5 test flights (leave-one-out):\n"
             "is the announced uncertainty honest when few points match?", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.text(0.01, -0.02, "Above 3 = overconfident, below 3 = too cautious. Terciles of the number of PnP inliers, pooled "
         "over MH_01..MH_05. Same-room maps (different flights).", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(R / "fig3_euroc_nees_by_inliers.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- fig 4
deg = json.loads((R / "failure_prediction_degraded.json").read_text())
cln = json.loads((R / "failure_prediction_clean.json").read_text())
keys = [("failure_image_only", "failure,\nimage only"), ("failure_with_matches", "failure,\n+ matches"),
        ("outlier_all", "outlier > 0.2 m,\nall indicators"), ("outlier_inliers_only", "outlier > 0.2 m,\ninliers only")]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.set_ylim(0, 1.05)
bars(ax, [k[1] for k in keys], {"clean images": [cln[k]["pooled"]["auc"] for k, _ in keys],
                                "blurred / darkened": [deg[k]["pooled"]["auc"] for k, _ in keys]},
     [BLUE, ORANGE], ref=0.5, ref_label="chance = 0.5")
ax.set_ylabel("ROC AUC (pooled over 5 test flights)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5)
fig.suptitle("Do the quality indicators predict relocalisation failures and outliers?\n"
             "(trained on 4 flights, tested on the 5th, leave-one-out)", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout()
fig.savefig(R / "fig4_euroc_failure_prediction.png", dpi=150, bbox_inches="tight")
print("saved fig3_euroc_nees_by_inliers.png, fig4_euroc_failure_prediction.png")
