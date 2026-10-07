"""Forest plot: per-subject adaptation effects with trial-level 95% CIs.

Every claim in this figure is bootstrapped over test trials. Bars whose interval
crosses zero are drawn hollow -- they are the honest majority, and hiding them
would be the whole problem this figure exists to expose.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ci = json.load(open("results/bootstrap_ci.json"))
fam = json.load(open("results/families_test.json"))
subs = sorted(ci)
actions = ["EA", "AdaBN", "T3A", "Tent"]
COL = {"EA": "#4C72B0", "AdaBN": "#DD8452", "T3A": "#55A868", "Tent": "#C44E52"}

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, a in zip(axes, actions):
    ys = np.arange(len(subs))
    for i, s in enumerate(subs):
        d = ci[s][a]
        solid = d["verdict"] != "unclear"
        ax.plot([d["lo"], d["hi"]], [i, i], color=COL[a],
                lw=2.4 if solid else 1.2, alpha=1.0 if solid else 0.45,
                solid_capstyle="round")
        ax.scatter([d["delta"]], [i], s=46 if solid else 26,
                   facecolor=COL[a] if solid else "white",
                   edgecolor=COL[a], zorder=3, linewidths=1.6)
    ax.axvline(0, color="k", lw=1.1)
    mean_d = np.mean([ci[s][a]["delta"] for s in subs])
    ax.axvline(mean_d, color=COL[a], ls="--", lw=1.3, alpha=0.8)
    n_help = sum(1 for s in subs if ci[s][a]["verdict"] == "helped")
    n_hurt = sum(1 for s in subs if ci[s][a]["verdict"] == "hurt")
    ax.set_title(f"{a}\nmean {mean_d:+.1f} pts · "
                 f"{n_help} confirmed help, {n_hurt} harm", fontsize=10)
    ax.set_xlabel("Δ accuracy vs. no adaptation (points)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-14, 30)

axes[0].set_yticks(np.arange(len(subs)))
axes[0].set_yticklabels(subs, fontsize=9)
axes[0].invert_yaxis()

fig.suptitle(
    "BCI IV-2a cross-session: no action wins on average "
    f"(F(4,90)={fam['F_action']:.2f}, p={fam['p_action']:.2f}), "
    "but the spread dwarfs the mean\n"
    "filled = 95% CI excludes zero · hollow = indistinguishable from doing nothing "
    "· dashed = mean across subjects",
    fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig("results/figure_forest.png", dpi=180, bbox_inches="tight")
print("wrote results/figure_forest.png")
