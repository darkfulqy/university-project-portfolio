"""The money figure: no action wins on average, but the winner changes per person."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.load(open("results/families_test.json"))
subs, actions = h["subjects"], h["actions"]
M = np.array(h["mean_by_subject_action"])          # (subject, action)
base = M[:, 0]
D = (M - base[:, None]) * 100                      # points relative to no-adapt

COL = {"No adapt": "#888888", "EA": "#4C72B0", "AdaBN": "#DD8452",
       "T3A": "#55A868", "Tent": "#C44E52"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# --- left: the average, which says nothing ---
means = D[:, 1:].mean(axis=0)
ax1.bar(range(1, len(actions)), means,
        color=[COL[a] for a in actions[1:]], alpha=0.85)
for i, a in enumerate(actions[1:], start=1):
    ax1.scatter([i] * len(subs), D[:, i], c="k", s=14, zorder=3, alpha=0.7)
ax1.axhline(0, c="k", lw=1.2)
ax1.set_xticks(range(1, len(actions)))
ax1.set_xticklabels(actions[1:])
ax1.set_ylabel("Δ accuracy vs. no adaptation (points)")
ax1.set_title(f"No main effect of action\nF(4,90)={h['F_action']:.2f}, "
              f"p={h['p_action']:.2f}", fontsize=10.5)
ax1.spines[["top", "right"]].set_visible(False)

# --- right: per subject, which action actually wins ---
x = np.arange(len(subs))
w = 0.16
for j, a in enumerate(actions[1:], start=1):
    ax2.bar(x + (j - 2.5) * w, D[:, j], w, label=a, color=COL[a], alpha=0.9)
best = [actions[int(M[i].argmax())] for i in range(len(subs))]
for i, b in enumerate(best):
    ax2.text(i, D[i].max() + 0.6, b, ha="center", fontsize=7,
             color=COL[b], fontweight="bold")
ax2.axhline(0, c="k", lw=1.2)
ax2.set_xticks(x); ax2.set_xticklabels(subs, fontsize=8)
ax2.set_ylabel("Δ accuracy vs. no adaptation (points)")
ax2.set_ylim(D.min() - 2, D.max() + 3.5)
ax2.set_title(f"But the winner depends on the subject\n"
              f"interaction F(32,90)={h['F_interaction']:.2f}, "
              f"p={h['p_interaction']:.4f}", fontsize=10.5)
ax2.legend(fontsize=8, ncol=4, loc="lower left")
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("BCI IV-2a cross-session: averaging over subjects erases the problem",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("results/figure_families.png", dpi=180, bbox_inches="tight")
print("wrote results/figure_families.png")
