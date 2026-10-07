"""One figure: the cross-session drop, and EA's effect falling apart per subject."""
import json
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("results/raw.json"))
h1 = json.load(open("results/h1_test.json"))

by = collections.defaultdict(list)
for r in rows:
    by[r["subject"]].append(r)
subs = sorted(by)

within = np.array([np.mean([r["within"] for r in by[s]]) for s in subs])
cross = np.array([np.mean([r["cross"] for r in by[s]]) for s in subs])
delta = np.array([[r["cross_ea"] - r["cross"] for r in by[s]] for s in subs])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- left: same-day vs different-day ---
x = np.arange(len(subs))
ax1.bar(x - 0.19, within, 0.38, label="within session (same day)", color="#4C72B0")
ax1.bar(x + 0.19, cross, 0.38, label="cross session (different day)", color="#DD8452")
ax1.axhline(0.25, ls=":", c="grey", lw=1)
ax1.text(len(subs) - 0.5, 0.262, "chance", fontsize=8, color="grey", ha="right")
ax1.set_xticks(x); ax1.set_xticklabels(subs, fontsize=8)
ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 0.9)
ax1.set_title(f"Cross-session drop: {within.mean():.3f} → {cross.mean():.3f} "
              f"({(cross-within).mean()*100:+.1f} pts)", fontsize=10)
ax1.legend(fontsize=8, loc="upper left"); ax1.spines[["top", "right"]].set_visible(False)

# --- right: EA effect per subject ---
order = np.argsort(delta.mean(axis=1))
m = delta.mean(axis=1)[order]
colors = ["#55A868" if v > 0 else "#C44E52" for v in m]
ax2.barh(np.arange(len(subs)), m, color=colors, alpha=0.75)
for i, j in enumerate(order):
    ax2.scatter(delta[j], [i] * delta.shape[1], c="k", s=9, zorder=3)
ax2.axvline(0, c="k", lw=1)
ax2.axvline(h1["mean_delta"], c="#4C72B0", ls="--", lw=1.5,
            label=f"mean {h1['mean_delta']*100:+.1f} pts")
ax2.set_yticks(np.arange(len(subs)))
ax2.set_yticklabels([subs[j] for j in order], fontsize=8)
ax2.set_xlabel("Δ accuracy from Euclidean Alignment")
ax2.set_title(f"EA helps {h1['helped']}, hurts {h1['hurt']}  |  "
              f"F={h1['F']:.2f}, p={h1['p']:.4f}, ICC={h1['icc']:.2f}", fontsize=10)
ax2.legend(fontsize=8, loc="lower right"); ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("BCI IV-2a: adaptation that does nothing on average, "
             "and a lot per person", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig("results/figure.png", dpi=180, bbox_inches="tight")
print("wrote results/figure.png")
