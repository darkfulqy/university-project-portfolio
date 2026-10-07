"""Do adaptation families differ, and does the best family depend on the subject?

Two questions, in order:
  1. Is there a main effect of action? (does any method beat doing nothing)
  2. Is there a subject x action interaction? (does the best method depend on who)

Question 2 is the one that matters. A main effect says "use method X". An
interaction says "it depends" -- and if it depends, something has to decide,
which is the whole motivation for drift-aware action selection.
"""
import json
import itertools
import numpy as np

ACTIONS = [("A0_none", "No adapt"), ("A1_ea", "EA"), ("A2_adabn", "AdaBN"),
           ("A3_t3a", "T3A"), ("A4_tent", "Tent")]

rows = []
for p in ["hit1", "hit2", "hit3"]:
    rows += json.load(open(f"results/{p}.json"))

subs = sorted({r["subject"] for r in rows})
seeds = sorted({r["seed"] for r in rows})
keys = [k for k, _ in ACTIONS]

# Y[subject, action, seed]
Y = np.array([[[next(r[a] for r in rows if r["subject"] == s and r["seed"] == d)
                for d in seeds] for a in keys] for s in subs])
n_s, n_a, n_d = Y.shape

print("=" * 78)
print("Five adaptation actions from one frozen source model")
print("=" * 78)
print(f"{'subject':>8} " + " ".join(f"{n:>10}" for _, n in ACTIONS) + "   best")
print("-" * 78)
M = Y.mean(axis=2)                                   # (subject, action)
for i, s in enumerate(subs):
    best = ACTIONS[int(M[i].argmax())][1]
    print(f"{s:>8} " + " ".join(f"{v:10.3f}" for v in M[i]) + f"   {best}")
print("-" * 78)
print(f"{'mean':>8} " + " ".join(f"{v:10.3f}" for v in M.mean(axis=0)))
print("=" * 78)

# ---- how often does each action win, and how often does it hurt? ----
print("\nPer-action summary, relative to doing nothing (POINT ESTIMATES):")
base = M[:, 0]
for j, (k, name) in enumerate(ACTIONS):
    if j == 0:
        continue
    d = M[:, j] - base
    # point estimates only -- see analyze_bootstrap.py for which of these
    # actually clear zero once trial sampling error is accounted for
    print(f"  {name:>8}: mean {d.mean()*100:+5.1f} pts | "
          f"positive {int((d > 0).sum())}/{n_s}, negative {int((d < 0).sum())}/{n_s} | "
          f"range {d.min()*100:+5.1f} to {d.max()*100:+5.1f}")

wins = [ACTIONS[int(M[i].argmax())][1] for i in range(n_s)]
print("\nBest action per subject:")
for name in {w for w in wins}:
    print(f"  {name:>8}: {wins.count(name)}/{n_s} subjects")

# ---- two-way ANOVA: subject x action, seeds as replicates ----
grand = Y.mean()
ss_sub = n_a * n_d * ((Y.mean(axis=(1, 2)) - grand) ** 2).sum()
ss_act = n_s * n_d * ((Y.mean(axis=(0, 2)) - grand) ** 2).sum()
cell = Y.mean(axis=2)
ss_int = n_d * ((cell - Y.mean(axis=(1, 2))[:, None]
                 - Y.mean(axis=(0, 2))[None, :] + grand) ** 2).sum()
ss_err = ((Y - cell[:, :, None]) ** 2).sum()

df_sub, df_act = n_s - 1, n_a - 1
df_int = df_sub * df_act
df_err = n_s * n_a * (n_d - 1)
ms_err = ss_err / df_err
F_act = (ss_act / df_act) / ms_err
F_int = (ss_int / df_int) / ms_err

try:
    from scipy import stats
    p_act = 1 - stats.f.cdf(F_act, df_act, df_err)
    p_int = 1 - stats.f.cdf(F_int, df_int, df_err)
except ImportError:
    p_act = p_int = float("nan")

print("\n" + "=" * 78)
print("Two-way ANOVA (subject x action, seeds as replicates)")
print("-" * 78)
print(f"  main effect of ACTION            F({df_act},{df_err}) = {F_act:6.2f}   p = {p_act:.4f}")
print(f"  SUBJECT x ACTION interaction     F({df_int},{df_err}) = {F_int:6.2f}   p = {p_int:.4f}")
print(f"  residual (seed) sd               {np.sqrt(ms_err):.4f}")
print("=" * 78)

var_int = max((ss_int / df_int - ms_err) / n_d, 0)
var_err = ms_err
print(f"\ninteraction variance share: {var_int/(var_int+var_err):.2f}")

print()
if p_int < 0.05:
    print("H1 holds. Which adaptation is best depends on the subject by more than\n"
          "seed noise, so a single fixed recipe is leaving accuracy on the table --\n"
          "and for some people the right action is to leave the model alone.")
else:
    print("H1 not established here. The action ranking looks stable across subjects\n"
          "at this sample size; a per-subject selection rule would have nothing to do.")

json.dump(dict(
    subjects=subs, actions=[n for _, n in ACTIONS],
    mean_by_subject_action=M.tolist(),
    mean_by_action=M.mean(axis=0).tolist(),
    best_per_subject=wins,
    F_action=float(F_act), p_action=float(p_act),
    F_interaction=float(F_int), p_interaction=float(p_int),
    seed_sd=float(np.sqrt(ms_err)),
    interaction_var_share=float(var_int / (var_int + var_err)),
), open("results/families_test.json", "w"), indent=1)
