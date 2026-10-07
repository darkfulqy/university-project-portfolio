"""Is EA's per-subject effect real, or is it seed noise wearing a costume?

ΔEA is paired within a seed (same seed for the adapted and unadapted run), so
the between-subject spread can be compared against the seed-to-seed spread of
the same quantity. That comparison is the whole point: a method that helps on
average is uninteresting if the average hides signs flipping per person.
"""
import json
import collections
import numpy as np

rows = json.load(open("results/raw.json"))

by = collections.defaultdict(list)
for r in rows:
    by[r["subject"]].append(r["cross_ea"] - r["cross"])   # paired within seed

subs = sorted(by)
D = np.array([by[s] for s in subs])          # (n_subjects, n_seeds)
n_sub, n_seed = D.shape

subj_mean = D.mean(axis=1)
grand = D.mean()

# one-way ANOVA: subject as the grouping factor
ss_between = n_seed * ((subj_mean - grand) ** 2).sum()
ss_within = ((D - subj_mean[:, None]) ** 2).sum()
df_b, df_w = n_sub - 1, n_sub * (n_seed - 1)
ms_b, ms_w = ss_between / df_b, ss_within / df_w
F = ms_b / ms_w

# variance components and the intraclass correlation
var_between = max((ms_b - ms_w) / n_seed, 0.0)
icc = var_between / (var_between + ms_w) if (var_between + ms_w) > 0 else 0.0

try:
    from scipy import stats
    p = 1 - stats.f.cdf(F, df_b, df_w)
except ImportError:
    p = float("nan")

print("=" * 70)
print("Does EA's effect differ by subject, beyond seed noise?")
print("=" * 70)
print(f"subjects {n_sub}, seeds each {n_seed}\n")
print(f"{'subject':>8} {'ΔEA mean':>10} {'seed sd':>9}   per-seed ΔEA")
print("-" * 70)
for s, d in zip(subs, D):
    print(f"{s:>8} {d.mean():+10.3f} {d.std(ddof=1):9.3f}   "
          + "  ".join(f"{v:+.3f}" for v in d))
print("-" * 70)
print(f"{'overall':>8} {grand:+10.3f}\n")

print(f"between-subject sd of ΔEA : {subj_mean.std(ddof=1):.4f}")
print(f"within-subject (seed) sd  : {np.sqrt(ms_w):.4f}")
print(f"F({df_b},{df_w}) = {F:.2f},  p = {p:.4f}")
print(f"ICC (subject share of ΔEA variance) = {icc:.2f}")
print()

helped = (subj_mean > 0).sum()
hurt = (subj_mean < 0).sum()
print(f"helped {helped}/{n_sub}   hurt {hurt}/{n_sub}   "
      f"range {subj_mean.min():+.3f} to {subj_mean.max():+.3f}")
print()

if p < 0.05:
    print("Read: subject identity explains a real share of ΔEA. Whether to apply\n"
          "EA is a per-subject decision, not a fixed recipe — which is what makes\n"
          "'when should we NOT adapt' a question worth asking.")
else:
    print("Read: with 9 subjects and 3 seeds the between-subject spread is not\n"
          "separable from seed noise. The heterogeneity is suggestive, not\n"
          "established; a longitudinal dataset with more subjects and sessions\n"
          "is needed before claiming it.")
print("=" * 70)

json.dump(dict(n_subjects=int(n_sub), n_seeds=int(n_seed),
               mean_delta=float(grand), between_sd=float(subj_mean.std(ddof=1)),
               within_sd=float(np.sqrt(ms_w)), F=float(F), p=float(p),
               icc=float(icc), helped=int(helped), hurt=int(hurt),
               per_subject={s: float(m) for s, m in zip(subs, subj_mean)}),
          open("results/h1_test.json", "w"), indent=1)
