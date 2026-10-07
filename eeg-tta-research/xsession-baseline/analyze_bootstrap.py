"""Per-subject adaptation effects with trial-level uncertainty.

The earlier analysis treated random seeds as replicates. Seeds are not
independent samples of anything biological -- every seed scores the *same*
test trials, so seed variance misses the sampling error that comes from having
a finite number of trials. That error is the dominant one here.

This redoes it properly: bootstrap over test trials, paired within a trial
(both actions predicted the same trial, so their errors are correlated and the
paired difference is tighter than two independent binomials would suggest).
A subject only counts as helped or hurt if its 95% CI clears zero.
"""
import json
import collections
import numpy as np

ACTIONS = [("A0_none", "No adapt"), ("A1_ea", "EA"), ("A2_adabn", "AdaBN"),
           ("A3_t3a", "T3A"), ("A4_tent", "Tent")]
B = 10000
RNG = np.random.default_rng(0)

rows = []
for p in ["hit1", "hit2", "hit3"]:
    try:
        rows += json.load(open(f"results/{p}.json"))
    except FileNotFoundError:
        pass
if not rows:
    raise SystemExit("no results/hit*.json — run run_families.py first")

by = collections.defaultdict(list)
for r in rows:
    by[r["subject"]].append(r)
subs = sorted(by)

print("=" * 88)
print("Per-subject Δ vs. no adaptation, with trial-level bootstrap 95% CI")
print(f"({B} resamples, paired within trial, seeds averaged)")
print("=" * 88)

verdicts = {name: {"helped": 0, "hurt": 0, "unclear": 0} for _, name in ACTIONS[1:]}
table = {}

for sid in subs:
    rs = by[sid]
    # average the per-trial hit vectors over seeds -> per-trial success rate
    H = {k: np.mean([r["_hit"][k] for r in rs], axis=0) for k, _ in ACTIONS}
    n = len(H["A0_none"])
    idx = RNG.integers(0, n, size=(B, n))          # one resample shared by all actions

    print(f"\n{sid}  (n = {n} test trials)")
    table[sid] = {}
    for k, name in ACTIONS[1:]:
        d = H[k] - H["A0_none"]                    # paired per-trial difference
        obs = d.mean() * 100
        boot = d[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if lo > 0:
            v, mark = "helped", "  ✓ helps"
        elif hi < 0:
            v, mark = "hurt", "  ✗ hurts"
        else:
            v, mark = "unclear", "    —"
        verdicts[name][v] += 1
        table[sid][name] = dict(delta=obs, lo=lo, hi=hi, verdict=v)
        print(f"    {name:>9}: {obs:+6.1f} pts   95% CI [{lo:+6.1f}, {hi:+6.1f}]{mark}")

print("\n" + "=" * 88)
print("Summary after trial-level uncertainty")
print("-" * 88)
print(f"{'action':>10} {'helps':>7} {'hurts':>7} {'unclear':>9}   mean Δ")
for k, name in ACTIONS[1:]:
    v = verdicts[name]
    mean_d = np.mean([table[s][name]["delta"] for s in subs])
    print(f"{name:>10} {v['helped']:>7} {v['hurt']:>7} {v['unclear']:>9}   {mean_d:+.1f} pts")
print("=" * 88)

# Does any subject have a *decisively* different best action?
print("\nSubjects where at least one action's CI clears zero:")
decisive = [s for s in subs
            if any(table[s][n]["verdict"] != "unclear" for _, n in ACTIONS[1:])]
print(f"  {len(decisive)}/{len(subs)}: {', '.join(decisive) if decisive else 'none'}")

n_sign_split = sum(
    1 for _, name in ACTIONS[1:]
    if verdicts[name]["helped"] > 0 and verdicts[name]["hurt"] > 0)
print(f"\nActions that provably help some subjects AND provably hurt others: "
      f"{n_sign_split}/{len(ACTIONS)-1}")
if n_sign_split:
    print("  -> The sign of the effect flips across people with trial-level\n"
          "     confidence. That is the claim worth making; it does not depend\n"
          "     on treating seeds as replicates.")
else:
    print("  -> No action both helps and hurts with confidence at this sample\n"
          "     size. The heterogeneity is suggestive, not established. Say so.")

json.dump(table, open("results/bootstrap_ci.json", "w"), indent=1)
